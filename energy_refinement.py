
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from typing import Optional, Dict
from phi_metrics_neuro import phi_synergy_features, kappa_floor, cge_boost, clamp_neuromorphic_terms

class EnergyRefiner(nn.Module):
    # Iterative "System-2" refinement head with temperature annealing and early stopping.
    # Minimizes E = L_fit + Î»_viol * L_violation + Î»_prior * (Î¦ + Îº + CGE + Hodge).
    def __init__(self, min_steps:int=3, max_steps:int=7, step_size:float=0.25, noise:float=0.0,
                 lambda_violation:float=1.0, lambda_prior:float=1e-3, lambda_size:float=0.0,
                 lambda_spike:float=0.0,  # Spike budget penalty weight
                 w_phi:float=0.1, w_kappa:float=0.1, w_cge:float=0.1,
                 temp_schedule=None, early_stop_threshold:float=1e-4, verbose:bool=False,
                 strict: bool = True):
        super().__init__()
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.step_size = step_size
        self.noise = noise
        self.lambda_violation = lambda_violation
        self.lambda_prior = lambda_prior
        self.lambda_size = lambda_size
        self.lambda_spike = lambda_spike
        self.w_phi = w_phi
        self.w_kappa = w_kappa
        self.w_cge = w_cge
        self.early_stop_threshold = early_stop_threshold
        self.verbose = verbose
        # strict mode -> fail loud and avoid silent fallbacks (set False for permissive debug runs)
        self.strict = bool(strict)
        
        # Temperature annealing schedule
        if temp_schedule is None:
            self.temp_schedule = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1]
        else:
            self.temp_schedule = temp_schedule

    def forward(self, logits: torch.Tensor, constraint_obj, prior_tensors: dict,
                prior_scales: Optional[Dict[str, float]] = None,
                extras: Optional[Dict] = None):
        # Activate only when real priors exist (no dummies)
        def _pri_ready(p):
            try:
                return (isinstance(p, dict) and
                        all(k in p for k in ("phi","kappa","cge")))
            except Exception:
                return False
        if not _pri_ready(prior_tensors):
            raise RuntimeError("[EBR] priors missing â€“ refinement cannot proceed")

        # logits: Float tensor, e.g., [B, C, H, W]; must be differentiable
        # constraint_obj must expose: fit_loss(logits), violation_loss(logits)
        # prior_tensors: dict of optional scalar tensors/floats: {'phi','kappa','cge','hodge'}
        # make a leaf tensor that can get grads, regardless of upstream context
        x = logits.detach().clone().requires_grad_(True)

        # Apply prior scaling with safe defaults
        if prior_scales is None:
            prior_scales = {"phi": 1.0, "kappa": 1.0, "cge": 1.0}
            
        # ensure autograd is ON inside refinement (eval often uses no_grad)
        with torch.enable_grad():
            for step in range(self.max_steps):
                # Anneal softmax temperature for better convergence
                temp_idx = min(step, len(self.temp_schedule) - 1)
                temp = self.temp_schedule[temp_idx]
                
                # Apply temperature to logits
                x_temp = x / temp
                
                fit = constraint_obj.fit_loss(x_temp)
                viol = constraint_obj.violation_loss(x_temp)
                # convert to python floats if they are 0-dim tensors for comparisons/logging
                try:
                    fit_scalar = float(fit.item()) if torch.is_tensor(fit) else float(fit)
                except Exception:
                    fit_scalar = float(fit) if not torch.is_tensor(fit) else float(fit.detach().cpu().item())
                try:
                    viol_scalar = float(viol.item()) if torch.is_tensor(viol) else float(viol)
                except Exception:
                    viol_scalar = float(viol) if not torch.is_tensor(viol) else float(viol.detach().cpu().item())

                # Early stopping if violation is low enough and we've done minimum steps
                if step >= self.min_steps and viol_scalar < float(self.early_stop_threshold):
                    logger = logging.getLogger(__name__)
                    logger.debug(f"[EBR] Early stopping at step {step}, violation={viol_scalar:.6e}")
                    break
                
                # Derive "soft tokens" from x_temp: [B, C, H, W] -> [B, H*W, C]
                B, C, H, W = x_temp.shape
                soft = torch.softmax(x_temp, dim=1).permute(0,2,3,1).reshape(B, H*W, C)
                
                # Compute differentiable Ï†/Îº/CGE penalties w.r.t. x
                phi = phi_synergy_features(soft, parts=2)
                kappa = kappa_floor(soft, H, W)
                cge = cge_boost(soft)
                
                # Clamp neuromorphic terms to prevent divergence
                phi, kappa, cge = clamp_neuromorphic_terms(phi, kappa, cge)
                
                # Include hodge term from prior_tensors if provided
                hodge = prior_tensors.get("hodge", 0.0)
                hodge_term = (hodge.mean().item() if torch.is_tensor(hodge) else float(hodge))
                
                # Scale internal priors by prior_scales
                w_phi_scaled = self.w_phi * prior_scales.get("phi", 1.0)
                w_kappa_scaled = self.w_kappa * prior_scales.get("kappa", 1.0)
                w_cge_scaled = self.w_cge * prior_scales.get("cge", 1.0)
                
                # Form weighted prior term: pri = wÏ†*Ï† + wÎº*Îº + wCGE*CGE + hodge
                pri = w_phi_scaled * phi + w_kappa_scaled * kappa + w_cge_scaled * cge + hodge_term
                
                # Add size constraint loss
                size_constraints = prior_tensors.get("size_constraints", None)
                size_loss = torch.tensor(0.0, device=x.device)
                if size_constraints and self.lambda_size > 0:
                    size_loss = self._compute_size_loss(x_temp, size_constraints)

                # Add spike budget term if spikes are provided
                spike_loss = torch.tensor(0.0, device=x.device)
                if self.lambda_spike > 0:
                    # Check for spike tensors in extras or prior_tensors
                    spikes_dict = extras if extras else {}
                    z_H_spikes = spikes_dict.get('z_H_spikes', prior_tensors.get('z_H_spikes', None))
                    z_L_spikes = spikes_dict.get('z_L_spikes', prior_tensors.get('z_L_spikes', None))

                    if z_H_spikes is not None or z_L_spikes is not None:
                        spike_penalty = torch.tensor(0.0, device=x.device)
                        if z_H_spikes is not None:
                            spike_penalty += z_H_spikes.abs().float().mean()
                        if z_L_spikes is not None:
                            spike_penalty += z_L_spikes.abs().float().mean()
                        spike_loss = spike_penalty / (2.0 if z_H_spikes is not None and z_L_spikes is not None else 1.0)

                E = fit + self.lambda_violation * viol + self.lambda_prior * pri + self.lambda_size * size_loss + self.lambda_spike * spike_loss
                
                # Debug print for first step to verify priors influence
                # Debug logging if verbose
                if step == 0 and self.verbose:
                    logger = logging.getLogger(__name__)
                    logger.debug(f"[EBR] step={step}, temp={temp:.2f}, fit={fit_scalar:.6f}, viol={viol_scalar:.6f}, size={float(size_loss):.6f}")
                    logger.debug(f"[EBR] phi={float(phi):.6f}, kappa={float(kappa):.6f}, cge={float(cge):.6f}, pri={float(pri):.6f}")
                
                # Backward pass: try a normal backward (no retain_graph) first; if user specifically needs
                # an additional backward, they must pass a graph-preserving flag or compute gradients elsewhere.
                try:
                    E.backward()
                except RuntimeError as e:
                    # If graph has been freed or second backward requested, surface it under strict mode.
                    msg = str(e)
                    if "backward through the graph a second time" in msg or "element 0 of tensors does not require grad" in msg:
                        if self.strict:
                            raise RuntimeError(f"[EBR] backward failed at step {step}: {msg}") from e
                        else:
                            # permissive: break refinement loop and log
                            logging.getLogger(__name__).warning(f"[EBR] permissive: backward failed at step {step}: {msg}")
                            break
                    else:
                        raise

                with torch.no_grad():
                    # Use temperature-adjusted step size for better convergence
                    adjusted_step_size = self.step_size * temp
                    # ensure gradient exists
                    if x.grad is None:
                        if self.strict:
                            raise RuntimeError(f"[EBR] expected gradient on refinement tensor but found None at step {step}")
                        else:
                            logging.getLogger(__name__).warning(f"[EBR] missing x.grad at step {step}; skipping update")
                    else:
                        x -= adjusted_step_size * x.grad
                    if self.noise > 0:
                        x += self.noise * torch.randn_like(x, device=x.device)
                    if x.grad is not None:
                        x.grad.zero_()

        return x.detach()
    
    def _compute_size_loss(self, logits: torch.Tensor, size_constraints: dict) -> torch.Tensor:
        """Compute size constraint loss based on predicted vs actual output size"""
        if not size_constraints or 'predicted_size' not in size_constraints:
            return torch.tensor(0.0, device=logits.device)
        
        predicted_size = size_constraints['predicted_size']
        confidence = size_constraints.get('confidence', 1.0)
        
        # Current logits size
        B, C, H, W = logits.shape
        actual_size = (H, W)
        
        # Size mismatch penalty
        if actual_size != predicted_size:
            # L1 distance between sizes, weighted by confidence
            size_diff = abs(actual_size[0] - predicted_size[0]) + abs(actual_size[1] - predicted_size[1])
            size_penalty = confidence * size_diff * 0.1
            
            if self.verbose:
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"[EBR] Size mismatch: actual={actual_size}, predicted={predicted_size}, penalty={size_penalty:.4f}")
            
            return torch.tensor(size_penalty, device=logits.device)
        
        # Size histogram matching if predicted size matches
        if 'demos' in size_constraints:
            hist_demos = size_constraints['demos']
            # In strict mode, bubble up histogram computation errors instead of silently returning zero.
            try:
                hist_loss = self._compute_histogram_loss(logits, hist_demos)
                return hist_loss * confidence * 0.05  # Small weight for histogram matching
            except Exception as e:
                if self.strict:
                    raise RuntimeError(f"[EBR] histogram loss failed: {e}") from e
                else:
                    logging.getLogger(__name__).warning(f"[EBR] permissive: histogram loss failed: {e}")
        return torch.tensor(0.0, device=logits.device)
    
    def _compute_histogram_loss(self, logits: torch.Tensor, demos) -> torch.Tensor:
        """Compute histogram matching loss for size-correct outputs"""
        if not demos:
            return torch.tensor(0.0, device=logits.device)
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1)  # [B, C, H, W]
        pred_hist = probs.mean(dim=(2, 3))    # [B, C] - average color distribution

        C = logits.size(1)
        target_hists = []
        for demo in demos:
            # Expect demo to be an (input, output) pair or an output tensor
            output = None
            if isinstance(demo, (tuple, list)) and len(demo) >= 2:
                output = demo[1]
            elif isinstance(demo, torch.Tensor) or isinstance(demo, np.ndarray):
                output = demo
            else:
                # skip invalid demo entry
                continue

            # Bring output into torch tensor form on same device as logits
            if isinstance(output, torch.Tensor):
                out_t = output.detach().to(logits.device)
            elif isinstance(output, np.ndarray):
                out_t = torch.from_numpy(output).to(logits.device)
            else:
                if self.strict:
                    raise RuntimeError("[EBR] demo output has unsupported type for histogramming")
                else:
                    continue

            # If output contains channel dimension (C,H,W) or (H,W,C), attempt to reduce to integer labels
            # We expect integer class labels in [0, C-1]. Try to handle common shapes.
            if out_t.dim() == 3 and out_t.shape[0] == C:
                # either (C,H,W) probabilities -> take argmax across channels
                out_labels = out_t.argmax(dim=0).flatten().long()
            elif out_t.dim() == 2:
                # (H,W) integer labels
                out_labels = out_t.flatten().long()
            elif out_t.dim() == 3 and out_t.shape[-1] == C:
                # (H,W,C) channel-last probabilities -> argmax
                out_labels = out_t.argmax(dim=-1).flatten().long()
            else:
                if self.strict:
                    raise RuntimeError(f"[EBR] demo output shape {tuple(out_t.shape)} not supported for histogram loss")
                else:
                    # permissive: skip this demo
                    continue

            if out_labels.numel() == 0:
                continue

            # ensure minlength equals C
            hist = torch.bincount(out_labels, minlength=C).float()
            total = hist.sum().clamp(min=1e-8)
            hist = hist / total
            target_hists.append(hist.to(logits.device))

        if not target_hists:
            if self.strict:
                raise RuntimeError("[EBR] no valid demo histograms found for histogram loss")
            else:
                logging.getLogger(__name__).warning("[EBR] permissive: no demo histograms found; skipping hist loss")
                return torch.tensor(0.0, device=logits.device)

        target_hist = torch.stack(target_hists).mean(dim=0).unsqueeze(0)  # [1, C]
        target_hist = target_hist.expand(pred_hist.size(0), -1)  # [B, C]
        hist_loss = F.mse_loss(pred_hist, target_hist)
        return hist_loss