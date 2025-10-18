"""
Multi-Head Pretrainer - World Grammar Pretraining
5-head supervised pretraining system for TOPAS ARC solver
Learns factorized representations of ARC concept algebra
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Import TOPAS model
from models.topas_arc_60M import TopasARC60M, ModelConfig

@dataclass 
class PretrainConfig:
    """Configuration for multi-head pretraining"""
    # Loss weights for each head
    lambda_grid: float = 1.0          # Final grid prediction
    lambda_program: float = 0.8       # DSL program tokens  
    lambda_size: float = 0.6          # Size classification
    lambda_symmetry: float = 0.4      # Symmetry classification
    lambda_histogram: float = 0.3     # Color histogram changes
    
    # Head architectures
    program_vocab_size: int = 128     # DSL operation vocabulary
    max_program_length: int = 8       # Maximum program sequence length
    size_classes: int = 16            # Number of size transformation classes
    symmetry_classes: int = 8         # Number of symmetry types
    histogram_dim: int = 10           # Color histogram dimension (0-9)
    
    # Training
    use_teacher_forcing: bool = True  # Teacher forcing for program head
    program_dropout: float = 0.1      # Dropout in program head
    head_dropout: float = 0.2         # General head dropout
    
    # Validation
    validate_sacred_signature: bool = True  # Enforce sacred signature compliance

class MultiHeadPretrainer(nn.Module):
    """
    Multi-head supervised pretraining for world grammar learning
    
    Five specialized heads:
    (a) final_grid_head - Predict output grid directly  
    (b) program_tokens_head - Teacher-forced DSL sequence prediction
    (c) size_class_head - Classify size transformations (tile/scale/pad/crop)
    (d) symmetry_class_head - Classify symmetry operations (rotation/reflection/translation)
    (e) color_histogram_head - Predict color distribution changes
    """
    
    def __init__(self, base_model: TopasARC60M, config: PretrainConfig):
        super().__init__()

        # CRITICAL: Don't store base_model (creates circular reference)
        # topas_model.pretrainer.base_model → topas_model causes infinite recursion
        # Instead, store references to specific components we need

        self.config = config

        # Get model dimensions from TOPAS
        self.width = base_model.config.width  # Encoder width
        self.slot_dim = base_model.config.slot_dim  # Slot dimension

        # Control dimension = encoder width + pooled slots
        self.ctrl_dim = self.width + base_model.slots.out_dim

        print(f"[MultiHead] Control dimension: {self.ctrl_dim} (width={self.width} + slots={base_model.slots.out_dim})")

        # Store component REFERENCES (not whole model) to avoid circular parameter graph
        # These are not registered as nn.Module attributes, so won't appear in .parameters()
        self._encoder_ref = base_model.encoder
        self._slots_ref = base_model.slots
        self._reln_ref = base_model.reln

        # === HEAD (A): FINAL GRID HEAD ===
        # Don't use painter directly (creates circular ref)
        # Instead, we'll use our own heads for prediction
        # self.final_grid_head = base_model.painter  # REMOVED
        
        # === HEAD (B): PROGRAM TOKENS HEAD ===
        # Predict sequence of DSL operation tokens with teacher forcing
        self.program_embed = nn.Embedding(config.program_vocab_size, 128)
        self.program_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=8,
                dim_feedforward=512,
                dropout=config.program_dropout,
                batch_first=True
            ),
            num_layers=3
        )
        self.program_head = nn.Linear(self.ctrl_dim, config.program_vocab_size * config.max_program_length)
        
        # === HEAD (C): SIZE CLASS HEAD ===
        # Classify type of size transformation
        self.size_class_head = nn.Sequential(
            nn.Linear(self.ctrl_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(256, 128),
            nn.ReLU(), 
            nn.Dropout(config.head_dropout),
            nn.Linear(128, config.size_classes)
        )
        
        # === HEAD (D): SYMMETRY CLASS HEAD ===
        # Classify type of symmetry operation
        self.symmetry_class_head = nn.Sequential(
            nn.Linear(self.ctrl_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(128, config.symmetry_classes)
        )
        
        # === HEAD (E): COLOR HISTOGRAM HEAD ===
        # Predict change in color histogram distribution
        self.color_histogram_head = nn.Sequential(
            nn.Linear(self.ctrl_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(128, config.histogram_dim * 2)  # Before and after histograms
        )

        # === HEAD (F): CRITIC HEAD ===
        # Predict EM likelihood
        self.critic_head = nn.Linear(self.ctrl_dim, 1)

        # Operation vocabulary for program head
        self.op_vocab = self._build_operation_vocabulary()
        self.vocab_size = len(self.op_vocab)
        
        print(f"[MultiHead] Initialized with {self.vocab_size} operation vocabulary")
        print(f"[MultiHead] Heads: "
              f"program({config.max_program_length}), size({config.size_classes}), "
              f"symmetry({config.symmetry_classes}), histogram({config.histogram_dim})")
    
    def _build_operation_vocabulary(self) -> Dict[str, int]:
        """Build vocabulary mapping from DSL operations to token IDs"""
        # Core DSL operations from DSLHead
        operations = [
            '<PAD>', '<START>', '<END>', '<UNK>',  # Special tokens
            
            # Geometric transformations
            'rotate90', 'rotate180', 'rotate270', 'flip_h', 'flip_v', 
            'translate', 'scale', 'resize_nn', 'center_pad_to',
            
            # Color operations  
            'color_map', 'extract_color', 'mask_color', 'flood_fill',
            
            # Spatial operations
            'crop_bbox', 'crop_nonzero', 'paste', 'tile', 'tile_pattern',
            
            # Pattern operations
            'outline', 'symmetry', 'boundary_extract',
            
            # Counting and arithmetic
            'count_objects', 'count_colors', 'arithmetic_op',
            
            # Pattern matching
            'find_pattern', 'extract_pattern', 'match_template',
            
            # Conditional logic
            'apply_rule', 'conditional_map',
            
            # Grid algebra
            'grid_union', 'grid_intersection', 'grid_xor', 'grid_difference',
            
            # Advanced selection
            'flood_select', 'select_by_property',
            
            # Per-object operations
            'for_each_object', 'for_each_object_translate', 'for_each_object_recolor',
            'for_each_object_rotate', 'for_each_object_scale', 'for_each_object_flip',
            
            # Composite operations
            'repeat_n', 'if_else', 'while_condition',
            
            # Identity and utility
            'identity', 'no_op'
        ]
        
        # Build vocabulary dictionary
        vocab = {op: idx for idx, op in enumerate(operations)}
        
        # Ensure vocabulary fits in config
        if len(vocab) > self.config.program_vocab_size:
            print(f"[WARN] Operation vocabulary ({len(vocab)}) exceeds config size ({self.config.program_vocab_size})")
            # Truncate to fit
            vocab = {op: idx for op, idx in vocab.items() if idx < self.config.program_vocab_size}
        
        return vocab
    
    def encode_program(self, operations: List[str], params: List[Dict] = None) -> torch.Tensor:
        """
        Encode DSL program as token sequence for program head training
        
        Args:
            operations: List of operation names
            params: List of parameter dictionaries (optional)
            
        Returns:
            Token sequence tensor [max_program_length]
        """
        tokens = [self.op_vocab['<START>']]  # Start token
        
        for op in operations[:self.config.max_program_length - 2]:  # Leave room for START/END
            if op in self.op_vocab:
                tokens.append(self.op_vocab[op])
            else:
                tokens.append(self.op_vocab['<UNK>'])  # Unknown operation
        
        tokens.append(self.op_vocab['<END>'])  # End token
        
        # Pad to max length
        while len(tokens) < self.config.max_program_length:
            tokens.append(self.op_vocab['<PAD>'])
        
        return torch.tensor(tokens[:self.config.max_program_length])
    
    def classify_size_transformation(self, input_size: Tuple[int, int], output_size: Tuple[int, int]) -> int:
        """
        Classify the type of size transformation between input and output
        
        Returns class ID for size transformation type
        """
        h_in, w_in = input_size
        h_out, w_out = output_size
        
        # Calculate size ratios
        h_ratio = h_out / max(h_in, 1)
        w_ratio = w_out / max(w_in, 1)
        
        # Classify transformation type
        if h_ratio == 1.0 and w_ratio == 1.0:
            return 0  # No size change
        elif h_ratio > 1.0 and w_ratio > 1.0:
            if abs(h_ratio - w_ratio) < 0.1:
                return 1  # Uniform scaling up
            else:
                return 2  # Non-uniform scaling up
        elif h_ratio < 1.0 and w_ratio < 1.0:
            if abs(h_ratio - w_ratio) < 0.1:
                return 3  # Uniform scaling down  
            else:
                return 4  # Non-uniform scaling down
        elif h_ratio > 1.0 or w_ratio > 1.0:
            return 5  # Padding/extension
        else:
            return 6  # Cropping/reduction
    
    def classify_symmetry_operation(self, operations: List[str]) -> int:
        """
        Classify the primary symmetry operation in the program
        
        Returns class ID for symmetry type
        """
        # Priority order for classification
        symmetry_ops = {
            'rotate90': 1,
            'rotate180': 2, 
            'rotate270': 3,
            'flip_h': 4,
            'flip_v': 5,
            'translate': 6,
            'identity': 7
        }
        
        # Find highest priority symmetry operation
        for op in operations:
            if op in symmetry_ops:
                return symmetry_ops[op]
        
        return 0  # No symmetry operation
    
    def compute_color_histogram(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized color histogram for grid
        
        Args:
            grid: Input grid tensor [H, W]
            
        Returns:
            Histogram tensor [10] for colors 0-9
        """
        if grid.numel() == 0:
            return torch.zeros(10, device=grid.device)
        
        # Clamp to valid color range
        grid_clamped = torch.clamp(grid.long(), 0, 9)
        
        # Compute histogram
        hist = torch.zeros(10, device=grid.device, dtype=torch.float32)
        unique, counts = torch.unique(grid_clamped, return_counts=True)
        hist[unique] = counts.float()
        
        # Normalize by total pixels
        hist = hist / max(grid.numel(), 1)
        
        return hist
    
    def forward(self, demos: List[Dict], test: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all heads for pretraining
        
        Args:
            demos: List of demonstration input/output pairs
            test: Test input (for consistency with base model)
            
        Returns:
            Dictionary of predictions from each head
        """
        # Get neural features from base model encoder
        test_grid = test['input']
        if test_grid.dim() == 2:
            test_grid = test_grid.unsqueeze(0)  # Add batch dim
        
        # Normalize for encoder
        enc_input = test_grid.float() / 9.0  # Scale to [0, 1]

        # Extract features using component references (not self.base_model - circular ref!)
        feat, glob = self._encoder_ref(enc_input)
        # slots returns 3 values: slot_vecs, attention, extras
        slot_vecs, _, _ = self._slots_ref(feat)
        slots_rel = self._reln_ref(slot_vecs)
        pooled = slots_rel.mean(dim=1)

        # Control vector (same as base model)
        brain = torch.cat([glob, pooled], dim=-1)  # [B, ctrl_dim]

        # === HEAD PREDICTIONS ===
        predictions = {}

        # (A) Final Grid Head - Skip (would create circular ref via painter)
        # We don't actually use grid predictions in as_op_bias/as_param_priors anyway
        # predictions['grid'] = grid_pred  # REMOVED to avoid circular ref
        
        # (B) Program Tokens Head - Predict operation sequence
        program_logits = self.program_head(brain)  # [B, vocab_size * max_length]
        program_logits = program_logits.view(-1, self.config.max_program_length, self.config.program_vocab_size)
        predictions['program_tokens'] = program_logits
        
        # (C) Size Class Head - Classify size transformation
        size_logits = self.size_class_head(brain)  # [B, size_classes]
        predictions['size_class'] = size_logits
        
        # (D) Symmetry Class Head - Classify symmetry operations
        symmetry_logits = self.symmetry_class_head(brain)  # [B, symmetry_classes]
        predictions['symmetry_class'] = symmetry_logits
        
        # (E) Color Histogram Head - Predict histogram changes
        histogram_pred = self.color_histogram_head(brain)  # [B, histogram_dim * 2]
        histogram_pred = histogram_pred.view(-1, 2, self.config.histogram_dim)  # [B, 2, 10]
        predictions['color_histogram'] = histogram_pred

        # (F) Critic
        predictions['critic_logit'] = self.critic_head(brain).squeeze(-1)

        return predictions
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-head loss for pretraining
        
        Args:
            predictions: Dictionary of predictions from each head
            targets: Dictionary of target values for each head
            
        Returns:
            total_loss: Combined weighted loss
            loss_components: Individual loss for each head
        """
        # Get device from any available prediction
        device = next((v.device for v in predictions.values() if torch.is_tensor(v)), torch.device('cuda'))
        loss_components = {}

        # (A) Grid Head Loss - Pixel-wise cross-entropy (optional, skipped if no grid pred)
        if 'grid' in predictions and 'grid' in targets:
            grid_pred = predictions['grid']  # [B, H, W]
            grid_target = targets['grid']    # [B, H, W]
            
            # Convert to logits format for cross-entropy
            B, H, W = grid_pred.shape
            grid_pred_flat = grid_pred.view(B, H * W)  # [B, H*W]
            grid_target_flat = grid_target.view(B, H * W)  # [B, H*W]
            
            # Create one-hot logits (assuming grid_pred contains class indices)
            grid_logits = F.one_hot(torch.clamp(grid_pred_flat.long(), 0, 9), num_classes=10).float()
            grid_logits = grid_logits.view(B, H * W, 10)  # [B, H*W, 10]
            
            grid_loss = F.cross_entropy(
                grid_logits.transpose(1, 2),  # [B, 10, H*W]
                grid_target_flat.long().clamp(0, 9)  # [B, H*W]
            )
            loss_components['grid'] = grid_loss
        else:
            loss_components['grid'] = torch.tensor(0.0, device=device)
        
        # (B) Program Tokens Loss - Sequence cross-entropy with teacher forcing
        if 'program_tokens' in targets:
            program_pred = predictions['program_tokens']  # [B, max_length, vocab_size]
            program_target = targets['program_tokens']    # [B, max_length]

            # Flatten for cross-entropy
            B, L_pred, V = program_pred.shape

            # Ensure program_target has correct shape
            if program_target.dim() == 1:
                # If flat, infer target length from total size
                total_size = program_target.numel()
                L_tgt = total_size // B
                program_target = program_target.view(B, L_tgt)
            elif program_target.dim() == 2:
                L_tgt = program_target.size(1)
            else:
                raise ValueError(f"Unexpected program_target shape: {program_target.shape}")

            # If lengths don't match, align them
            if L_tgt != L_pred:
                if L_tgt > L_pred:
                    # Truncate target to match pred length
                    program_target = program_target[:, :L_pred]
                else:
                    # Pad target with zeros
                    program_target = F.pad(program_target, (0, L_pred - L_tgt), value=0)

            # Now flatten both (they have matching length L_pred)
            program_loss = F.cross_entropy(
                program_pred.reshape(B * L_pred, V),
                program_target.reshape(B * L_pred).long().clamp(0, V - 1)
            )
            loss_components['program'] = program_loss
        else:
            loss_components['program'] = torch.tensor(0.0, device=device)
        
        # (C) Size Class Loss - Classification cross-entropy
        if 'size_class' in targets:
            size_pred = predictions['size_class']    # [B, size_classes]
            size_target = targets['size_class']      # [B]
            
            size_loss = F.cross_entropy(
                size_pred, 
                size_target.long().clamp(0, self.config.size_classes - 1)
            )
            loss_components['size'] = size_loss
        else:
            loss_components['size'] = torch.tensor(0.0, device=device)
        
        # (D) Symmetry Class Loss - Classification cross-entropy
        if 'symmetry_class' in targets:
            symmetry_pred = predictions['symmetry_class']  # [B, symmetry_classes]
            symmetry_target = targets['symmetry_class']    # [B]
            
            symmetry_loss = F.cross_entropy(
                symmetry_pred,
                symmetry_target.long().clamp(0, self.config.symmetry_classes - 1)
            )
            loss_components['symmetry'] = symmetry_loss
        else:
            loss_components['symmetry'] = torch.tensor(0.0, device=device)
        
        # (E) Color Histogram Loss - L1 regression
        if 'color_histogram' in targets:
            hist_pred = predictions['color_histogram']  # [B, 2, 10]
            hist_target = targets['color_histogram']    # [B, 2, 10]

            histogram_loss = F.l1_loss(hist_pred, hist_target)
            loss_components['histogram'] = histogram_loss
        else:
            loss_components['histogram'] = torch.tensor(0.0, device=device)

        # (F) Critic loss (optional target: 1 if exact match on demos)
        if 'critic' in targets:
            y = targets['critic'].float().clamp(0,1)
            crit = torch.nn.functional.binary_cross_entropy_with_logits(
                predictions['critic_logit'], y
            )
            loss_components['critic'] = crit
        else:
            loss_components['critic'] = torch.tensor(0.0, device=device)

        # Combined weighted loss (lightly regularize critic)
        total_loss = (
            self.config.lambda_grid * loss_components['grid'] +
            self.config.lambda_program * loss_components['program'] +
            self.config.lambda_size * loss_components['size'] +
            self.config.lambda_symmetry * loss_components['symmetry'] +
            self.config.lambda_histogram * loss_components['histogram'] +
            0.2 * loss_components['critic']
        )
        
        return total_loss, loss_components
    
    def prepare_targets(self, synthetic_task) -> Dict[str, torch.Tensor]:
        """
        Prepare target values for all heads from synthetic task
        
        Args:
            synthetic_task: SyntheticTask object with ground truth
            
        Returns:
            Dictionary of target tensors for each head
        """
        targets = {}
        
        # Grid target - output grid
        targets['grid'] = synthetic_task.test_output.unsqueeze(0)  # Add batch dim
        
        # Program tokens target - encode DSL operations
        program_tokens = self.encode_program(synthetic_task.program, synthetic_task.params)
        targets['program_tokens'] = program_tokens.unsqueeze(0)  # Add batch dim
        
        # Size class target - classify size transformation
        input_size = synthetic_task.test_input.shape
        output_size = synthetic_task.test_output.shape
        size_class = self.classify_size_transformation(input_size, output_size)
        targets['size_class'] = torch.tensor([size_class])
        
        # Symmetry class target - classify symmetry operations
        symmetry_class = self.classify_symmetry_operation(synthetic_task.program)
        targets['symmetry_class'] = torch.tensor([symmetry_class])
        
        # Color histogram target - input and output histograms
        input_hist = self.compute_color_histogram(synthetic_task.test_input)
        output_hist = self.compute_color_histogram(synthetic_task.test_output)
        histogram_target = torch.stack([input_hist, output_hist], dim=0)  # [2, 10]
        targets['color_histogram'] = histogram_target.unsqueeze(0)  # Add batch dim
        
        return targets
    
    def get_pretraining_mode(self) -> bool:
        """Check if model is in pretraining mode"""
        return hasattr(self, '_pretraining_mode') and self._pretraining_mode
    
    def set_pretraining_mode(self, enabled: bool = True):
        """Set pretraining mode flag"""
        self._pretraining_mode = enabled
        if enabled:
            print("[MultiHead] Pretraining mode enabled")
        else:
            print("[MultiHead] Pretraining mode disabled")

    def as_op_bias(self, predictions: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Convert program_tokens predictions to op_bias dictionary for brain_priors.

        Args:
            predictions: Dictionary from forward() containing 'program_tokens' logits

        Returns:
            op_bias: Dict mapping operation names to bias values (probabilities)
        """
        if 'program_tokens' not in predictions:
            return {}

        # program_tokens shape: [B, max_length, vocab_size]
        program_logits = predictions['program_tokens']

        # Take first position (after START token) as primary operation prior
        # Shape: [B, vocab_size]
        first_op_logits = program_logits[:, 1, :]  # Skip START token at position 0

        # Convert to probabilities
        first_op_probs = torch.softmax(first_op_logits, dim=-1)

        # Average over batch
        op_probs = first_op_probs.mean(dim=0).detach()  # [vocab_size]

        # Build dictionary mapping operation names to probabilities
        op_bias = {}
        reverse_vocab = {idx: op for op, idx in self.op_vocab.items()}

        for idx in range(min(len(reverse_vocab), op_probs.shape[0])):
            op_name = reverse_vocab.get(idx)
            if op_name and op_name not in ['<PAD>', '<START>', '<END>', '<UNK>']:
                # Keep on GPU until final conversion
                op_bias[op_name] = float(op_probs[idx].item())

        return op_bias

    def as_param_priors(self, predictions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Convert multi-head predictions into param_priors dictionary for DSL operations.
        Extracts parameter guidance from size_class, symmetry_class, and color_histogram heads.

        Args:
            predictions: Dictionary from forward() containing all head predictions

        Returns:
            param_priors: Dict mapping parameter names to prior tensors (ALL ON GPU!)
        """
        param_priors = {}

        # Get device from first available tensor in predictions (robust to missing keys)
        device = None
        for value in predictions.values():
            if torch.is_tensor(value):
                device = value.device
                break

        if device is None:
            # Fallback to cuda if no tensors found
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # === Extract from size_class head ===
        if 'size_class' in predictions:
            size_logits = predictions['size_class']  # [B, size_classes]
            # Keep on GPU for downstream ops
            param_priors['scale_logits'] = size_logits[:, :2]  # First 2 classes for scale factors

            # Crop bbox hints from size classes (classes 3-6 indicate cropping)
            # Provide coarse bbox guidance: [y0, x0, y1, x1] normalized to [0,1]
            crop_classes = size_logits[:, 3:7] if size_logits.shape[1] >= 7 else torch.zeros(size_logits.shape[0], 4, device=device)
            param_priors['crop_bbox_logits'] = crop_classes

        # === Extract from symmetry_class head ===
        if 'symmetry_class' in predictions:
            symmetry_logits = predictions['symmetry_class']  # [B, symmetry_classes]

            # Map symmetry classes to rotation/flip parameters
            # Classes: 0=none, 1=rot90, 2=rot180, 3=rot270, 4=flip_h, 5=flip_v, 6=translate, 7=identity

            # Rotation hints (classes 1-3)
            rotation_logits = torch.zeros(symmetry_logits.shape[0], 3, device=device)
            if symmetry_logits.shape[1] >= 4:
                rotation_logits = symmetry_logits[:, 1:4]  # rot90, rot180, rot270
            param_priors['rotation_logits'] = rotation_logits

            # Flip axis hints (classes 4-5: horizontal vs vertical)
            flip_logits = torch.zeros(symmetry_logits.shape[0], 2, device=device)
            if symmetry_logits.shape[1] >= 6:
                flip_logits = symmetry_logits[:, 4:6]  # flip_h, flip_v
            param_priors['flip_axis_logits'] = flip_logits

            # Translation hints (class 6)
            if symmetry_logits.shape[1] >= 7:
                translate_prior = symmetry_logits[:, 6:7]  # Single logit for translation likelihood
                # Expand to dx/dy priors (use same value for both)
                param_priors['dx_logits'] = translate_prior.expand(-1, 4)  # 4 bins for dx
                param_priors['dy_logits'] = translate_prior.expand(-1, 4)  # 4 bins for dy

        # === Extract from color_histogram head ===
        if 'color_histogram' in predictions:
            hist_pred = predictions['color_histogram']  # [B, 2, 10]
            # Output histogram (second one) provides color map guidance
            output_hist = hist_pred[:, 1, :]  # [B, 10]
            param_priors['color_map_logits'] = output_hist

        return param_priors

    def as_certificates(self, predictions: Dict[str, torch.Tensor], demos: List[Dict] = None) -> Dict[str, Any]:
        """
        Convert pretrainer predictions into CEGIS-style certificates for search pruning.
        Extracts invariants from size_class and symmetry_class predictions.

        Args:
            predictions: Dictionary from forward() containing all head predictions
            demos: Optional demo pairs for computing input/output statistics

        Returns:
            certificates: Dict of invariant predicates for CEGIS soft/hard penalties
        """
        certificates = {}

        try:
            # === Size invariants from size_class head ===
            if 'size_class' in predictions:
                size_logits = predictions['size_class']  # [B, size_classes]
                size_class = torch.argmax(size_logits, dim=-1).item()

                # Map size class to invariants
                # Classes: 0=no-change, 1=uniform-scale-up, 2=nonuniform-scale-up,
                #          3=uniform-scale-down, 4=nonuniform-scale-down, 5=padding, 6=cropping

                if size_class == 0:
                    # No size change expected
                    certificates['shape_preservation'] = True
                    certificates['mass_conservation'] = True
                elif size_class in [1, 2]:
                    # Scaling up
                    certificates['scale_up'] = True
                    certificates['uniform_scale'] = (size_class == 1)
                elif size_class in [3, 4]:
                    # Scaling down
                    certificates['scale_down'] = True
                    certificates['uniform_scale'] = (size_class == 3)
                elif size_class == 5:
                    # Padding/extension
                    certificates['padding'] = True
                    certificates['bbox_area_monotonic'] = 'increase'
                elif size_class == 6:
                    # Cropping/reduction
                    certificates['cropping'] = True
                    certificates['bbox_area_monotonic'] = 'decrease'

            # === Symmetry invariants from symmetry_class head ===
            if 'symmetry_class' in predictions:
                symmetry_logits = predictions['symmetry_class']  # [B, symmetry_classes]
                symmetry_class = torch.argmax(symmetry_logits, dim=-1).item()

                # Map symmetry class to operation constraints
                # Classes: 0=none, 1=rot90, 2=rot180, 3=rot270, 4=flip_h, 5=flip_v, 6=translate, 7=identity

                if symmetry_class == 0:
                    # No symmetry operation
                    certificates['no_rotation'] = True
                    certificates['no_flip'] = True
                elif symmetry_class in [1, 2, 3]:
                    # Rotation expected
                    certificates['rotation_degrees'] = [90, 180, 270][symmetry_class - 1]
                    certificates['mass_conservation'] = True
                    # After rotation, shape may change (90/270 swap H/W)
                    if symmetry_class in [1, 3]:
                        certificates['shape_preservation'] = False
                elif symmetry_class in [4, 5]:
                    # Flip expected
                    certificates['flip_axis'] = 'h' if symmetry_class == 4 else 'v'
                    certificates['shape_preservation'] = True
                    certificates['mass_conservation'] = True
                elif symmetry_class == 6:
                    # Translation expected
                    certificates['translation'] = True
                    certificates['shape_preservation'] = True
                elif symmetry_class == 7:
                    # Identity/no-op
                    certificates['identity'] = True
                    certificates['shape_preservation'] = True
                    certificates['mass_conservation'] = True

            # === Color invariants from histogram head ===
            if 'color_histogram' in predictions:
                hist_pred = predictions['color_histogram']  # [B, 2, 10]
                input_hist = hist_pred[0, 0, :]  # Input histogram
                output_hist = hist_pred[0, 1, :]  # Output histogram

                # Check if palette subset (output colors ⊆ input colors)
                input_colors = (input_hist > 0.01).nonzero(as_tuple=True)[0].tolist()
                output_colors = (output_hist > 0.01).nonzero(as_tuple=True)[0].tolist()

                palette_subset = all(c in input_colors for c in output_colors)
                certificates['palette_subset'] = palette_subset

                # Check if dominant colors preserved
                input_dominant = torch.argmax(input_hist).item()
                output_dominant = torch.argmax(output_hist).item()
                certificates['dominant_color_preserved'] = (input_dominant == output_dominant)

            # === Demo-based invariants (if demos provided) ===
            if demos and len(demos) > 0:
                # Compute mass conservation from demos
                mass_conserved = True
                for demo in demos[:3]:  # Check first 3 demos
                    demo_in = demo.get('input') if isinstance(demo, dict) else demo[0]
                    demo_out = demo.get('output') if isinstance(demo, dict) else demo[1]

                    if demo_in is not None and demo_out is not None:
                        mass_in = (demo_in != 0).sum().item()
                        mass_out = (demo_out != 0).sum().item()
                        if abs(mass_in - mass_out) > 0.1 * mass_in:
                            mass_conserved = False
                            break

                if mass_conserved:
                    certificates['mass_conservation'] = True

        except Exception as e:
            import logging
            logging.warning(f"[MultiHead] Certificate extraction failed: {e}")

        return certificates


# Quick test if run directly
if __name__ == "__main__":
    if not TOPAS_AVAILABLE:
        print("TOPAS model not available - cannot test MultiHeadPretrainer")
        exit(1)
    
    print("Testing Multi-Head Pretrainer...")
    
    # Create base model
    config = ModelConfig(width=320, depth=8, slots=40)  # Smaller for testing
    base_model = TopasARC60M(config)
    
    # Create pretrainer
    pretrain_config = PretrainConfig()
    pretrainer = MultiHeadPretrainer(base_model, pretrain_config)
    pretrainer.set_pretraining_mode(True)
    
    # Test forward pass
    test_input = torch.randint(0, 10, (8, 8))
    demos = [{'input': test_input, 'output': torch.randint(0, 10, (8, 8))}]
    test = {'input': test_input}
    
    # Forward pass
    with torch.no_grad():
        predictions = pretrainer(demos, test)
    
    print(f"\nPredictions:")
    for head_name, pred in predictions.items():
        print(f"  {head_name}: {pred.shape}")
    
    # Test target preparation (requires synthetic task)
    print(f"\nOperation vocabulary size: {len(pretrainer.op_vocab)}")
    print(f"Sample operations: {list(pretrainer.op_vocab.keys())[:10]}")
    
    print("\n✅ Multi-Head Pretrainer test completed!")