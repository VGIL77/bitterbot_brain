import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Union, Tuple, List

class SpikeTensor(torch.Tensor):
    pass  # for type tagging only

def adaptive_threshold_spike(x: torch.Tensor, k: float = 8.0, mode: str = "int",
                           ripple_context: Optional[Dict] = None,
                           attention_weights: Optional[torch.Tensor] = None,
                           phase_lock: bool = False) -> torch.Tensor:
    """
    Enhanced adaptive threshold spike coding with ripple and attention integration.

    Collapse activations -> integer spike counts (Eq. 22-25 SpikingBrain)
    Modes:
      - int: single-step integer spike count (training mode)
      - ternary: expand into {-1,0,1} spike trains (eval probe)
      - bitwise: expand spike counts into bitplanes (eval probe)

    k controls sparsity: smaller k = higher threshold = more sparsity

    Enhanced features:
    - ripple_context: Dict with 'is_active', 'phase', 'coherence' for ripple-aware thresholding
    - attention_weights: Apply different k values based on attention (sparse for low attention)
    - phase_lock: Lock spike timing to ripple phase for temporal binding
    """
    with torch.no_grad():
        # Start with base threshold calculation
        std = x.std() + 1e-6
        base_k = k

        # === RIPPLE-AWARE THRESHOLD MODULATION ===
        if ripple_context is not None and isinstance(ripple_context, dict):
            is_active = ripple_context.get('is_active', False)
            phase = ripple_context.get('phase', 0.0)
            coherence = ripple_context.get('coherence', 0.0)

            if is_active and coherence > 0.75:
                # During high-coherence ripples: more selective (lower k = higher threshold)
                ripple_factor = 0.3 + 0.4 * (1.0 - coherence)  # 0.3-0.7 range
                base_k = base_k * ripple_factor

                # Phase-locked modulation for temporal binding
                if phase_lock:
                    # Boost spikes at optimal phases (0, Ï€/2, Ï€, 3Ï€/2)
                    optimal_phases = [0, np.pi/2, np.pi, 3*np.pi/2]
                    phase_distances = [abs(phase - opt) for opt in optimal_phases]
                    min_distance = min(phase_distances)
                    phase_bonus = torch.exp(torch.tensor(-min_distance, device=x.device))
                    base_k = base_k * (2.0 - phase_bonus)  # Lower k at optimal phases

        # === ATTENTION-WEIGHTED ADAPTIVE THRESHOLDING ===
        if attention_weights is not None:
            # Expand attention weights to match x dimensions if needed
            if attention_weights.dim() < x.dim():
                for _ in range(x.dim() - attention_weights.dim()):
                    attention_weights = attention_weights.unsqueeze(-1)
            attention_weights = attention_weights.expand_as(x)

            # Attention-based k modulation: high attention = permissive, low attention = aggressive
            attention_threshold = torch.quantile(attention_weights.flatten(), 0.7)
            k_adaptive = torch.where(
                attention_weights < attention_threshold,
                base_k * 0.4,  # Aggressive spikes for low attention (70% sparsity)
                base_k * 2.0   # Permissive spikes for high attention (30% sparsity)
            )
            vth = std * k_adaptive
        else:
            vth = std * base_k

        # Compute spikes with clamping
        s_int = torch.round(x / vth).clamp(-127, 127).to(torch.int8)

    if mode == "int":
        return s_int

    elif mode == "ternary":
        # map to {-1, 0, 1} with symmetric quantization
        s_tern = torch.sign(s_int).clamp(min=-1, max=1).to(torch.int8)
        return s_tern

    elif mode == "bitwise":
        # unfold into bitplanes (for eval probes)
        max_bits = 8
        shape = (*s_int.shape, max_bits)
        bitplanes = torch.zeros(shape, dtype=torch.int8, device=x.device)
        for b in range(max_bits):
            bitplanes[..., b] = ((s_int >> b) & 1).to(torch.int8)
        return bitplanes

    else:
        raise ValueError(f"Unknown spike mode {mode}")

def spike_stats(spikes: torch.Tensor):
    nonzero = (spikes != 0).float().mean().item()
    # Keep histogram computation on GPU, only convert final result
    hist_gpu = torch.bincount(spikes.flatten().abs(), minlength=8)
    hist = hist_gpu.cpu().numpy()  # Minimal CPU transfer for final numpy conversion
    return {
        "pct_active": nonzero * 100,
        "hist": hist.tolist(),
    }

def hierarchical_spike_cascade(x: torch.Tensor, level_configs: Optional[List[Dict]] = None) -> Tuple[torch.Tensor, ...]:
    """
    Multi-level spike propagation for complex reasoning.

    Args:
        x: Input activations [B, T, D] or [B, D]
        level_configs: List of dicts with 'k' and 'mode' for each level

    Returns:
        Tuple of spike tensors for each level
    """
    if level_configs is None:
        # Default 3-level hierarchy: perceptual -> conceptual -> abstract
        level_configs = [
            {'k': 2.0, 'mode': 'int'},      # Level 1: Preserve detail
            {'k': 1.0, 'mode': 'ternary'},  # Level 2: More selective
            {'k': 0.3, 'mode': 'ternary'}   # Level 3: Highly selective
        ]

    current_input = x
    spike_levels = []

    for i, config in enumerate(level_configs):
        spikes = adaptive_threshold_spike(
            current_input,
            k=config['k'],
            mode=config['mode']
        )
        spike_levels.append(spikes)

        # Convert back to float for next level (preserve sparsity pattern)
        current_input = spikes.float()

        # Optional: Add small linear transformation between levels
        if i < len(level_configs) - 1:
            # Simple compression for next level
            if current_input.dim() > 1:
                current_input = current_input * 0.9  # Light decay between levels

    return tuple(spike_levels)

def compute_spike_density(spike_tensor: torch.Tensor, window_size: int = 16) -> torch.Tensor:
    """
    Compute local spike density for exploration guidance.

    Args:
        spike_tensor: Spike tensor [B, T, D] or [B, D]
        window_size: Local window size for density computation

    Returns:
        Density map same shape as input
    """
    # Convert to binary activity map
    activity = (spike_tensor != 0).float()

    if activity.dim() == 2:  # [B, D]
        B, D = activity.shape
        # Use 1D convolution for density
        kernel = torch.ones(1, 1, window_size, device=activity.device) / window_size
        # Pad and reshape for conv1d
        padded = F.pad(activity.unsqueeze(1), (window_size//2, window_size//2))
        density = F.conv1d(padded, kernel, padding=0).squeeze(1)

    elif activity.dim() == 3:  # [B, T, D]
        B, T, D = activity.shape
        # Compute density along last dimension
        kernel = torch.ones(1, 1, window_size, device=activity.device) / window_size
        # Reshape for batch processing
        reshaped = activity.view(B*T, 1, D)
        padded = F.pad(reshaped, (window_size//2, window_size//2))
        density = F.conv1d(padded, kernel, padding=0)
        density = density.view(B, T, D)

    else:
        # Fallback: simple moving average
        density = activity

    return density

def hamming_distance(spikes1: torch.Tensor, spikes2: torch.Tensor) -> float:
    """
    Compute normalized Hamming distance between two spike tensors.
    Optimized for bitwise mode comparison.

    Args:
        spikes1, spikes2: Spike tensors to compare

    Returns:
        Normalized Hamming distance [0, 1]
    """
    # Ensure same shape
    if spikes1.shape != spikes2.shape:
        # Broadcast to same shape
        spikes1 = spikes1.expand_as(spikes2) if spikes1.numel() < spikes2.numel() else spikes1
        spikes2 = spikes2.expand_as(spikes1) if spikes2.numel() < spikes1.numel() else spikes2

    # Compute binary mismatch
    mismatch = (spikes1 != spikes2).float()
    hamming_dist = mismatch.mean().item()

    return hamming_dist

def interpolate_spike_patterns(spike_patterns: torch.Tensor, num_interpolants: int = 4) -> torch.Tensor:
    """
    Generate interpolated spike patterns for exploration.

    Args:
        spike_patterns: Source spike patterns [N, D]
        num_interpolants: Number of interpolated patterns to generate

    Returns:
        Interpolated spike patterns [num_interpolants, D]
    """
    if spike_patterns.size(0) < 2:
        return spike_patterns

    N, D = spike_patterns.shape
    interpolants = []

    for i in range(num_interpolants):
        # Random pair selection
        idx1, idx2 = torch.randint(0, N, (2,))
        if idx1 == idx2:
            idx2 = (idx1 + 1) % N

        pattern1 = spike_patterns[idx1]
        pattern2 = spike_patterns[idx2]

        # Spike interpolation: blend and re-quantize
        alpha = torch.rand(1).item()
        blended = alpha * pattern1.float() + (1 - alpha) * pattern2.float()

        # Re-quantize to spikes
        interpolant = adaptive_threshold_spike(blended, k=1.0, mode="ternary")
        interpolants.append(interpolant)

    return torch.stack(interpolants)

class ARCSpikeVocabulary:
    """
    Pre-trained spike patterns for common ARC operations.
    Provides fast template matching and pattern amplification.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.vocabularies = {}
        self._initialize_arc_patterns()

    def _initialize_arc_patterns(self):
        """Initialize comprehensive ARC operation spike patterns optimized for puzzle solving."""
        # Create sophisticated spike patterns targeting common ARC reasoning patterns

        # === GEOMETRIC TRANSFORMATION PATTERNS ===

        # Rotation pattern: 4-fold rotational symmetry with phase encoding
        rotation_pattern = torch.zeros(64, device=self.device)
        for i in range(64):
            # Create spiral pattern that encodes rotation
            angle = (i / 64) * 8 * np.pi  # 4 full rotations
            rotation_pattern[i] = np.sin(angle) + 0.5 * np.cos(2 * angle)
        self.vocabularies['rotation'] = adaptive_threshold_spike(
            rotation_pattern, k=0.2, mode="ternary"
        )

        # Scaling pattern: geometric progression
        scaling_pattern = torch.zeros(64, device=self.device)
        for i in range(64):
            scaling_pattern[i] = np.exp(-i/16) * np.sin(i/4)  # Exponential decay with oscillation
        self.vocabularies['scaling'] = adaptive_threshold_spike(
            scaling_pattern, k=0.3, mode="int"
        )

        # Mirror/reflection pattern: precise bilateral symmetry
        mirror_pattern = torch.zeros(64, device=self.device)
        for i in range(32):
            val = np.sin(i * np.pi / 16) + 0.3 * np.cos(i * np.pi / 8)
            mirror_pattern[i] = val
            mirror_pattern[63-i] = val  # Perfect mirror
        self.vocabularies['mirror'] = adaptive_threshold_spike(
            mirror_pattern, k=0.25, mode="ternary"
        )

        # === COLOR AND PATTERN OPERATIONS ===

        # Color mapping: cyclic color relationships
        color_map_pattern = torch.zeros(64, device=self.device)
        for i in range(64):
            # Encode color cycles (common in ARC: 0->1->2->0 patterns)
            color_map_pattern[i] = np.cos(2 * np.pi * i / 10) * (1 + 0.3 * np.sin(i / 5))
        self.vocabularies['color_mapping'] = adaptive_threshold_spike(
            color_map_pattern, k=0.35, mode="int"
        )

        # Flood fill pattern: spreading activation
        flood_pattern = torch.zeros(64, device=self.device)
        center = 32
        for i in range(64):
            distance = abs(i - center)
            flood_pattern[i] = np.exp(-distance/8) * (2 + np.sin(distance))
        self.vocabularies['flood_fill'] = adaptive_threshold_spike(
            flood_pattern, k=0.4, mode="int"
        )

        # === STRUCTURAL AND TOPOLOGICAL PATTERNS ===

        # Connectivity pattern: graph-like structure
        connectivity_pattern = torch.zeros(64, device=self.device)
        # Create hub-and-spoke pattern (common in ARC puzzles)
        hubs = [16, 32, 48]
        for hub in hubs:
            connectivity_pattern[hub] = 3.0  # Strong hub activation
            # Connect to neighbors
            for offset in [-1, 1, -8, 8]:  # 4-connectivity
                if 0 <= hub + offset < 64:
                    connectivity_pattern[hub + offset] = 1.5
        self.vocabularies['connectivity'] = adaptive_threshold_spike(
            connectivity_pattern, k=0.5, mode="int"
        )

        # Containment pattern: nested structures
        containment_pattern = torch.zeros(64, device=self.device)
        for i in range(64):
            x, y = i % 8, i // 8
            # Create concentric squares pattern
            level = min(x, y, 7-x, 7-y)
            containment_pattern[i] = (level + 1) * 0.5
        self.vocabularies['containment'] = adaptive_threshold_spike(
            containment_pattern, k=0.3, mode="ternary"
        )

        # === SEQUENTIAL AND RULE PATTERNS ===

        # Progression pattern: arithmetic/geometric sequences
        progression_pattern = torch.zeros(64, device=self.device)
        for i in range(64):
            # Fibonacci-like progression encoded in spikes
            if i < 2:
                progression_pattern[i] = 1.0
            else:
                # Encode sequence relationships
                progression_pattern[i] = 0.618 * progression_pattern[i-1] + 0.382 * progression_pattern[i-2]
        self.vocabularies['progression'] = adaptive_threshold_spike(
            progression_pattern, k=0.4, mode="ternary"
        )

        # Alternation pattern: ABAB... patterns common in ARC
        alternation_pattern = torch.zeros(64, device=self.device)
        for i in range(64):
            alternation_pattern[i] = 2.0 * np.sin(np.pi * i) + np.cos(2 * np.pi * i)
        self.vocabularies['alternation'] = adaptive_threshold_spike(
            alternation_pattern, k=0.6, mode="ternary"
        )

        # === COMPLEX COMPOSITE PATTERNS ===

        # Grid transformation: combines rotation + scaling + translation
        grid_transform_pattern = torch.zeros(64, device=self.device)
        for i in range(64):
            x, y = i % 8, i // 8
            # Complex transformation encoding
            transformed_x = x * np.cos(np.pi/4) - y * np.sin(np.pi/4)  # Rotation
            transformed_y = x * np.sin(np.pi/4) + y * np.cos(np.pi/4)
            scale_factor = 1.0 + 0.2 * np.sin(transformed_x + transformed_y)
            grid_transform_pattern[i] = scale_factor * np.exp(-(transformed_x**2 + transformed_y**2)/16)
        self.vocabularies['grid_transform'] = adaptive_threshold_spike(
            grid_transform_pattern, k=0.45, mode="int"
        )

        # Rule application: conditional logic patterns
        rule_pattern = torch.zeros(64, device=self.device)
        for i in range(64):
            # Encode if-then-else logic structure
            condition = i % 16 < 8  # Boolean condition
            rule_pattern[i] = 2.0 if condition else -1.0
            if i % 4 == 0:  # Exception cases
                rule_pattern[i] *= 1.5
        self.vocabularies['rule_application'] = adaptive_threshold_spike(
            rule_pattern, k=0.3, mode="ternary"
        )

    def match_pattern(self, current_spikes: torch.Tensor, threshold: float = 0.8) -> Tuple[str, float]:
        """
        Find best matching ARC pattern.

        Args:
            current_spikes: Current spike pattern to match
            threshold: Minimum similarity threshold

        Returns:
            (pattern_name, similarity_score) or (None, 0.0)
        """
        if current_spikes.numel() == 0:
            return None, 0.0

        best_match = None
        best_score = 0.0

        # Flatten current pattern for comparison
        current_flat = current_spikes.flatten()

        for pattern_name, pattern_spikes in self.vocabularies.items():
            # Resize patterns to match if needed
            if current_flat.numel() != pattern_spikes.numel():
                if current_flat.numel() < pattern_spikes.numel():
                    # Pad current pattern
                    padded = torch.zeros_like(pattern_spikes)
                    padded[:current_flat.numel()] = current_flat
                    current_comp = padded
                    pattern_comp = pattern_spikes
                else:
                    # Truncate current pattern
                    current_comp = current_flat[:pattern_spikes.numel()]
                    pattern_comp = pattern_spikes
            else:
                current_comp = current_flat
                pattern_comp = pattern_spikes

            # Compute similarity (1 - normalized hamming distance)
            similarity = 1.0 - hamming_distance(current_comp, pattern_comp)

            if similarity > best_score and similarity > threshold:
                best_match = pattern_name
                best_score = similarity

        return best_match, best_score

    def get_pattern_ops(self, pattern_name: str) -> List[str]:
        """Get DSL operations associated with a pattern."""
        pattern_to_ops = {
            # Geometric transformations
            'rotation': ['rotate90', 'rotate180', 'rotate270'],
            'scaling': ['scale_up', 'scale_down', 'resize'],
            'mirror': ['flip_h', 'flip_v', 'transpose', 'symmetry'],

            # Color and pattern operations
            'color_mapping': ['map_colors', 'cycle_colors', 'recolor'],
            'flood_fill': ['flood_fill', 'tile_pattern', 'paste'],

            # Structural and topological
            'connectivity': ['connect_pixels', 'for_each_object', 'identify_components'],
            'containment': ['crop_bbox', 'extract_pattern', 'nest_objects'],

            # Sequential and rule patterns
            'progression': ['extend_pattern', 'continue_sequence', 'arithmetic_progression'],
            'alternation': ['alternate_pattern', 'toggle_colors', 'repeat_pattern'],

            # Complex composite patterns
            'grid_transform': ['transform_grid', 'apply_transformation', 'composite_operation'],
            'rule_application': ['apply_rule', 'conditional_operation', 'if_then_else'],

            # Legacy patterns (kept for compatibility)
            'symmetry': ['symmetry', 'flip_h', 'flip_v', 'transpose'],
            'pattern_fill': ['flood_fill', 'tile_pattern', 'paste'],
            'object_detection': ['for_each_object', 'crop_bbox', 'extract_pattern']
        }
        # Check static patterns first
        ops = pattern_to_ops.get(pattern_name, [])

        # Check dynamic patterns (learned during training)
        if not ops and hasattr(self, '_dynamic_pattern_ops'):
            ops = self._dynamic_pattern_ops.get(pattern_name, [])

        return ops