"""
Orbit Canonicalization for ARC-AGI: D₄ × Color Permutations

Implements symmetry-quotiented search via:
1. D₄ dihedral group (rotations + reflections)
2. Canonical color permutations (frequency + bbox signatures)
3. Orbit-contrastive loss (OrthoNCE) for invariant features

References:
- Group equivariance: Cohen & Welling 2016 (ICML)
- Burnside's lemma for orbit counting: Polya enumeration theory
- Contrastive learning under augmentations: Chen et al. 2020 (SimCLR)

Reduces search space by ~8x (D₄) and eliminates color-aliasing noise.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# D₄ Dihedral Group Operations
# ═══════════════════════════════════════════════════════════

def d4_apply(grid: torch.Tensor, rot: int = 0, reflect: bool = False) -> torch.Tensor:
    """
    Apply D₄ transformation to grid.

    Args:
        grid: Input grid [H, W]
        rot: Number of 90° rotations (0-3)
        reflect: Whether to flip horizontally after rotation

    Returns:
        Transformed grid [H', W']
    """
    g = torch.rot90(grid, k=int(rot) % 4, dims=(-2, -1))
    return torch.flip(g, dims=(-1,)) if reflect else g


def d4_inverse(rot: int, reflect: bool) -> Tuple[int, bool]:
    """
    Compute inverse D₄ transformation.

    Args:
        rot: Original rotation
        reflect: Original reflection

    Returns:
        (inverse_rot, inverse_reflect)
    """
    # Reflection is self-inverse
    # Rotation inverse: -rot mod 4
    inv_rot = (-rot) % 4
    return inv_rot, reflect


# ═══════════════════════════════════════════════════════════
# Color Canonicalization (Frequency + BBox Signatures)
# ═══════════════════════════════════════════════════════════

def color_counts(grid: torch.Tensor) -> torch.Tensor:
    """Count occurrences of each color (0-9)"""
    return torch.bincount(grid.view(-1).long().clamp(0, 9), minlength=10)


def _bbox_sig(grid: torch.Tensor, c: int) -> Tuple[int, int, int, int, int]:
    """
    Compute bounding box signature for color c.

    Returns: (count, height, width, min_row, min_col)
    Used for stable tie-breaking in canonical color ordering.
    """
    mask = (grid == c)
    cnt = int(mask.sum().item())
    if cnt == 0:
        return (0, 0, 0, 999, 999)

    idx = mask.nonzero(as_tuple=False)
    r0, c0 = int(idx[:, 0].min()), int(idx[:, 1].min())
    r1, c1 = int(idx[:, 0].max()), int(idx[:, 1].max())

    return (cnt, (r1 - r0 + 1), (c1 - c0 + 1), r0, c0)


def canonical_color_perm(a: torch.Tensor, b: torch.Tensor) -> Dict[int, int]:
    """
    Build canonical color permutation for paired grids (input, output).

    Orders colors by:
    1. Total frequency (descending)
    2. Bbox signatures (lexicographic tie-break)

    Args:
        a: Input grid [H, W]
        b: Output grid [H, W]

    Returns:
        Permutation dict {old_color: new_color}
    """
    ab = torch.stack([a, b], dim=0)
    counts = color_counts(ab)
    active = [c for c in range(10) if counts[c] > 0]

    # Compute signatures for stable ordering
    sigs = []
    for c in active:
        s1 = _bbox_sig(a, c)
        s2 = _bbox_sig(b, c)
        # Sort by: (descending count, bbox heuristics)
        sigs.append((c, counts[c].item(), s1, s2))

    # Higher count first, then lexicographic on bbox signatures
    sigs.sort(key=lambda x: (-x[1], x[2], x[3], x[0]))

    # Map to canonical indices 0..k-1 in sorted order
    mapping = {old: i for i, (old, *_rest) in enumerate(sigs)}

    # Unused colors map to themselves
    for c in range(10):
        mapping.setdefault(c, c)

    return mapping


def apply_color_perm(grid: torch.Tensor, perm: Dict[int, int]) -> torch.Tensor:
    """Apply color permutation via lookup table"""
    lut = torch.arange(10, device=grid.device)
    for k, v in perm.items():
        lut[k] = v
    return lut[grid.long().clamp(0, 9)]


# ═══════════════════════════════════════════════════════════
# Full Canonicalization (D₄ + Color)
# ═══════════════════════════════════════════════════════════

def canonicalize_pair(inp: torch.Tensor, out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Canonicalize (input, output) pair under D₄ × color permutations.

    Searches all 8 D₄ transformations × color canonicalizations,
    returns lexicographically smallest representation.

    Args:
        inp: Input grid [H, W]
        out: Output grid [H, W]

    Returns:
        (canonical_input, canonical_output, metadata)
        metadata = {"rot": int, "reflect": bool, "perm": Dict[int,int]}
    """
    assert inp.dim() == 2 and out.dim() == 2, "canonicalize_pair expects 2D grids"

    best = None
    best_meta = None
    best_ic, best_oc = None, None

    # Try all 8 D₄ transformations
    for rot in range(4):
        for ref in (False, True):
            i2 = d4_apply(inp, rot, ref)
            o2 = d4_apply(out, rot, ref)

            # Compute canonical color permutation for this orientation
            perm = canonical_color_perm(i2, o2)
            ic = apply_color_perm(i2, perm)
            oc = apply_color_perm(o2, perm)

            # Lexicographic key for comparison (use reshape for non-contiguous tensors)
            key = (tuple(ic.reshape(-1).tolist()), tuple(oc.reshape(-1).tolist()))

            if (best is None) or (key < best):
                best = key
                best_meta = {"rot": rot, "reflect": ref, "perm": perm}
                best_ic, best_oc = ic, oc

    return best_ic, best_oc, best_meta


def invert_transform(grid: torch.Tensor, meta: Dict) -> torch.Tensor:
    """
    Invert canonicalization to recover original representation.

    Args:
        grid: Canonical grid [H, W]
        meta: Metadata from canonicalize_pair

    Returns:
        Original grid [H, W]
    """
    # Invert color permutation
    invp = {v: k for k, v in meta["perm"].items()}
    g = apply_color_perm(grid, invp)

    # Invert D₄ transformation
    rot = meta["rot"]
    reflect = meta["reflect"]

    # Reflection is self-inverse
    if reflect:
        g = torch.flip(g, dims=(-1,))

    # Rotation inverse: -rot mod 4
    inv_rot = (-rot) % 4
    g = torch.rot90(g, k=inv_rot, dims=(-2, -1))

    return g


# ═══════════════════════════════════════════════════════════
# Orbit-Contrastive Loss (OrthoNCE)
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def _quick_feature(model, grid: torch.Tensor) -> torch.Tensor:
    """
    Extract feature vector from grid using model encoder + slots.

    Args:
        model: TopasARC60M model
        grid: Input grid [H, W]

    Returns:
        Feature vector [D]
    """
    x = grid.unsqueeze(0).unsqueeze(1).float() / 9.0  # [1, 1, H, W]

    # Get encoder features
    feat, glob = model.encoder(x)  # [1, C, H, W], [1, Dw]

    # Get slots
    slots = model.slots(feat)
    if isinstance(slots, tuple):
        slots = slots[0]

    # Concatenate global + mean slot
    z = torch.cat([glob, slots.mean(dim=1)], dim=-1)  # [1, Dw+Ds]

    # FIDELITY GUARD: Sanitize before return
    z = torch.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0)

    return z.squeeze(0)  # [D]


def orbit_invariance_loss(model, inp: torch.Tensor, k: int = 2, tau: float = 0.2) -> torch.Tensor:
    """
    Orbit-contrastive loss (OrthoNCE): enforces feature invariance under D₄ orbit.

    Creates k random D₄ transformations of input, computes features for each,
    and pulls them together in feature space via NT-Xent loss.

    Args:
        model: TopasARC60M model
        inp: Input grid [H, W]
        k: Number of positive samples (D₄ augmentations)
        tau: Temperature for NT-Xent

    Returns:
        Contrastive loss (scalar tensor)
    """
    # Get feature for original (anchor)
    with torch.set_grad_enabled(True):
        # Encode anchor WITH gradients
        x_anchor = inp.unsqueeze(0).unsqueeze(1).float() / 9.0  # [1, 1, H, W]
        feat_a, glob_a = model.encoder(x_anchor)
        slots_a = model.slots(feat_a)
        if isinstance(slots_a, tuple):
            slots_a = slots_a[0]
        z_anchor = torch.cat([glob_a, slots_a.mean(dim=1)], dim=-1).squeeze(0)  # [D]

        # FIDELITY GUARD: Sanitize z_anchor
        z_anchor = torch.nan_to_num(z_anchor, nan=0.0, posinf=10.0, neginf=-10.0)

    # Generate positive samples (D₄ orbit members) with no_grad
    positives = []
    for _ in range(max(1, k)):
        rot = int(torch.randint(0, 4, (1,)).item())
        ref = bool(torch.randint(0, 2, (1,)).item())

        with torch.no_grad():  # Augmentations don't need gradients
            t = d4_apply(inp, rot, ref)
            z_t = _quick_feature(model, t)
            positives.append(z_t)

    # Stack: [1 + k, D]
    z = torch.stack([z_anchor] + positives, dim=0)

    # FIDELITY GUARD: Add eps to prevent NaN from zero-norm vectors
    z = F.normalize(z, p=2, dim=-1, eps=1e-8)

    # NT-Xent: anchor (index 0) should match all positives
    sim = z @ z.t() / tau  # [1+k, 1+k]

    # FIDELITY GUARD: Sanitize similarity matrix
    sim = torch.nan_to_num(sim, nan=0.0, posinf=50.0, neginf=-50.0)

    # Labels: anchor at index 0, all others are positives
    labels = torch.zeros(sim.size(0), dtype=torch.long, device=sim.device)

    # Cross-entropy: pulls orbit together, pushes against implicit negatives
    loss = F.cross_entropy(sim, labels)

    # FIDELITY GUARD: Sanitize final loss
    loss = torch.nan_to_num(loss, nan=0.0, posinf=10.0, neginf=0.0)

    return loss


# ═══════════════════════════════════════════════════════════
# Batch Canonicalization Helpers
# ═══════════════════════════════════════════════════════════

def canonicalize_batch(inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """
    Canonicalize a batch of (input, target) pairs.

    Args:
        inputs: [B, H, W]
        targets: [B, H, W]

    Returns:
        (canonical_inputs, canonical_targets, metadata_list)
    """
    B = inputs.shape[0]
    canon_inputs = []
    canon_targets = []
    meta_list = []

    for b in range(B):
        ic, oc, meta = canonicalize_pair(inputs[b], targets[b])
        canon_inputs.append(ic)
        canon_targets.append(oc)
        meta_list.append(meta)

    # === Dynamic Padding to Restore Staircase Behavior ===
    Hmax = max([t.shape[-2] for t in canon_inputs])
    Wmax = max([t.shape[-1] for t in canon_inputs])

    def pad_to(x, H, W):
        h, w = x.shape[-2], x.shape[-1]
        if h == H and w == W:
            return x
        out = x.new_zeros((H, W))
        out[:h, :w] = x
        return out

    canon_inputs = [pad_to(t, Hmax, Wmax) for t in canon_inputs]
    canon_targets = [pad_to(t, Hmax, Wmax) for t in canon_targets]

    try:
        stacked_inputs = torch.stack(canon_inputs)
        stacked_targets = torch.stack(canon_targets)
    except Exception as e:
        import logging
        logging.warning(f"[Orbit] Padding fallback triggered: {e}")
        return inputs, targets, meta_list

    return stacked_inputs, stacked_targets, meta_list


# ═══════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════

def orbit_statistics(demos: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict:
    """
    Compute orbit statistics for a set of demos.

    Args:
        demos: List of (input, output) pairs

    Returns:
        Statistics dict with orbit compression metrics
    """
    if not demos:
        return {}

    # Count unique canonical representations
    canonical_keys = set()
    for inp, out in demos:
        ic, oc, _ = canonicalize_pair(inp, out)
        key = (tuple(ic.view(-1).tolist()), tuple(oc.view(-1).tolist()))
        canonical_keys.add(key)

    return {
        "total_demos": len(demos),
        "unique_canonical": len(canonical_keys),
        "compression_ratio": len(demos) / max(1, len(canonical_keys))
    }
