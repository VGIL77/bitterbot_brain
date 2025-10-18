
# relation_classifier.py
# Small MLP to map pairs of slot vectors to a distribution over DSL ops.
from typing import Tuple, Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.dsl_registry import DSL_OPS
except Exception:
    # Fallback for standalone testing
    DSL_OPS = [
        "identity","rotate90","rotate180","rotate270","flip_h","flip_v",
        "translate","scale","resize_nn","color_map","flood_fill","extract_color",
        "crop_bbox","crop_nonzero","tile_pattern","tile","paste","center_pad_to","outline","symmetry",
        "grid_union","grid_intersection","grid_xor","grid_difference","count_objects","count_colors",
        "find_pattern","extract_pattern","match_template","for_each_object","for_each_object_translate",
        "for_each_object_recolor","for_each_object_rotate","for_each_object_scale","for_each_object_flip",
        "conditional_map","apply_rule","select_by_property","flood_select","boundary_extract","arithmetic_op"
    ]

class RelationClassifier(nn.Module):
    def __init__(self, slot_dim: int, hidden: int = 256, num_ops: int = None):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_ops = num_ops or len(DSL_OPS)
        self.net = nn.Sequential(
            nn.Linear(2*slot_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.num_ops)
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """a,b: (N, D) slot vectors -> logits (N, num_ops)"""
        x = torch.cat([a, b], dim=-1)
        return self.net(x)

    @torch.no_grad()
    def predict_op_bias(self, a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
        logits = self.forward(a, b)
        probs = torch.softmax(logits, dim=-1).mean(dim=0)  # average over N
        return {op: float(probs[i].item()) for i, op in enumerate(DSL_OPS)}

# (Optional) tiny trainer stub for synthetic pairs; integrate with your DSL later.
def sample_synthetic_pairs(encoder, make_pair_fn, n: int, device: str = "cuda"):
    """make_pair_fn should return two images (H,W) that are related by a sampled DSL op, 
    plus the op_name; encoder maps images->slot vectors. This is deliberately abstract here."""
    X1, X2, y = [], [], []
    for _ in range(n):
        img1, img2, op = make_pair_fn()
        z1 = encoder(img1.to(device))
        z2 = encoder(img2.to(device))
        X1.append(z1); X2.append(z2); y.append(op)
    X1 = torch.stack(X1); X2 = torch.stack(X2)
    y_idx = torch.tensor([DSL_OPS.index(op) for op in y], dtype=torch.long, device=device)
    return X1, X2, y_idx
