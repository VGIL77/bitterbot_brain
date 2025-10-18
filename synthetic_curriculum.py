
# synthetic_curriculum.py
# Tiny curriculum generator for 1-step and 2-step DSL programs using your registry.
from typing import List, Dict, Tuple, Any, Optional
import random
import torch
from models.dsl_registry import DSL_OPS
try:
    from models.dsl_search import DSLProgram, apply_program  # adjust import path if needed
except Exception:
    DSLProgram = None
    def apply_program(grid, program): 
        return grid  # placeholder

COLORS = list(range(0,10))

def random_grid(h:int=10, w:int=10, density:float=0.15) -> torch.Tensor:
    g = torch.zeros(h,w, dtype=torch.long)
    for i in range(h):
        for j in range(w):
            if random.random() < density:
                g[i,j] = random.choice(COLORS[1:])  # non-zero color
    return g

def sample_program(depth:int=1) -> Tuple[List[str], List[Dict[str,Any]]]:
    ops = []
    params = []
    pool = [op for op in DSL_OPS if op not in {"apply_rule","conditional_map"}]  # keep simple
    for _ in range(depth):
        op = random.choice(pool)
        ops.append(op)
        # extremely small param dictionary; real code should use valid params per op
        params.append({})
    return ops, params

def make_task(max_depth:int=2):
    depth = random.choice([1,1,2]) if max_depth>=2 else 1
    inp = random_grid()
    ops, params = sample_program(depth)
    program = DSLProgram(ops=ops, params=params) if DSLProgram else None
    out = apply_program(inp, program) if program else inp.clone()
    return {"input": inp, "output": out, "ops": ops, "params": params}

# ============ WGO SUPERVISION LABELS ============

def encode_program_as_tokens(ops, pretrainer_vocab, max_length=8):
    """
    Convert operation list to token sequence for WGO program head supervision.

    Args:
        ops: List of operation names (e.g., ['flip_v', 'color_map'])
        pretrainer_vocab: WGO vocabulary dict {op_name: token_id}
        max_length: Maximum sequence length

    Returns:
        Token tensor [max_length]
    """
    tokens = [pretrainer_vocab.get('<START>', 1)]  # START token

    for op in ops[:max_length-2]:  # Leave room for START/END
        token_id = pretrainer_vocab.get(op, pretrainer_vocab.get('<UNK>', 3))
        tokens.append(token_id)

    tokens.append(pretrainer_vocab.get('<END>', 2))  # END token

    # Pad to max_length
    pad_token = pretrainer_vocab.get('<PAD>', 0)
    while len(tokens) < max_length:
        tokens.append(pad_token)

    return torch.tensor(tokens[:max_length], dtype=torch.long)

def classify_size_transform(input_shape, output_shape):
    """
    Classify size transformation type for WGO size head supervision.

    Returns class ID:
      0: no_change (H,W same)
      1: uniform_scale_up
      2: nonuniform_scale_up
      3: uniform_scale_down
      4: nonuniform_scale_down
      5: padding/extension
      6: cropping/reduction
    """
    h_in, w_in = input_shape
    h_out, w_out = output_shape

    # No change
    if h_in == h_out and w_in == w_out:
        return 0

    # Scale up
    if h_out > h_in and w_out > w_in:
        ratio_match = abs((h_out/h_in) - (w_out/w_in)) < 0.1
        return 1 if ratio_match else 2

    # Scale down
    if h_out < h_in and w_out < w_in:
        ratio_match = abs((h_out/h_in) - (w_out/w_in)) < 0.1
        return 3 if ratio_match else 4

    # Padding (one or both dimensions increased)
    if h_out >= h_in or w_out >= w_in:
        return 5

    # Cropping
    return 6

def detect_symmetry_in_ops(ops):
    """
    Detect primary symmetry operation for WGO symmetry head supervision.

    Returns class ID:
      0: none
      1: rotate90
      2: rotate180
      3: rotate270
      4: flip_h
      5: flip_v
      6: translate
      7: identity
    """
    symmetry_map = {
        'rotate90': 1,
        'rotate180': 2,
        'rotate270': 3,
        'flip_h': 4,
        'flip_v': 5,
        'translate': 6,
        'identity': 7
    }

    # Return first symmetry operation found
    for op in ops:
        if op in symmetry_map:
            return symmetry_map[op]

    return 0  # no_symmetry

def compute_histogram(grid):
    """
    Compute normalized color histogram for WGO histogram head supervision.

    Args:
        grid: Tensor [H, W]

    Returns:
        Histogram tensor [10] for colors 0-9
    """
    if grid.numel() == 0:
        return torch.zeros(10)

    hist = torch.zeros(10, dtype=torch.float32)
    unique, counts = torch.unique(grid.long().clamp(0, 9), return_counts=True)
    hist[unique] = counts.float()

    # Normalize
    return hist / grid.numel()

def make_supervised_task(max_depth=3, pretrainer_vocab=None):
    """
    Generate synthetic task with complete supervision labels for WGO training.

    Args:
        max_depth: Maximum DSL program depth
        pretrainer_vocab: WGO operation vocabulary (pass from pretrainer.op_vocab)

    Returns:
        Task dict with input/output grids and supervision labels:
          - input: Grid [H, W]
          - output: Grid [H, W]
          - program_tokens: Token sequence [8]
          - size_class: Size transform class (0-6)
          - symmetry_class: Symmetry class (0-7)
          - color_histogram: Input/output histograms [2, 10]
          - critic: Success indicator (1.0)
    """
    # Generate base task
    base_task = make_task(max_depth)

    # Default vocab if not provided
    if pretrainer_vocab is None:
        pretrainer_vocab = {
            '<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3,
            'rotate90': 4, 'rotate180': 5, 'flip_v': 6, 'flip_h': 7,
            'translate': 8, 'color_map': 9, 'identity': 10
        }

    # Add supervision labels
    supervised_task = {
        'input': base_task['input'],
        'output': base_task['output'],
        'ops': base_task['ops'],
        'params': base_task['params'],

        # WGO supervision labels
        'program_tokens': encode_program_as_tokens(
            base_task['ops'], pretrainer_vocab, max_length=8
        ),
        'size_class': classify_size_transform(
            base_task['input'].shape,
            base_task['output'].shape
        ),
        'symmetry_class': detect_symmetry_in_ops(base_task['ops']),
        'color_histogram': torch.stack([
            compute_histogram(base_task['input']),
            compute_histogram(base_task['output'])
        ]),
        'critic': 1.0  # Synthetic tasks always succeed
    }

    return supervised_task
