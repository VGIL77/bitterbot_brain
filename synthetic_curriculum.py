
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
