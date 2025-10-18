from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class NeuroPlannerInnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class NeuroPlannerCarry:
    inner_carry: NeuroPlannerInnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class NeuroPlannerConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"


class NeuroPlannerBlock(nn.Module):
    def __init__(self, config: NeuroPlannerConfig) -> None:
        super().__init__()

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm
        # Self Attention
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class NeuroPlannerReasoningModule(nn.Module):
    def __init__(self, layers: List[NeuroPlannerBlock]):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states


class NeuroPlanner_Inner(nn.Module):
    def __init__(self, config: NeuroPlannerConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale  = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        # ARC-II: puzzle_emb_len = 0 (injection via addition, not concatenation)
        self.puzzle_emb_len = 0
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            # ARC-II: RoPE max_pos = seq_len (no +puzzle_emb_len since we use addition)
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            # ARC-II: Learned embeddings = seq_len (no +puzzle_emb_len)
            self.embed_pos = CastedEmbedding(self.config.seq_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            raise NotImplementedError()

        # Reasoning Layers
        self.H_level = NeuroPlannerReasoningModule(layers=[NeuroPlannerBlock(self.config) for _i in range(self.config.H_layers)])
        self.L_level = NeuroPlannerReasoningModule(layers=[NeuroPlannerBlock(self.config) for _i in range(self.config.L_layers)])
        
        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings - INJECT VIA ADDITION (not concatenation)
        # This avoids phantom +1 token that breaks carry dimension matching
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)  # [B, puzzle_emb_ndim]

            # Pad to hidden_size if needed
            if puzzle_embedding.shape[-1] < self.config.hidden_size:
                pad_count = self.config.hidden_size - puzzle_embedding.shape[-1]
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            elif puzzle_embedding.shape[-1] > self.config.hidden_size:
                # Project down if needed
                puzzle_embedding = puzzle_embedding[..., :self.config.hidden_size]

            # Inject into first token via addition (not concatenation)
            # This keeps seq_len = actual_tokens, no phantom +1
            embedding[:, 0, :] = embedding[:, 0, :] + puzzle_embedding

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # Slice learned positional embeddings to actual seq_len (ARC-2 variable sizes)
            seq_len = embedding.shape[1]
            pos_emb = self.embed_pos.embedding_weight[:seq_len].to(self.forward_dtype)
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + pos_emb)

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int, seq_len: int = None):
        """
        Create an inner carry. If seq_len is provided, use that length (useful for per-example allocation).
        Otherwise fall back to config-defined max length.
        """
        try:
            device = next(self.parameters()).device
        except StopIteration:
            # Force GPU mode - no CPU fallback
            device = torch.device("cuda")

        if seq_len is None:
            seq_len = getattr(self.config, "seq_len", None)
            if hasattr(self, "puzzle_emb_len"):
                seq_len = (seq_len or 0) + getattr(self, "puzzle_emb_len", 0)

        seq_len = int(seq_len)

        return NeuroPlannerInnerCarry(
            z_H=torch.empty(batch_size, seq_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            z_L=torch.empty(batch_size, seq_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: NeuroPlannerInnerCarry):
        return NeuroPlannerInnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: NeuroPlannerInnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[NeuroPlannerInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Get actual sequence length from batch inputs (ARC-2 grids vary from 3×3 to 30×30)
        actual_seq_len = batch["inputs"].shape[1] if batch["inputs"].ndim >= 2 else self.config.seq_len

        # Input encoding (puzzle_emb injected via addition, so seq_len stays == actual_seq_len)
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # ARC-II: Check if carry dimensions match embedding dimensions
        # If mismatch (e.g., config changed between calls), rebuild carry
        expected_seq_len = input_embeddings.shape[1]
        if carry.z_H.shape[1] != expected_seq_len:
            batch_size = carry.z_H.shape[0]
            carry = self.empty_carry(batch_size, seq_len=expected_seq_len)

        # Dynamic RoPE slicing (only allocate what we need)
        seq_info = dict(
            cos_sin=self.rotary_emb(seq_len=expected_seq_len) if hasattr(self, "rotary_emb") else None,
        )

        # Forward iterations - detached for memory efficiency
        # Only the final 1-step pass (below) carries gradients
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

                if not (_H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, **seq_info)

        # 1-step gradient pass (GRADIENT FLOW enabled here)
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        # LM Outputs (keep gradients for TOPAS feedback)
        new_carry = NeuroPlannerInnerCarry(z_H=z_H, z_L=z_L)  # Gradients flow to HRM
        # ARC-II: No slicing needed (puzzle_emb_len=0, injected via addition)
        output = self.lm_head(z_H)

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class NeuroPlanner(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = NeuroPlannerConfig(**config_dict)
        self.inner = NeuroPlanner_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device
        # Use actual input length to allocate inner carry to the required size.
        seq_len = batch["inputs"].shape[1] if batch["inputs"].ndim >= 2 else None

        return NeuroPlannerCarry(
            inner_carry=self.inner.empty_carry(batch_size, seq_len=seq_len),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: NeuroPlannerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[NeuroPlannerCarry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):
                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)

                halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q
                # NOTE: No replay buffer and target networks for computing target Q-value.
                # As batch_size is large, there're many parallel envs.
                # Similar concept as PQN https://arxiv.org/abs/2407.04811
                next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data)[-1]
                
                outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return NeuroPlannerCarry(new_inner_carry, new_steps, halted, new_current_data), outputs