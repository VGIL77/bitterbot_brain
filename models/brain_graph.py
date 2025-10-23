"""
Brain Graph: Reversible Knowledge Graph with Inheritance and Exceptions

Provides structured symbolic reasoning for ARC-2 via:
- Perceptual search: Observed attributes → Candidate concepts (reverse traversal)
- Attribute recall: Concept → Inherited attributes (forward + inheritance)
- DSL prior generation: Attributes → Op bias + Param logits
- Puzzle embedding: Learnable attribute aggregation

Designed to integrate with HRM-TOPAS bridge for neurosymbolic fusion.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from collections import deque, defaultdict
import torch
import torch.nn as nn
import math

EdgeType = str  # "is_a", "has", "does", "part_of", "exception"


@dataclass
class Node:
    """Graph node representing concepts, attributes, or motifs."""
    name: str
    kind: str  # "concept", "attribute", "motif"
    features: Dict[str, float] = field(default_factory=dict)

    # Outgoing edges (forward links)
    out_edges: Dict[EdgeType, Set[str]] = field(
        default_factory=lambda: {
            "is_a": set(),      # Inheritance (child → parent)
            "has": set(),       # Attributes (concept → attribute)
            "does": set(),      # Actions (concept → operation)
            "part_of": set(),   # Composition (part → whole)
            "exception": set()  # Negations (concept → forbidden attribute)
        }
    )

    # Incoming edges (reverse links for perceptual search)
    in_edges: Dict[EdgeType, Set[str]] = field(
        default_factory=lambda: {
            "is_a": set(),
            "has": set(),
            "does": set(),
            "part_of": set(),
            "exception": set()
        }
    )


class BrainGraph:
    """
    Lightweight knowledge graph with bidirectional traversal.

    Perceptual search: Attributes → Concepts (bottom-up, reverse links)
    Attribute recall: Concept → Attributes (top-down, inheritance)
    """

    def __init__(self, device: Optional[torch.device] = None, puzzle_emb_dim: int = 128, perceptual_backend: str = "auto"):
        self.nodes: Dict[str, Node] = {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Learnable attribute embedding table for puzzle embedding synthesis
        self.puzzle_emb_dim = int(puzzle_emb_dim)
        self._attr_bank: Dict[str, int] = {}  # stable index per attribute

        if self.puzzle_emb_dim > 0:
            self.attr_table = nn.Embedding(4096, self.puzzle_emb_dim).to(self.device)
            # Xavier init for stability
            nn.init.xavier_uniform_(self.attr_table.weight)
        else:
            self.attr_table = None

        # --- NEW: light graph versioning and GPU cache for wave propagation ---
        self._version: int = 0
        self._gpu_cache_version: int = -1
        self._name2idx: Dict[str, int] = {}
        self._idx2name: List[str] = []
        self._rev_adj_coo: Optional[torch.Tensor] = None  # sparse COO [N, N]

        # Toggle: "auto" -> prefer GPU if available, else CPU; "gpu" | "cpu" force backends
        assert perceptual_backend in ("auto", "gpu", "cpu"), f"Invalid backend: {perceptual_backend}"
        self.perceptual_backend: str = perceptual_backend

        # Debug: last perceptual search metrics (does not change API)
        self._last_perceptual_debug: Dict[str, Dict[str, float]] = {}

    # ═══════════════════════════════════════════════════════════
    # Construction API
    # ═══════════════════════════════════════════════════════════

    def add_node(self, name: str, kind: str = "concept", features: Optional[Dict[str, float]] = None):
        """Add node if not exists."""
        if name not in self.nodes:
            self.nodes[name] = Node(name=name, kind=kind, features=features or {})
            self._version += 1

    def add_edge(self, src: str, etype: EdgeType, dst: str):
        """Add directed edge with automatic reverse link."""
        self.add_node(src)
        self.add_node(dst)
        self.nodes[src].out_edges.setdefault(etype, set()).add(dst)
        self.nodes[dst].in_edges.setdefault(etype, set()).add(src)
        self._auto_fix_node_kind(src, etype, dst)  # Auto-type inference
        self._version += 1

    def _auto_fix_node_kind(self, src: str, etype: str, dst: str):
        """Infer node kinds dynamically based on edge semantics."""
        s, d = self.nodes[src], self.nodes[dst]

        # If src uses 'has' → likely concept; dst → attribute
        if etype == "has":
            if s.kind == "concept" and d.kind == "concept":
                d.kind = "attribute"
        # If src uses 'is_a' → both should be concepts
        elif etype == "is_a":
            s.kind, d.kind = "concept", "concept"
        # If src uses 'does' → action edge, dst is likely 'motif'
        elif etype == "does":
            d.kind = "motif"
        # If src uses 'part_of' → both concepts or structural attributes
        elif etype == "part_of":
            if s.kind != "concept":
                s.kind = "concept"
            if d.kind not in ("concept", "motif"):
                d.kind = "concept"
        # If src uses 'exception' → both concept and attribute
        elif etype == "exception":
            if s.kind != "concept":
                s.kind = "concept"
            if d.kind == "concept":
                d.kind = "attribute"

    # Syntactic sugar for common relations
    def is_a(self, child: str, parent: str):
        """Inheritance: child is_a parent."""
        self.add_edge(child, "is_a", parent)

    def has(self, concept: str, attribute: str):
        """Attribution: concept has attribute."""
        self.add_edge(concept, "has", attribute)

    def does(self, concept: str, action: str):
        """Action: concept does action."""
        self.add_edge(concept, "does", action)

    def part_of(self, part: str, whole: str):
        """Composition: part is part_of whole."""
        self.add_edge(part, "part_of", whole)

    def exception(self, concept: str, neg_attr: str):
        """Negation: concept explicitly does NOT have neg_attr."""
        self.add_edge(concept, "exception", neg_attr)

    def reclassify_nodes(self):
        """
        Pass over all edges to retroactively correct node kinds.

        Useful after deserializing a graph or when loading legacy graphs
        where all nodes were initially marked as 'concept'.
        """
        for s_name, node in self.nodes.items():
            for etype, targets in node.out_edges.items():
                for d_name in targets:
                    self._auto_fix_node_kind(s_name, etype, d_name)

    # ═══════════════════════════════════════════════════════════
    # Search Primitives
    # ═══════════════════════════════════════════════════════════

    def perceptual_search(
        self,
        observed: List[str],
        max_hops: int = 3,
        backend: Optional[str] = None,
        return_extras: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Reverse traversal from observed attributes → candidate concepts.

        New (non-breaking):
        - Uses hit-count (#observed that reach a concept) and coverage (hit_count / |observed|) in the score.
        - Can run a GPU wave-propagation backend (multi-source BFS) if available.
        - Keeps the public return type: List[(concept_name, score)]. Detailed metrics are available via
          `get_last_perceptual_debug()` if needed.
        """
        if not observed:
            return []

        observed = [a for a in observed if a in self.nodes]
        if not observed:
            return []

        # Decide backend: "gpu" | "cpu"
        use_backend = backend or self.perceptual_backend
        if use_backend == "auto":
            use_backend = "gpu" if (self.device.type == "cuda" and torch.cuda.is_available()) else "cpu"

        if use_backend == "gpu":
            try:
                items = self._perceptual_search_gpu(observed, max_hops=max_hops)
                if return_extras:
                    return items  # same list type; extras are in self._last_perceptual_debug
                return items
            except Exception as e:
                # Fallback to CPU if anything goes wrong
                print(f"[BrainGraph] GPU perceptual_search failed ({e}); falling back to CPU.")

        # --- CPU reference implementation with hit-count + coverage ---
        q = deque()
        visited = set()
        # Track which observed attribute reached which candidate (for hit-count)
        hits: Dict[str, Set[str]] = defaultdict(set)
        min_depth: Dict[str, int] = defaultdict(lambda: math.inf)

        # Seed one queue entry per observed attribute, keep origin for hit-count
        for attr in observed:
            q.append((attr, 0, attr))  # (node_name, depth, origin_attr)
            visited.add((attr, attr))

        def push_neighbors(nm: str, depth: int, origin_attr: str):
            node = self.nodes.get(nm)
            if node is None:
                return
            # Reverse "has": attribute → concepts that have it
            for src in node.in_edges.get("has", set()):
                key = (src, origin_attr)
                if key not in visited:
                    visited.add(key)
                    q.append((src, depth + 1, origin_attr))

            # Reverse "is_a": parent → children
            for src in node.in_edges.get("is_a", set()):
                key = (src, origin_attr)
                if key not in visited:
                    visited.add(key)
                    q.append((src, depth + 1, origin_attr))

            # Reverse "part_of": parts → wholes
            for src in node.in_edges.get("part_of", set()):
                key = (src, origin_attr)
                if key not in visited:
                    visited.add(key)
                    q.append((src, depth + 1, origin_attr))

        while q:
            node_name, d, origin = q.popleft()
            if d > max_hops:
                continue
            node = self.nodes.get(node_name)
            if node is None:
                continue

            if node.kind == "concept":
                hits[node_name].add(origin)
                if d < min_depth[node_name]:
                    min_depth[node_name] = d

            # Expand neighbors (propagate the same origin)
            push_neighbors(node_name, d, origin)

        # Compute scores with hit-count and coverage
        total_obs = max(1, len(observed))
        scored: List[Tuple[str, float]] = []
        debug: Dict[str, Dict[str, float]] = {}
        for cname, attr_hits in hits.items():
            kc = len(attr_hits)                          # hit-count
            cov = kc / total_obs                         # coverage
            dmin = min_depth[cname] if min_depth[cname] != math.inf else max_hops + 1
            depth_score = 1.0 / (1.0 + float(dmin))      # closer is better
            # Base concept score (kept simple & stable)
            score = kc + 0.5 * cov + 0.5 * depth_score

            # Exception penalty: if concept negates any observed attribute explicitly
            exceptions = self.nodes[cname].out_edges.get("exception", set())
            if any(a in exceptions for a in observed):
                score -= 0.75

            # Prefer concepts with better historical success (if present)
            sr = float(self.nodes[cname].features.get("success_rate", 0.0))
            score += 0.25 * sr

            scored.append((cname, float(score)))
            debug[cname] = {
                "hit_count": float(kc),
                "coverage": float(cov),
                "min_depth": float(dmin),
                "depth_score": float(depth_score),
                "success_rate": sr,
                "score": float(score),
            }

        scored.sort(key=lambda kv: kv[1], reverse=True)
        self._last_perceptual_debug = debug
        return scored[:32]

    def get_last_perceptual_debug(self) -> Dict[str, Dict[str, float]]:
        """Optional inspection of last perceptual_search metrics per concept."""
        return self._last_perceptual_debug

    def attribute_recall(self, concept: str, max_hops: int = 4) -> Dict[str, float]:
        """
        Forward/upward traversal from concept → inherited attributes.

        Follows:
        - Direct "has" links
        - Upward "is_a" (inheritance)
        - Contextual "part_of" (compositional attributes)
        - Applies "exception" (negative overrides)

        Returns:
            Dict of {attribute_name: confidence}
        """
        if concept not in self.nodes:
            return {}

        q = deque([(concept, 0)])
        visited = {concept}
        attrs = defaultdict(float)

        while q:
            nm, d = q.popleft()
            if d > max_hops:
                continue

            node = self.nodes[nm]

            # Direct attributes
            for a in node.out_edges.get("has", set()):
                attrs[a] += 1.0 / (1.0 + d)

            # Exceptions (negative facts suppress attributes)
            for neg in node.out_edges.get("exception", set()):
                attrs[neg] -= 1.0 / (1.0 + d)

            # Climb "is_a" hierarchy (inherit from parents)
            for parent in node.out_edges.get("is_a", set()):
                if parent not in visited:
                    visited.add(parent)
                    q.append((parent, d + 1))

            # Also climb "part_of" for contextual attributes
            for whole in node.out_edges.get("part_of", set()):
                if whole not in visited:
                    visited.add(whole)
                    q.append((whole, d + 1))

        # Clamp negatives to 0
        return {k: float(max(0.0, v)) for k, v in attrs.items() if v > 0.0}

    # ═══════════════════════════════════════════════════════════
    # GPU Wave Propagation Backend
    # ═══════════════════════════════════════════════════════════

    def _ensure_gpu_graph(self, reverse_edge_types: Tuple[str, ...] = ("has", "is_a", "part_of")):
        """Build or refresh sparse reverse adjacency for multi-source BFS on GPU."""
        if self._gpu_cache_version == self._version and self._rev_adj_coo is not None:
            return
        # Stable indexing
        self._name2idx = {nm: i for i, nm in enumerate(self.nodes.keys())}
        self._idx2name = [nm for nm in self.nodes.keys()]

        src_idx: List[int] = []
        dst_idx: List[int] = []
        # Edge: current node -> its reverse neighbors (so A[row=curr, col=neighbor] = 1)
        for nm, node in self.nodes.items():
            r = self._name2idx[nm]
            for et in reverse_edge_types:
                for neighbor in node.in_edges.get(et, set()):
                    if neighbor in self._name2idx:
                        c = self._name2idx[neighbor]
                        src_idx.append(r)
                        dst_idx.append(c)

        N = max(1, len(self._name2idx))
        if len(src_idx) == 0:
            # Empty graph is valid
            indices = torch.empty((2, 0), dtype=torch.long, device=self.device)
            values = torch.empty((0,), dtype=torch.float32, device=self.device)
        else:
            indices = torch.stack([
                torch.tensor(src_idx, dtype=torch.long, device=self.device),
                torch.tensor(dst_idx, dtype=torch.long, device=self.device)
            ], dim=0)
            values = torch.ones((len(src_idx),), dtype=torch.float32, device=self.device)

        self._rev_adj_coo = torch.sparse_coo_tensor(indices, values, (N, N), device=self.device).coalesce()
        self._gpu_cache_version = self._version

    def _perceptual_search_gpu(self, observed: List[str], max_hops: int = 3) -> List[Tuple[str, float]]:
        """Multi-source BFS on GPU with hit-count and coverage."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        self._ensure_gpu_graph()
        if self._rev_adj_coo is None:
            return []

        # Map observed to indices, keep only present ones
        seed_idx = [self._name2idx[a] for a in observed if a in self._name2idx]
        if not seed_idx:
            return []

        N = len(self._idx2name)
        O = len(seed_idx)
        device = self.device

        # Seed matrix S: [N, O], one column per observed attribute
        S = torch.zeros((N, O), dtype=torch.float32, device=device)
        S[torch.tensor(seed_idx, dtype=torch.long, device=device), torch.arange(O, device=device)] = 1.0

        visited = S > 0
        depths = torch.full((N, O), fill_value=max_hops + 1, dtype=torch.int16, device=device)
        depths[torch.tensor(seed_idx, dtype=torch.long, device=device), torch.arange(O, device=device)] = 0
        frontier = S  # float

        for d in range(1, max_hops + 1):
            # next_frontier = A @ frontier
            nxt = torch.sparse.mm(self._rev_adj_coo, frontier)
            nxt = nxt > 0  # boolean
            new = nxt & (~visited)
            if not new.any():
                break
            depths.masked_fill_(new, d)
            visited = visited | new
            frontier = new.float()

        # Concept mask
        concept_mask = torch.tensor(
            [1 if self.nodes[nm].kind == "concept" else 0 for nm in self._idx2name],
            dtype=torch.bool, device=device
        )
        Vc = visited[concept_mask]          # [Nc, O]
        Dc = depths[concept_mask]           # [Nc, O]

        hit_counts = Vc.sum(dim=1).to(torch.float32)              # [Nc]
        coverage = hit_counts / float(O)                          # [Nc]

        # min depth across observed sources (treat unreachable as +inf via sentinel)
        sentinel = (max_hops + 1)
        Dc_masked = Dc.clone()
        Dc_masked[Dc_masked == sentinel] = 32767  # large
        min_depth = Dc_masked.min(dim=1).values.to(torch.float32)
        min_depth[min_depth >= 32767] = float(sentinel)
        depth_score = 1.0 / (1.0 + min_depth)                     # [Nc]

        # Base score: hit-count + small boosts from coverage & depth
        scores = hit_counts + 0.5 * coverage + 0.5 * depth_score  # [Nc]

        # Exception penalty + success_rate (done on CPU loop for clarity)
        names = [nm for nm in self._idx2name if self.nodes[nm].kind == "concept"]
        scored: List[Tuple[str, float]] = []
        debug: Dict[str, Dict[str, float]] = {}
        obs_set = set(observed)
        for i, nm in enumerate(names):
            s = float(scores[i].item())
            exceptions = self.nodes[nm].out_edges.get("exception", set())
            if any(a in exceptions for a in obs_set):
                s -= 0.75
            sr = float(self.nodes[nm].features.get("success_rate", 0.0))
            s += 0.25 * sr
            kc = float(hit_counts[i].item())
            cv = float(coverage[i].item())
            md = float(min_depth[i].item())
            dp = float(depth_score[i].item())
            scored.append((nm, s))
            debug[nm] = {
                "hit_count": kc,
                "coverage": cv,
                "min_depth": md,
                "depth_score": dp,
                "success_rate": sr,
                "score": s,
            }

        scored.sort(key=lambda kv: kv[1], reverse=True)
        self._last_perceptual_debug = debug
        return scored[:32]

    # ═══════════════════════════════════════════════════════════
    # ARC-2 Helpers: Produce DSL Priors
    # ═══════════════════════════════════════════════════════════

    def _attr_index(self, name: str) -> int:
        """Get stable index for attribute (for embedding lookup)."""
        if name not in self._attr_bank:
            self._attr_bank[name] = len(self._attr_bank) % 4096
        return self._attr_bank[name]

    def to_puzzle_embedding(self, attrs: Dict[str, float]) -> Optional[torch.Tensor]:
        """
        Synthesize puzzle embedding from weighted attributes.

        Args:
            attrs: {attribute_name: confidence}

        Returns:
            [puzzle_emb_dim] tensor or None
        """
        if self.attr_table is None or not attrs:
            return None

        names = list(attrs.keys())
        idx = torch.tensor([self._attr_index(a) for a in names],
                          device=self.device, dtype=torch.long)
        weights = torch.tensor([attrs[a] for a in names],
                              device=self.device, dtype=torch.float32)

        # Weighted average of attribute embeddings
        emb = self.attr_table(idx)  # [K, D]
        weights = (weights / (weights.sum() + 1e-6)).unsqueeze(-1)  # [K, 1]
        vec = (emb * weights).sum(dim=0, keepdim=False)  # [D]

        return vec

    def to_dsl_priors(self, concept: str, recalled_attrs: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """
        Map recalled attributes to DSL operation biases and parameter priors.
        Enhanced to cover all 41 DSL operations with intelligent attribute→op rules.

        Returns:
            (op_bias, param_priors)
            - op_bias: {op_name: bias_weight}
            - param_priors: {param_name: logits_tensor}
        """
        op_bias = {}
        param_priors = {}

        # === GEOMETRIC TRANSFORMATIONS ===
        if "symmetry_quadrant" in recalled_attrs or "symmetry_h" in recalled_attrs or "symmetry_v" in recalled_attrs:
            op_bias["rotate90"] = 0.6
            op_bias["rotate180"] = 0.5
            op_bias["rotate270"] = 0.4
            op_bias["flip_h"] = 0.7 if "symmetry_h" in recalled_attrs else 0.5
            op_bias["flip_v"] = 0.7 if "symmetry_v" in recalled_attrs else 0.5
            op_bias["symmetry"] = 0.6

        # === COLOR OPERATIONS ===
        if "needs_color_map" in recalled_attrs or "color_count_change" in recalled_attrs:
            op_bias["color_map"] = 0.8
            # Color map param priors (favor common ARC colors: 0,1,2,3,5,7)
            color_logits = torch.tensor(
                [0.15, 0.15, 0.15, 0.15, 0.05, 0.15, 0.05, 0.15, 0.05, 0.05],
                device=self.device, dtype=torch.float32
            ).unsqueeze(0)
            param_priors["color_map_logits"] = color_logits

        if "color_reduction" in recalled_attrs:
            op_bias["extract_color"] = 0.6

        # === SPATIAL TRANSFORMATIONS ===
        if "translation" in recalled_attrs:
            op_bias["translate"] = 0.7
            # Translation priors: favor small offsets ([-2, -1, 0, +1, +2])
            dx_logits = torch.tensor([0.15, 0.25, 0.20, 0.25, 0.15],
                                    device=self.device, dtype=torch.float32).unsqueeze(0)
            param_priors["dx_logits"] = dx_logits
            param_priors["dy_logits"] = dx_logits  # Symmetric

        if "scaling" in recalled_attrs:
            op_bias["scale"] = 0.6
            op_bias["resize_nn"] = 0.5

        if "grid_crop" in recalled_attrs:
            op_bias["crop_bbox"] = 0.7
            op_bias["crop_nonzero"] = 0.6

        if "dense_grid" in recalled_attrs:
            op_bias["center_pad_to"] = 0.5

        # === PATTERN OPERATIONS ===
        if "pattern_fill" in recalled_attrs or "tiling_pattern" in recalled_attrs:
            op_bias["flood_fill"] = 0.7
            op_bias["tile"] = 0.6
            op_bias["tile_pattern"] = 0.6
            op_bias["paste"] = 0.5

        # === OBJECT OPERATIONS ===
        if "object_manipulation" in recalled_attrs or "sparse_objects" in recalled_attrs:
            op_bias["extract_objects"] = 0.5
            op_bias["for_each_object"] = 0.6
            op_bias["for_each_object_translate"] = 0.5
            op_bias["for_each_object_recolor"] = 0.5
            op_bias["for_each_object_rotate"] = 0.4
            op_bias["for_each_object_scale"] = 0.4
            op_bias["for_each_object_flip"] = 0.4
            op_bias["select_by_property"] = 0.5

        # === ANALYSIS OPERATIONS ===
        if "complex_palette" in recalled_attrs or "simple_palette" in recalled_attrs:
            op_bias["count_colors"] = 0.4
            op_bias["find_pattern"] = 0.5

        if "sparse_objects" in recalled_attrs:
            op_bias["count_objects"] = 0.5

        # === LOGIC OPERATIONS ===
        if "shape_preserved" in recalled_attrs and "mass_conserved" in recalled_attrs:
            # Likely pixel-level logic operations
            op_bias["grid_union"] = 0.4
            op_bias["grid_intersection"] = 0.4
            op_bias["grid_xor"] = 0.3
            op_bias["grid_difference"] = 0.3

        # === BOUNDARY/OUTLINE ===
        if "dense_grid" in recalled_attrs or "sparse_objects" in recalled_attrs:
            op_bias["outline"] = 0.6
            op_bias["boundary_extract"] = 0.5

        # === PATTERN MATCHING ===
        if "tiling_pattern" in recalled_attrs:
            op_bias["match_template"] = 0.5
            op_bias["extract_pattern"] = 0.5
            op_bias["find_pattern"] = 0.6

        # === CONDITIONAL/ADVANCED ===
        if "complex_palette" in recalled_attrs:
            op_bias["apply_rule"] = 0.4
            op_bias["conditional_map"] = 0.5

        if "sparse_objects" in recalled_attrs:
            op_bias["flood_select"] = 0.5
            op_bias["arithmetic_op"] = 0.3

        # Safe fallback if no attributes matched
        if not op_bias:
            op_bias["identity"] = 0.1

        return op_bias, param_priors

    # ═══════════════════════════════════════════════════════════
    # High-Level Inference API
    # ═══════════════════════════════════════════════════════════

    def infer(self, observed_attrs: List[str]) -> Dict[str, object]:
        """
        Complete inference pipeline from observations to DSL priors.

        Args:
            observed_attrs: List of observed attribute names

        Returns:
            {
                'concept': str,
                'recalled_attrs': Dict[str, float],
                'op_bias': Dict[str, float],
                'param_priors': Dict[str, torch.Tensor],
                'puzzle_embedding': Optional[torch.Tensor]
            }
        """
        # Step 1: Perceptual search (reverse traversal)
        candidates = self.perceptual_search(observed_attrs)

        if not candidates:
            return {
                "concept": None,
                "recalled_attrs": {},
                "op_bias": {},
                "param_priors": {},
                "puzzle_embedding": None
            }

        # Step 2: Select top candidate
        concept, _score = candidates[0]

        # Step 3: Attribute recall (forward/upward traversal)
        recalled = self.attribute_recall(concept)

        # Step 4: Generate DSL priors
        op_bias, param_priors = self.to_dsl_priors(concept, recalled)

        # Step 5: Synthesize puzzle embedding
        puzzle_emb = self.to_puzzle_embedding(recalled)

        return {
            "concept": concept,
            "recalled_attrs": recalled,
            "op_bias": op_bias,
            "param_priors": param_priors,
            "puzzle_embedding": puzzle_emb
        }

    def observe_success(self, puzzle_attrs: List[str], dsl_ops_used: List[str], success: bool, task_id: Optional[str] = None):
        """
        Learn from successful (or failed) DSL programs during training.
        This is the ETHICAL learning approach - no hardcoded patterns, pure experience-based learning.

        Args:
            puzzle_attrs: Observed attributes of the puzzle (e.g., ["has_symmetry", "color_palette_3"])
            dsl_ops_used: DSL operations that were used (e.g., ["rotate90", "mirror_h"])
            success: Whether the program succeeded (True) or failed (False)
            task_id: Optional task identifier for tracking

        This implements Hebbian-style learning: "patterns that fire together, wire together"
        """
        if not success or not puzzle_attrs or not dsl_ops_used:
            return

        # Create a concept node for this pattern (hash-based unique ID)
        attr_signature = tuple(sorted(puzzle_attrs))
        concept_id = f"learned_pattern_{hash(attr_signature) % 100000}"

        self.add_node(concept_id, kind="concept")

        # Associate observed attributes with this concept
        for attr in puzzle_attrs:
            self.add_node(attr, kind="attribute")
            self.has(concept_id, attr)

        # Associate successful operations with this concept
        for op in dsl_ops_used:
            self.add_node(op, kind="operation")
            self.does(concept_id, op)

        # Track success count as a feature for future weighting
        if concept_id in self.nodes:
            features = self.nodes[concept_id].features
            features['success_count'] = features.get('success_count', 0) + 1
            features['total_count'] = features.get('total_count', 0) + 1
            features['success_rate'] = features['success_count'] / features['total_count']


def seed_arc2_knowledge(graph: BrainGraph):
    """
    Seed brain graph with common ARC-2 patterns and transformations.

    This provides initial structured knowledge for neurosymbolic reasoning.
    Extendable with discovered patterns during training.
    """
    # ═══════════════════════════════════════════════════════════
    # Core Transformation Concepts
    # ═══════════════════════════════════════════════════════════

    # Symmetry patterns
    graph.add_node("symmetry_transform", kind="concept")
    graph.has("symmetry_transform", "symmetry_quadrant")
    graph.has("symmetry_transform", "reflection_axis")
    graph.does("symmetry_transform", "rotate90")
    graph.does("symmetry_transform", "flip_h")
    graph.does("symmetry_transform", "flip_v")

    # Color mapping
    graph.add_node("color_transformation", kind="concept")
    graph.has("color_transformation", "needs_color_map")
    graph.has("color_transformation", "palette_change")
    graph.does("color_transformation", "color_map")

    # Translation
    graph.add_node("spatial_shift", kind="concept")
    graph.has("spatial_shift", "translation")
    graph.has("spatial_shift", "position_offset")
    graph.does("spatial_shift", "translate")

    # Scaling
    graph.add_node("size_change", kind="concept")
    graph.has("size_change", "scaling")
    graph.has("size_change", "grid_resize")
    graph.does("size_change", "scale")
    graph.does("size_change", "resize_nn")

    # Pattern filling
    graph.add_node("fill_operation", kind="concept")
    graph.has("fill_operation", "pattern_fill")
    graph.has("fill_operation", "flood_region")
    graph.does("fill_operation", "flood_fill")
    graph.does("fill_operation", "fill_pattern")

    # Object manipulation
    graph.add_node("object_transform", kind="concept")
    graph.has("object_transform", "object_manipulation")
    graph.has("object_transform", "per_object_rule")
    graph.does("object_transform", "extract_objects")
    graph.does("object_transform", "for_each_object")

    # Grid cropping
    graph.add_node("grid_extraction", kind="concept")
    graph.has("grid_extraction", "grid_crop")
    graph.has("grid_extraction", "bbox_focus")
    graph.does("grid_extraction", "crop_bbox")

    # Outline detection
    graph.add_node("boundary_detection", kind="concept")
    graph.has("boundary_detection", "outline_detection")
    graph.has("boundary_detection", "edge_emphasis")
    graph.does("boundary_detection", "outline")

    # ═══════════════════════════════════════════════════════════
    # Compositional Patterns (inheritance hierarchy)
    # ═══════════════════════════════════════════════════════════

    # Quadrant operations inherit from symmetry
    graph.add_node("quadrant_split", kind="concept")
    graph.is_a("quadrant_split", "symmetry_transform")
    graph.has("quadrant_split", "four_way_division")
    graph.does("quadrant_split", "crop_bbox")
    graph.does("quadrant_split", "rotate90")

    # Checkerboard patterns
    graph.add_node("checkerboard_pattern", kind="concept")
    graph.is_a("checkerboard_pattern", "pattern_fill")
    graph.has("checkerboard_pattern", "alternating_colors")
    graph.does("checkerboard_pattern", "fill_pattern")

    # Mirror composition
    graph.add_node("mirror_composition", kind="concept")
    graph.is_a("mirror_composition", "symmetry_transform")
    graph.has("mirror_composition", "bilateral_symmetry")
    graph.does("mirror_composition", "flip_h")
    graph.does("mirror_composition", "flip_v")

    # ═══════════════════════════════════════════════════════════
    # Exceptions (negative facts)
    # ═══════════════════════════════════════════════════════════

    # Symmetry transforms typically don't need color mapping
    graph.exception("symmetry_transform", "needs_color_map")

    # Pure color transforms don't need spatial operations
    graph.exception("color_transformation", "translation")
    graph.exception("color_transformation", "scaling")

    return graph
