"""
PUCT (Predictor + Upper Confidence bounds applied to Trees) Search Implementation
Neural-guided Monte Carlo Tree Search for program synthesis

This implementation provides:
- MCTSNode class for tree structure
- PUCT selection formula with exploration bonus
- Neural policy and value network integration
- Dirichlet noise for exploration at root
- Parameter-aware DSL operation expansion
- Efficient tree traversal and backpropagation
- Production-ready integration with existing DSL system
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import time
import logging
from collections import defaultdict

# Import DSL operations and registry
from models.dsl_search import DSLProgram, apply_program, generate_op_parameters, CORE_OPS
from models.dsl_registry import DSL_OPS, get_op_index, NUM_DSL_OPS

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class MCTSNode:
    """
    Monte Carlo Tree Search Node for program synthesis

    Represents a partial program state in the search tree with:
    - Program operations and parameters accumulated so far
    - Visit statistics for PUCT calculation
    - Policy priors from neural network
    - Value estimates and bounds
    """
    # Program state
    ops: List[str] = field(default_factory=list)
    params: List[Dict[str, Any]] = field(default_factory=list)
    depth: int = 0

    # MCTS statistics
    visit_count: int = 0
    total_value: float = 0.0
    mean_value: float = 0.0

    # Neural network outputs
    policy_priors: Optional[torch.Tensor] = None  # P(a|s) from policy network
    value_estimate: float = 0.0  # V(s) from value network

    # Tree structure
    parent: Optional['MCTSNode'] = None
    children: Dict[Tuple[str, frozenset], 'MCTSNode'] = field(default_factory=dict)

    # State properties
    is_terminal: bool = False
    is_solved: bool = False
    terminal_reward: float = 0.0

    # Caching for efficiency
    program_hash: str = ""
    cached_results: Dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize computed fields"""
        if not self.program_hash:
            ops_str = "|".join(self.ops)
            params_str = "|".join(str(p) for p in self.params)
            self.program_hash = f"{ops_str}#{params_str}"

    def get_program(self) -> DSLProgram:
        """Convert node state to DSLProgram"""
        return DSLProgram(ops=self.ops.copy(), params=self.params.copy())

    def apply_program(self, grid: torch.Tensor) -> torch.Tensor:
        """Apply the program represented by this node to a grid"""
        program = self.get_program()
        return apply_program(grid, program)

    def is_fully_expanded(self, available_ops: List[str], max_depth: int) -> bool:
        """Check if all possible actions have been tried from this node"""
        if self.depth >= max_depth or self.is_terminal:
            return True

        # Count expected number of expansions
        expected_children = 0
        for op in available_ops:
            param_options = generate_op_parameters(op, None)  # Use None context for now
            expected_children += len(param_options)

        return len(self.children) >= expected_children

    def get_action_key(self, op: str, params: Dict[str, Any]) -> Tuple[str, frozenset]:
        """Create a hashable key for an action (op, params pair)"""
        # Convert params dict to frozenset of items for hashing
        # Handle nested unhashable types (dicts, lists, tensors)
        if params:
            hashable_items = []
            for k, v in params.items():
                if isinstance(v, dict):
                    # Recursively convert nested dicts
                    v_hashable = frozenset(v.items()) if v else frozenset()
                elif isinstance(v, (list, tuple)):
                    # Convert lists/tuples to tuples
                    v_hashable = tuple(v)
                elif torch.is_tensor(v):
                    # Convert tensors to tuple of values
                    v_hashable = tuple(v.flatten().tolist())
                else:
                    v_hashable = v
                hashable_items.append((k, v_hashable))
            param_items = frozenset(hashable_items)
        else:
            param_items = frozenset()
        return (op, param_items)

    def add_child(self, op: str, params: Dict[str, Any]) -> 'MCTSNode':
        """Add a child node for the given action"""
        action_key = self.get_action_key(op, params)

        if action_key not in self.children:
            child = MCTSNode(
                ops=self.ops + [op],
                params=self.params + [params],
                depth=self.depth + 1,
                parent=self
            )
            self.children[action_key] = child

        return self.children[action_key]

    def update_value(self, value: float):
        """Update visit statistics with new value"""
        self.visit_count += 1
        self.total_value += value
        self.mean_value = self.total_value / self.visit_count

    def get_puct_score(self, c_puct: float, parent_visits: int) -> float:
        """
        Calculate PUCT score for node selection

        PUCT(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Where:
        - Q(s,a) is the mean action value (exploitation)
        - P(s,a) is the policy prior probability
        - N(s) is parent visit count, N(s,a) is child visit count
        - c_puct controls exploration vs exploitation balance
        """
        if self.visit_count == 0:
            return float('inf')  # Unvisited nodes get highest priority

        # Q(s,a) - mean value (exploitation term)
        exploitation = self.mean_value

        # MDL PENALTY: Prefer shorter programs (ARC-AGI-2 simplicity bias)
        mdl_penalty = 0.02 * self.depth  # Penalize by program depth
        exploitation = exploitation - mdl_penalty

        # P(s,a) * sqrt(N(s)) / (1 + N(s,a)) - exploration term
        if self.parent is not None and self.parent.policy_priors is not None:
            # Find this node's policy prior from parent
            # This is a simplified version - in practice, you'd map actions to policy indices
            prior_prob = 1.0 / len(self.parent.children) if len(self.parent.children) > 0 else 1.0
        else:
            prior_prob = 1.0  # Default uniform prior

        exploration = c_puct * prior_prob * math.sqrt(parent_visits) / (1.0 + self.visit_count)

        return exploitation + exploration

    def select_best_child(self, c_puct: float) -> 'MCTSNode':
        """Select child with highest PUCT score"""
        if not self.children:
            return None

        best_child = None
        best_score = -float('inf')

        for child in self.children.values():
            score = child.get_puct_score(c_puct, self.visit_count)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def get_visit_counts(self) -> Dict[Tuple[str, frozenset], int]:
        """Get visit counts for all children (useful for policy training)"""
        return {action_key: child.visit_count for action_key, child in self.children.items()}

    def __repr__(self) -> str:
        return f"MCTSNode(depth={self.depth}, ops={self.ops}, visits={self.visit_count}, value={self.mean_value:.3f})"


class PUCTSearcher:
    """
    PUCT Search implementation for neural-guided program synthesis

    Combines Monte Carlo Tree Search with neural policy and value networks
    to efficiently explore the space of DSL programs.
    """

    def __init__(
        self,
        policy_net: torch.nn.Module,
        value_net: torch.nn.Module,
        topas_model: torch.nn.Module,
        c_puct: float = 1.4,
        dirichlet_alpha: float = 0.3,
        dirichlet_noise_weight: float = 0.25,
        max_depth: int = 10,
        temperature: float = 1.0,
        device: torch.device = None,
        op_bias: Optional[Dict[str, float]] = None
    ):
        """
        Initialize PUCT searcher

        Args:
            policy_net: OpPolicyNet that outputs policy priors P(a|s)
            value_net: ValueNet that outputs value estimates V(s)
            topas_model: TOPAS model for feature extraction
            c_puct: Exploration constant for PUCT formula
            dirichlet_alpha: Alpha parameter for Dirichlet noise
            dirichlet_noise_weight: Weight of Dirichlet noise vs policy prior
            max_depth: Maximum program depth to search
            temperature: Temperature for action selection
            device: Torch device for computations
            op_bias: Optional dict mapping op names to bias weights (for memory-driven search)
        """
        self.policy_net = policy_net
        self.value_net = value_net
        self.topas_model = topas_model
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_noise_weight = dirichlet_noise_weight
        self.max_depth = max_depth
        self.temperature = temperature

        # Accept str or torch.device, default CUDA when available
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.op_bias = op_bias or {}

        # Ensure policy/value nets are on the same device
        if self.policy_net:
            self.policy_net.to(self.device)
        if self.value_net:
            self.value_net.to(self.device)

        # Available DSL operations
        self.available_ops = CORE_OPS.copy()

        # Statistics
        self.search_stats = {
            'total_simulations': 0,
            'total_nodes_created': 0,
            'cache_hits': 0,
            'neural_evaluations': 0,
            'op_bias_applications': 0
        }

        if self.op_bias:
            logger.info(f"Initialized PUCT searcher with {len(self.available_ops)} DSL operations and {len(self.op_bias)} biased ops")
        else:
            logger.info(f"Initialized PUCT searcher with {len(self.available_ops)} DSL operations")

    def _ensure_device(self, t):
        """
        ✅ GPU-FIRST: Ensure tensor is on correct device

        Prevents CPU↔GPU mismatches from DSL ops that return CPU tensors
        """
        if isinstance(t, torch.Tensor) and t.device != self.device:
            return t.to(self.device)
        return t

    def get_policy_value(
        self,
        node: MCTSNode,
        demos: List[Tuple[torch.Tensor, torch.Tensor]],
        test_input: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Get policy priors and value estimate from neural networks

        Args:
            node: Current MCTS node
            demos: Demonstration input/output pairs
            test_input: Test input grid

        Returns:
            policy_priors: Probability distribution over actions
            value_estimate: Scalar value estimate for the state
        """
        self.search_stats['neural_evaluations'] += 1

        # Ensure policy/value nets are on the same device
        if self.policy_net:
            self.policy_net.to(self.device)
        if self.value_net:
            self.value_net.to(self.device)

        try:
            # Extract proper features from TOPAS
            current_grid, brain, rel_features, size_oracle, theme_priors = self._encode_program_state(
                node, demos, test_input
            )

            # Call OpPolicyNet with proper signature
            with torch.no_grad():
                policy_output = self.policy_net(
                    grid=current_grid.unsqueeze(0) if current_grid.dim() == 2 else current_grid,
                    rel_features=rel_features,
                    size_oracle=size_oracle,
                    theme_priors=theme_priors,
                    program_ops=node.ops if node.ops else [],
                    seq_pos=node.depth
                )

                # Extract op_logits from PolicyPrediction
                if hasattr(policy_output, 'op_logits'):
                    policy_logits = policy_output.op_logits
                else:
                    policy_logits = policy_output  # Fallback if returns tensor directly

                policy_priors = F.softmax(policy_logits, dim=-1).to(self.device)

            # Apply op_bias if provided (memory-driven biasing from RelMem/Wormhole)
            if self.op_bias:
                from models.dsl_registry import get_op_index  # Import here for fallback path
                self.search_stats['op_bias_applications'] += 1
                # Blend op_bias weights into policy priors
                for op_name, bias_weight in self.op_bias.items():
                    op_idx = get_op_index(op_name)
                    if 0 <= op_idx < policy_priors.shape[-1]:
                        # Apply bias: boost probability by (1 + bias_weight)
                        policy_priors[0, op_idx] *= (1.0 + bias_weight)

                # Re-normalize to maintain valid probability distribution
                policy_priors = policy_priors / policy_priors.sum(dim=-1, keepdim=True)

            # Call ValueNet with proper signature
            with torch.no_grad():
                value_output = self.value_net(
                    grid=current_grid.unsqueeze(0) if current_grid.dim() == 2 else current_grid,
                    rel_features=rel_features,
                    size_oracle=size_oracle,
                    theme_priors=theme_priors,
                    program_ops=node.ops if node.ops else [],
                    program_depth=node.depth
                )

                # Extract value from ValuePrediction
                if hasattr(value_output, 'solvability'):
                    solvability = value_output.solvability.squeeze()
                    # Take mean if multiple values (batch dimension or spatial)
                    value_estimate = solvability.mean().item() if solvability.numel() > 1 else solvability.item()
                elif hasattr(value_output, 'value'):
                    value_tensor = value_output.value.squeeze()
                    value_estimate = value_tensor.mean().item() if value_tensor.numel() > 1 else value_tensor.item()
                else:
                    if torch.is_tensor(value_output):
                        value_output = value_output.squeeze()
                        value_estimate = value_output.mean().item() if value_output.numel() > 1 else value_output.item()
                    else:
                        value_estimate = float(value_output)

            logger.info(f"[PUCT] Eval: depth={node.depth}, ops={node.ops[-3:] if len(node.ops) > 3 else node.ops}, value={value_estimate:.3f}")
            return policy_priors, value_estimate

        except Exception as e:
            logger.error(f"[PUCT] Neural network evaluation FAILED: {e}")
            import traceback
            traceback.print_exc()

            # Fallback for inference when policy/value nets are None
            if self.policy_net is None or self.value_net is None:
                logger.warning("[PUCT] Using uniform policy fallback (nets not available)")
                # Uniform policy over available operations
                num_ops = len(self.dsl_ops) if hasattr(self, 'dsl_ops') else 19
                policy_priors = torch.ones(1, num_ops, device=self.device) / num_ops

                # Apply op_bias if available
                if self.op_bias:
                    for op_name, bias_weight in self.op_bias.items():
                        from models.dsl_registry import get_op_index
                        op_idx = get_op_index(op_name)
                        if 0 <= op_idx < num_ops:
                            policy_priors[0, op_idx] *= (1.0 + bias_weight)
                    policy_priors = policy_priors / policy_priors.sum(dim=-1, keepdim=True)

                value_estimate = 0.0  # Neutral value
                return policy_priors, value_estimate
            else:
                # FAIL LOUDLY in training - uniform policy pollutes training signal
                raise RuntimeError(f"PUCT network eval failed (no fallback allowed): {e}") from e

    def _encode_program_state(
        self,
        node: MCTSNode,
        demos: List[Tuple[torch.Tensor, torch.Tensor]],
        test_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract proper features from TOPAS for OpPolicyNet/ValueNet

        Returns:
            (current_grid, brain, rel_features, size_oracle, theme_priors)
        """
        # Apply program to get current grid state
        if len(node.ops) > 0:
            try:
                program = node.get_program()
                current_grid = self._ensure_device(apply_program(test_input, program))
                if current_grid is None or not torch.is_tensor(current_grid):
                    current_grid = test_input
            except:
                current_grid = test_input
        else:
            current_grid = test_input

        # Extract features from TOPAS model
        with torch.no_grad():
            # Ensure grid has batch dimension
            grid_input = current_grid.unsqueeze(0) if current_grid.dim() == 2 else current_grid

            # Use forward_pretraining - counter is reset per task in eval loop
            try:
                forward_out = self.topas_model.forward_pretraining(grid_input)
            except RuntimeError as e:
                if "Infinite loop" in str(e):
                    # Emergency fallback: return zero features
                    B = grid_input.shape[0]
                    forward_out = {
                        'brain': torch.zeros(B, self.topas_model.ctrl_dim, device=grid_input.device),
                        'rel_features': torch.zeros(B, 64, 256, device=grid_input.device)
                    }
                else:
                    raise

            # Extract brain (control features)
            brain = forward_out.get('brain')  # [B, ctrl_dim]
            if brain is None:
                B = grid_input.shape[0]
                brain = torch.zeros(B, self.topas_model.ctrl_dim, device=self.device)
            else:
                B = brain.shape[0]

            # Extract relational features (if available)
            rel_features = forward_out.get('rel_features')
            if rel_features is None:
                rel_features = torch.zeros(B, 64, device=self.device)  # Use actual batch size

            # Compute size oracle
            H, W = current_grid.shape[-2:]
            size_oracle = torch.tensor([[H, W, H, W]], device=self.device).float()
            size_oracle = size_oracle.expand(B, -1)  # Expand to match batch size

            # Extract theme priors from TOPAS prior heads
            # ✅ FIX: Handle dimension mismatch (brain might be 512 but prior_transform expects 768)
            try:
                # Check if brain dimension matches prior_transform input
                if brain.shape[-1] != self.topas_model.ctrl_dim:
                    # Project brain to correct dimension
                    if not hasattr(self, '_brain_projector') or self._brain_projector.in_features != brain.shape[-1]:
                        self._brain_projector = torch.nn.Linear(
                            brain.shape[-1],
                            self.topas_model.ctrl_dim,
                            device=brain.device
                        )
                    brain = self._brain_projector(brain)

                theme_priors = self.topas_model.prior_transform(brain)  # [B, 8]
            except RuntimeError as e:
                # Emergency fallback: return zeros if dimension mismatch
                import logging
                logging.warning(f"[PUCT] prior_transform failed: {e}, using zero priors")
                theme_priors = torch.zeros(brain.size(0), 8, device=self.device)

            # 🔥 FIX: Ensure theme_priors has correct batch dimension before padding
            if theme_priors.dim() == 1:
                theme_priors = theme_priors.unsqueeze(0)  # [D] → [1, D]

            # Match batch dimension if mismatch (e.g., [30, 5] vs expected [1, 5])
            if theme_priors.size(0) != B:
                # Take first B rows if too many, or expand if too few
                if theme_priors.size(0) > B:
                    theme_priors = theme_priors[:B, :]
                else:
                    theme_priors = theme_priors.expand(B, -1)

            if theme_priors.shape[-1] > 10:
                theme_priors = theme_priors[:, :10]
            elif theme_priors.shape[-1] < 10:
                padding = torch.zeros(theme_priors.size(0), 10 - theme_priors.shape[-1], device=self.device)
                theme_priors = torch.cat([theme_priors, padding], dim=-1)

        return current_grid, brain, rel_features, size_oracle, theme_priors

    def add_dirichlet_noise(self, policy_priors: torch.Tensor) -> torch.Tensor:
        """Add Dirichlet noise to policy priors for exploration"""
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(policy_priors))
        noise_tensor = torch.tensor(noise, dtype=policy_priors.dtype, device=policy_priors.device)

        return (1 - self.dirichlet_noise_weight) * policy_priors + \
               self.dirichlet_noise_weight * noise_tensor

    def expand_node(
        self,
        node: MCTSNode,
        demos: List[Tuple[torch.Tensor, torch.Tensor]],
        test_input: torch.Tensor
    ) -> MCTSNode:
        """
        Expand a node by adding all possible children and evaluating with neural networks

        Args:
            node: Node to expand
            demos: Demonstration pairs for evaluation
            test_input: Test input grid

        Returns:
            A newly created child node (for selection in simulation)
        """
        if node.is_terminal or node.depth >= self.max_depth:
            return node

        # Get neural network predictions for this state
        policy_priors, value_estimate = self.get_policy_value(node, demos, test_input)

        # Store in node
        node.policy_priors = policy_priors
        node.value_estimate = value_estimate

        # Add Dirichlet noise at root for exploration
        if node.parent is None:  # Root node
            node.policy_priors = self.add_dirichlet_noise(policy_priors)

        # Create children for all possible actions
        created_children = []

        for op in self.available_ops:
            # Generate parameters for this operation
            param_options = generate_op_parameters(op, None)

            for params in param_options:
                child = node.add_child(op, params)
                created_children.append(child)
                self.search_stats['total_nodes_created'] += 1

        # Return a random new child for simulation continuation
        if created_children:
            return np.random.choice(created_children)
        else:
            return node

    def simulate_once(
        self,
        root: MCTSNode,
        demos: List[Tuple[torch.Tensor, torch.Tensor]],
        test_input: torch.Tensor
    ) -> float:
        """
        Run one simulation from root to leaf and backpropagate value

        Args:
            root: Root node of search tree
            demos: Demonstration input/output pairs
            test_input: Test input grid

        Returns:
            Value that was backpropagated
        """
        self.search_stats['total_simulations'] += 1
        path = []
        node = root

        # Selection phase: traverse tree using PUCT
        while not node.is_terminal and node.is_fully_expanded(self.available_ops, self.max_depth):
            node = node.select_best_child(self.c_puct)
            if node is None:
                break
            path.append(node)

        # Expansion phase: add new children if not terminal
        if not node.is_terminal and node.depth < self.max_depth:
            node = self.expand_node(node, demos, test_input)
            path.append(node)

        # Evaluation phase: get value for leaf node
        if node.is_terminal:
            value = node.terminal_reward
        else:
            # Evaluate current program on demonstrations
            value = self._evaluate_program(node, demos, test_input)

            # Also use neural network value estimate
            if hasattr(node, 'value_estimate'):
                # Combine program evaluation with neural value estimate
                value = 0.7 * value + 0.3 * node.value_estimate

        # Backpropagation phase: update all nodes in path
        for node in reversed(path):
            node.update_value(value)

        return value

    def _evaluate_program(
        self,
        node: MCTSNode,
        demos: List[Tuple[torch.Tensor, torch.Tensor]],
        test_input: torch.Tensor
    ) -> float:
        """
        Evaluate a program represented by a node on demonstration pairs

        Args:
            node: Node representing program state
            demos: Demonstration input/output pairs
            test_input: Test input (unused for evaluation, but kept for consistency)

        Returns:
            Evaluation score between -1 and 1
        """
        if len(node.ops) == 0:
            return 0.0  # Empty program gets neutral score

        try:
            program = node.get_program()
            total_score = 0.0
            valid_demos = 0

            for input_grid, target_output in demos:
                try:
                    # Apply program to input
                    predicted_output = self._ensure_device(apply_program(input_grid, program))

                    # Compute similarity score
                    if predicted_output.shape == target_output.shape:
                        # Pixel-wise accuracy
                        accuracy = (predicted_output == target_output).float().mean().item()
                        total_score += accuracy
                    else:
                        # Penalize shape mismatch but give some partial credit
                        total_score += 0.1  # Small positive score for at least running

                    valid_demos += 1

                except Exception:
                    # Program failed on this demo - give negative score
                    total_score -= 0.5
                    valid_demos += 1

            if valid_demos > 0:
                avg_score = total_score / valid_demos
                # Map to [-1, 1] range with 0 as neutral
                return max(-1.0, min(1.0, avg_score * 2.0 - 1.0))
            else:
                return -1.0  # No valid evaluations

        except Exception as e:
            logger.debug(f"Program evaluation failed: {e}")
            return -1.0

    def search(
        self,
        demos: List[Tuple[torch.Tensor, torch.Tensor]],
        test_input: torch.Tensor,
        num_simulations: int = 800,
        timeout_seconds: float = 30.0
    ) -> Tuple[Optional[DSLProgram], Dict[str, Any]]:
        """
        Run PUCT search to find best program

        Args:
            demos: List of (input, output) demonstration pairs
            test_input: Test input grid to transform
            num_simulations: Number of MCTS simulations to run
            timeout_seconds: Maximum search time

        Returns:
            best_program: Best program found (or None if no solution)
            search_info: Dictionary with search statistics and tree info
        """
        start_time = time.time()

        # Initialize root node
        root = MCTSNode()
        self.search_stats['total_nodes_created'] += 1

        # Check for immediate solution (empty program)
        if self._is_solved(root, demos):
            root.is_solved = True
            root.terminal_reward = 1.0
            return root.get_program(), {"simulations_run": 0, "tree_size": 1}

        # Run simulations
        simulations_run = 0
        for sim in range(num_simulations):
            if time.time() - start_time > timeout_seconds:
                logger.info(f"PUCT search timeout after {simulations_run} simulations")
                break

            # Run one simulation
            value = self.simulate_once(root, demos, test_input)
            simulations_run += 1

            # Check if we found a perfect solution
            if self._check_for_solution(root, demos):
                logger.info(f"PUCT found solution after {simulations_run} simulations")
                break

            # Early stopping if we have a very good candidate
            if simulations_run >= 100 and simulations_run % 50 == 0:
                best_child = self._get_best_child(root, temperature=0.1)
                if best_child and best_child.mean_value > 0.9:
                    logger.info(f"PUCT found high-confidence solution after {simulations_run} simulations")
                    break

        # Extract best program
        best_program = None
        best_child = self._get_best_child(root, temperature=0.1)  # Low temperature for best selection

        if best_child:
            best_program = best_child.get_program()

        # Compile search statistics
        search_info = {
            "simulations_run": simulations_run,
            "tree_size": self.search_stats['total_nodes_created'],
            "max_depth_reached": self._get_max_depth(root),
            "best_value": best_child.mean_value if best_child else 0.0,
            "neural_evaluations": self.search_stats['neural_evaluations'],
            "search_time": time.time() - start_time,
            "nodes_per_second": simulations_run / max(time.time() - start_time, 0.001)
        }

        return best_program, search_info

    def _is_solved(self, node: MCTSNode, demos: List[Tuple[torch.Tensor, torch.Tensor]]) -> bool:
        """Check if a node represents a solution to all demonstrations"""
        if len(node.ops) == 0:
            return False  # Empty program is never a solution

        try:
            program = node.get_program()
            for input_grid, target_output in demos:
                predicted = self._ensure_device(apply_program(input_grid, program))
                if not torch.equal(predicted, target_output):
                    return False
            return True
        except:
            return False

    def _check_for_solution(self, root: MCTSNode, demos: List[Tuple[torch.Tensor, torch.Tensor]]) -> bool:
        """Check if any node in the tree represents a solution"""
        def check_node(node):
            if self._is_solved(node, demos):
                node.is_solved = True
                node.terminal_reward = 1.0
                return True

            for child in node.children.values():
                if check_node(child):
                    return True
            return False

        return check_node(root)

    def _get_best_child(self, node: MCTSNode, temperature: float = 1.0) -> Optional[MCTSNode]:
        """Get best child using visit count and temperature"""
        if not node.children:
            return None

        if temperature == 0.0:
            # Greedy selection - choose child with highest visit count
            return max(node.children.values(), key=lambda c: c.visit_count)
        else:
            # Temperature-based selection
            visit_counts = np.array([child.visit_count for child in node.children.values()])
            if visit_counts.sum() == 0:
                return np.random.choice(list(node.children.values()))

            # Apply temperature
            probs = visit_counts ** (1.0 / temperature)
            probs = probs / probs.sum()

            children = list(node.children.values())
            return np.random.choice(children, p=probs)

    def _get_max_depth(self, root: MCTSNode) -> int:
        """Get maximum depth reached in the search tree"""
        def get_depth(node):
            if not node.children:
                return node.depth
            return max(get_depth(child) for child in node.children.values())

        return get_depth(root)

    def get_action_probs(self, root: MCTSNode, temperature: float = 1.0) -> Dict[Tuple[str, frozenset], float]:
        """
        Get action probabilities for policy training

        Args:
            root: Root node of search tree
            temperature: Temperature for probability computation

        Returns:
            Dictionary mapping actions to probabilities
        """
        if not root.children:
            return {}

        visit_counts = root.get_visit_counts()
        total_visits = sum(visit_counts.values())

        if total_visits == 0:
            # Uniform distribution if no visits
            num_actions = len(visit_counts)
            return {action: 1.0 / num_actions for action in visit_counts.keys()}

        if temperature == 0.0:
            # One-hot on most visited action
            best_action = max(visit_counts.keys(), key=lambda a: visit_counts[a])
            return {action: 1.0 if action == best_action else 0.0 for action in visit_counts.keys()}
        else:
            # Temperature-based probabilities
            probs = {}
            for action, count in visit_counts.items():
                probs[action] = (count / total_visits) ** (1.0 / temperature)

            # Normalize
            total_prob = sum(probs.values())
            if total_prob > 0:
                for action in probs:
                    probs[action] /= total_prob

            return probs

    def reset_stats(self):
        """Reset search statistics"""
        self.search_stats = {
            'total_simulations': 0,
            'total_nodes_created': 0,
            'cache_hits': 0,
            'neural_evaluations': 0
        }


def puct_search(
    demos: List[Tuple[torch.Tensor, torch.Tensor]],
    test_input: torch.Tensor = None,  # Made optional for alpha_evolve compatibility
    policy_net: torch.nn.Module = None,  # Made optional for alpha_evolve compatibility
    value_net: torch.nn.Module = None,  # Made optional for alpha_evolve compatibility
    topas_model: torch.nn.Module = None,  # Required for PUCTSearcher
    num_simulations: int = 800,
    max_nodes: int = None,  # BitterBot enhancement: support legacy max_nodes parameter
    c_puct: float = 1.4,
    max_depth: int = 10,
    timeout_seconds: float = 30.0,
    device: torch.device = None,
    op_bias: Optional[Dict[str, float]] = None,
    beam: int = 24,  # Alpha-Evolve: beam width for search
    # Alpha-Evolve compatibility aliases
    test: torch.Tensor = None,  # Alias for test_input
    dsl_ops: List[str] = None,  # DSL operations list
    sims: int = None,  # Alias for num_simulations
    depth: int = None,  # Alias for max_depth
    op_priors: Optional[Dict[str, float]] = None,  # Alias for op_bias
    relmem_bias: Optional[Dict[str, float]] = None,  # Additional bias source
    return_info: bool = False  # Return detailed search info
) -> Tuple[Optional[str], float]:
    """
    Convenience function to run PUCT search with default parameters

    Args:
        demos: List of (input, output) demonstration pairs
        test_input: Test input grid
        policy_net: Neural policy network
        value_net: Neural value network
        num_simulations: Number of MCTS simulations
        beam: Beam width for search
        c_puct: Exploration constant
        max_depth: Maximum program depth
        timeout_seconds: Search timeout
        device: Compute device
        op_bias: Optional dict mapping op names to bias weights (from memory/wormhole)

    Returns:
        best_op: Best operation found (string)
        best_value: Value estimate (float)
        OR (if return_info=True):
        program: Best DSL program
        info: Dict with search statistics
    """
    # === ALPHA-EVOLVE COMPATIBILITY: Handle parameter aliases ===
    if test is not None:
        test_input = test
    if sims is not None:
        num_simulations = sims
    if depth is not None:
        max_depth = depth
    if op_priors is not None:
        op_bias = op_priors

    # Merge relmem_bias into op_bias if provided
    if relmem_bias is not None:
        if op_bias is None:
            op_bias = relmem_bias
        else:
            # Blend both bias sources
            op_bias = {**op_bias, **relmem_bias}

    # BitterBot enhancement: Support both max_nodes and num_simulations parameters
    if max_nodes is not None:
        num_simulations = max_nodes

    # Create searcher (beam parameter stored but not used in current implementation)
    searcher = PUCTSearcher(
        policy_net=policy_net,
        value_net=value_net,
        topas_model=topas_model,  # Required parameter
        c_puct=c_puct,
        max_depth=max_depth,
        device=device,
        op_bias=op_bias
    )
    # Store beam for potential future use in node expansion
    searcher.beam_width = beam

    best_program, search_info = searcher.search(demos, test_input, num_simulations, timeout_seconds)

    # === ALPHA-EVOLVE: Return program + info if requested ===
    if return_info:
        return best_program, search_info

    # === STANDARD RETURN: Extract best operation and value ===
    if best_program and hasattr(best_program, 'operations') and best_program.operations:
        best_op = str(best_program.operations[0])
        best_value = float(search_info.get('best_value', 0.0))
    else:
        best_op = None
        best_value = 0.0

    # Ensure consistent return type
    try:
        # If best_op is a dict, stringify it
        if isinstance(best_op, dict):
            from trainers.utils import _stringify_ops
            safe_ops = _stringify_ops([best_op])
            best_op = safe_ops[0] if safe_ops else str(best_op)
        # If best_value is not numeric, pick a representative score
        if isinstance(best_value, dict):
            best_value = float(best_value.get("value", 0.0))
        else:
            best_value = float(best_value)
        return best_op, best_value
    except Exception:
        return best_op, 0.0


# Legacy interface compatibility for existing code
def puct_program_search(model,
                       demos: List[Tuple[torch.Tensor, torch.Tensor]],
                       test_grid: torch.Tensor,
                       target_grid: Optional[torch.Tensor] = None,
                       max_length: int = 8,
                       max_nodes: int = 500,
                       c_puct: float = 1.5) -> List[str]:
    """
    Legacy interface for backward compatibility
    Full program search using PUCT for multi-step DSL programs.

    Args:
        model: TOPAS model with HRM bridge
        demos: List of (input, output) demonstration pairs
        test_grid: Input grid to transform
        target_grid: Target output grid (if known)
        max_length: Maximum program length
        max_nodes: PUCT search budget per step
        c_puct: Exploration constant

    Returns:
        program: List of operation names forming the discovered program
    """
    # This is a simplified version that maintains compatibility
    # In practice, you'd extract policy/value nets from the model

    try:
        # Extract or create simple policy/value networks
        if hasattr(model, 'policy_net') and hasattr(model, 'value_net'):
            program, _ = puct_search(
                demos=demos,
                test_input=test_grid,
                policy_net=model.policy_net,
                value_net=model.value_net,
                num_simulations=max_nodes,
                c_puct=c_puct,
                max_depth=max_length
            )
            return program.ops if program else []
        else:
            # Fallback to simple beam search-like approach
            logger.warning("Model doesn't have policy/value nets, using simplified search")
            return []

    except Exception as e:
        logger.error(f"PUCT program search failed: {e}")
        return []


# Example usage and testing
if __name__ == "__main__":
    # This would be run for testing the implementation
    import torch.nn as nn

    # Mock neural networks for testing
    class MockPolicyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(NUM_DSL_OPS + 16, NUM_DSL_OPS)

        def forward(self, x):
            return self.fc(x)

    class MockValueNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(NUM_DSL_OPS + 16, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

        def forward(self, x):
            return self.fc(x)

    # Create mock demonstration
    demo_input = torch.zeros(5, 5)
    demo_output = torch.ones(5, 5)
    demos = [(demo_input, demo_output)]

    test_input = torch.zeros(5, 5)

    # Run PUCT search
    policy_net = MockPolicyNet()
    value_net = MockValueNet()

    program, stats = puct_search(
        demos=demos,
        test_input=test_input,
        policy_net=policy_net,
        value_net=value_net,
        num_simulations=50,
        timeout_seconds=10.0
    )

    print(f"Found program: {program}")
    print(f"Search stats: {stats}")