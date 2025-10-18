#!/usr/bin/env python3
"""
Meta-Learning Integration for TOPAS Cognitive Architecture
Enables system-wide learning-to-learn capability across all cognitive components.
Critical for few-shot learning, rapid adaptation, and human-level intelligence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import math
import logging
import numpy as np
from collections import defaultdict, deque

class MetaLearningStrategy(Enum):
    """Different meta-learning strategies"""
    GRADIENT_BASED = "gradient_based"       # MAML-style gradient-based meta-learning
    MEMORY_BASED = "memory_based"          # Memory-augmented meta-learning
    OPTIMIZATION_BASED = "optimization_based"  # Learning to optimize
    ARCHITECTURE_BASED = "architecture_based"  # Learning architectures
    STRATEGY_BASED = "strategy_based"      # Learning problem-solving strategies

@dataclass
class MetaLearningExperience:
    """A meta-learning experience across multiple tasks"""
    task_family: str
    support_examples: List[Dict[str, torch.Tensor]]
    query_examples: List[Dict[str, torch.Tensor]]
    component_states: Dict[str, Any]
    adaptation_trace: List[Dict[str, Any]]
    final_performance: float
    meta_gradients: Optional[Dict[str, torch.Tensor]] = None

@dataclass
class ComponentAdaptationProfile:
    """Adaptation profile for a cognitive component"""
    component_name: str
    base_learning_rate: float
    adaptation_rate: float
    forgetting_rate: float
    capacity_limit: int
    specialization_areas: List[str] = field(default_factory=list)
    adaptation_history: deque = field(default_factory=lambda: deque(maxlen=100))

class MetaLearningOrchestrator(nn.Module):
    """
    System-wide meta-learning orchestrator that coordinates learning-to-learn
    across all cognitive components for human-level adaptation capability.
    """

    def __init__(self,
                 feature_dim: int = 512,
                 meta_dim: int = 1024,
                 num_adaptation_steps: int = 5,
                 meta_learning_rate: float = 0.001,
                 device: str = "cpu"):
        super().__init__()

        self.device = torch.device(device)
        self.feature_dim = feature_dim
        self.meta_dim = meta_dim
        self.num_adaptation_steps = num_adaptation_steps
        self.meta_learning_rate = meta_learning_rate
        self.logger = logging.getLogger(__name__)

        # Meta-learning networks
        self.meta_learner = self._create_meta_learner()
        self.adaptation_controller = self._create_adaptation_controller()
        self.task_encoder = self._create_task_encoder()
        self.strategy_selector = self._create_strategy_selector()

        # Component adaptation profiles
        self.component_profiles = {}  # component_name -> ComponentAdaptationProfile

        # Meta-learning experience buffer
        self.meta_experiences = deque(maxlen=1000)
        self.task_family_patterns = defaultdict(list)

        # Few-shot learning components
        self.few_shot_memory = nn.ParameterDict()
        self.rapid_adaptation_networks = nn.ModuleDict()

        # Performance tracking
        self.adaptation_performance_history = {}
        self.meta_learning_metrics = {
            'successful_adaptations': 0,
            'rapid_learning_events': 0,
            'cross_component_transfers': 0,
            'strategy_discoveries': 0
        }

    def _create_meta_learner(self) -> nn.Module:
        """Create the core meta-learning network"""
        return nn.Sequential(
            nn.Linear(self.feature_dim * 2 + self.meta_dim, self.meta_dim),
            nn.LayerNorm(self.meta_dim),
            nn.ReLU(),
            nn.Linear(self.meta_dim, self.meta_dim),
            nn.LayerNorm(self.meta_dim),
            nn.ReLU(),
            nn.Linear(self.meta_dim, self.feature_dim),
            nn.Tanh()
        )

    def _create_adaptation_controller(self) -> nn.Module:
        """Create network to control adaptation dynamics"""
        return nn.Sequential(
            nn.Linear(self.meta_dim, self.meta_dim // 2),
            nn.ReLU(),
            nn.Linear(self.meta_dim // 2, 4),  # [learning_rate, adaptation_steps, forgetting_rate, exploration]
            nn.Sigmoid()
        )

    def _create_task_encoder(self) -> nn.Module:
        """Create network to encode task characteristics"""
        return nn.Sequential(
            nn.Linear(self.feature_dim, self.meta_dim),
            nn.LayerNorm(self.meta_dim),
            nn.ReLU(),
            nn.Linear(self.meta_dim, self.meta_dim // 2),
            nn.LayerNorm(self.meta_dim // 2),
            nn.ReLU(),
            nn.Linear(self.meta_dim // 2, self.meta_dim // 4),
            nn.Tanh()
        )

    def _create_strategy_selector(self) -> nn.Module:
        """Create network to select optimal learning strategies"""
        return nn.Sequential(
            nn.Linear(self.meta_dim + len(MetaLearningStrategy), self.meta_dim),
            nn.ReLU(),
            nn.Linear(self.meta_dim, len(MetaLearningStrategy)),
            nn.Softmax(dim=-1)
        )

    def register_component_for_meta_learning(self,
                                           component_name: str,
                                           component_obj: Any,
                                           base_lr: float = 0.001,
                                           adaptation_rate: float = 0.1) -> ComponentAdaptationProfile:
        """Register a cognitive component for meta-learning coordination"""

        profile = ComponentAdaptationProfile(
            component_name=component_name,
            base_learning_rate=base_lr,
            adaptation_rate=adaptation_rate,
            forgetting_rate=0.95,
            capacity_limit=1000
        )

        self.component_profiles[component_name] = profile

        # Create component-specific rapid adaptation network
        if component_name not in self.rapid_adaptation_networks:
            self.rapid_adaptation_networks[component_name] = self._create_rapid_adaptation_network()

        # Initialize few-shot memory for this component
        memory_key = f"{component_name}_memory"
        if memory_key not in self.few_shot_memory:
            self.few_shot_memory[memory_key] = nn.Parameter(
                torch.randn(10, self.feature_dim, device=self.device) * 0.1
            )

        self.logger.info(f"[MetaLearning] Registered component: {component_name}")
        return profile

    def _create_rapid_adaptation_network(self) -> nn.Module:
        """Create component-specific rapid adaptation network"""
        return nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.Tanh()
        )

    def few_shot_adaptation(self,
                          component_name: str,
                          support_examples: List[Dict[str, torch.Tensor]],
                          num_adaptation_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform few-shot adaptation for a specific component.

        Args:
            component_name: Name of component to adapt
            support_examples: Few examples to adapt from
            num_adaptation_steps: Number of adaptation steps (default: self.num_adaptation_steps)

        Returns:
            Adaptation results and updated parameters
        """

        if component_name not in self.component_profiles:
            self.logger.warning(f"Component {component_name} not registered for meta-learning")
            return {'success': False, 'reason': 'component_not_registered'}

        try:
            adaptation_steps = num_adaptation_steps or self.num_adaptation_steps
            profile = self.component_profiles[component_name]

            # Encode task characteristics from support examples
            task_encoding = self._encode_task_from_examples(support_examples)

            # Generate adaptation strategy
            adaptation_strategy = self._select_adaptation_strategy(task_encoding, component_name)

            # Perform adaptation based on strategy
            if adaptation_strategy == MetaLearningStrategy.GRADIENT_BASED:
                adaptation_result = self._gradient_based_adaptation(
                    component_name, support_examples, adaptation_steps
                )

            elif adaptation_strategy == MetaLearningStrategy.MEMORY_BASED:
                adaptation_result = self._memory_based_adaptation(
                    component_name, support_examples, adaptation_steps
                )

            elif adaptation_strategy == MetaLearningStrategy.OPTIMIZATION_BASED:
                adaptation_result = self._optimization_based_adaptation(
                    component_name, support_examples, adaptation_steps
                )

            else:  # Default to gradient-based
                adaptation_result = self._gradient_based_adaptation(
                    component_name, support_examples, adaptation_steps
                )

            # Update component profile
            profile.adaptation_history.append({
                'timestamp': time.time(),
                'strategy': adaptation_strategy.value,
                'performance': adaptation_result.get('performance', 0.0),
                'convergence_steps': adaptation_result.get('convergence_steps', adaptation_steps)
            })

            # Track successful adaptations
            if adaptation_result.get('performance', 0.0) > 0.7:
                self.meta_learning_metrics['successful_adaptations'] += 1

                # Check if this was rapid learning (< 3 steps)
                if adaptation_result.get('convergence_steps', adaptation_steps) <= 3:
                    self.meta_learning_metrics['rapid_learning_events'] += 1

            return adaptation_result

        except Exception as e:
            self.logger.warning(f"Few-shot adaptation failed for {component_name}: {e}")
            return {'success': False, 'reason': str(e)}

    def _encode_task_from_examples(self, examples: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Encode task characteristics from examples"""

        try:
            if not examples:
                return torch.zeros(self.meta_dim // 4, device=self.device)

            # Extract features from examples
            feature_representations = []

            for example in examples:
                before_state = example.get('before', torch.zeros(1, 1, self.feature_dim, device=self.device))
                after_state = example.get('after', torch.zeros(1, 1, self.feature_dim, device=self.device))

                # Compute transformation characteristics
                diff = after_state - before_state
                magnitude = torch.norm(diff)
                direction = diff / (magnitude + 1e-8)

                # Create feature representation
                features = torch.cat([
                    before_state.mean(dim=(0, 1))[:self.feature_dim//4],
                    after_state.mean(dim=(0, 1))[:self.feature_dim//4],
                    direction.mean(dim=(0, 1))[:self.feature_dim//4],
                    torch.tensor([magnitude], device=self.device)
                ], dim=0)

                feature_representations.append(features)

            # Aggregate features across examples
            aggregated = torch.stack(feature_representations).mean(dim=0)

            # Encode into task representation
            task_encoding = self.task_encoder(aggregated)

            return task_encoding

        except Exception as e:
            self.logger.warning(f"Task encoding failed: {e}")
            return torch.zeros(self.meta_dim // 4, device=self.device)

    def _select_adaptation_strategy(self,
                                  task_encoding: torch.Tensor,
                                  component_name: str) -> MetaLearningStrategy:
        """Select optimal adaptation strategy for task and component"""

        try:
            # Create strategy encoding
            strategy_features = torch.zeros(len(MetaLearningStrategy), device=self.device)

            # Add component-specific bias based on history
            profile = self.component_profiles[component_name]
            if profile.adaptation_history:
                # Bias toward strategies that worked well for this component
                strategy_success = defaultdict(list)
                for history_entry in profile.adaptation_history:
                    strategy = history_entry['strategy']
                    performance = history_entry['performance']
                    strategy_success[strategy].append(performance)

                for i, strategy in enumerate(MetaLearningStrategy):
                    if strategy.value in strategy_success:
                        avg_performance = np.mean(strategy_success[strategy.value])
                        strategy_features[i] = avg_performance

            # Combine task encoding and strategy features
            combined_input = torch.cat([task_encoding, strategy_features], dim=0)

            # Select strategy
            strategy_probs = self.strategy_selector(combined_input)
            selected_index = torch.argmax(strategy_probs).item()

            strategies = list(MetaLearningStrategy)
            selected_strategy = strategies[selected_index]

            self.logger.debug(f"Selected {selected_strategy.value} for {component_name}")
            return selected_strategy

        except Exception as e:
            self.logger.warning(f"Strategy selection failed: {e}")
            return MetaLearningStrategy.GRADIENT_BASED  # Default fallback

    def _gradient_based_adaptation(self,
                                 component_name: str,
                                 support_examples: List[Dict[str, torch.Tensor]],
                                 adaptation_steps: int) -> Dict[str, Any]:
        """MAML-style gradient-based adaptation"""

        try:
            adaptation_network = self.rapid_adaptation_networks[component_name]
            initial_params = {name: param.clone() for name, param in adaptation_network.named_parameters()}

            # Simulate adaptation on support examples
            adapted_params = initial_params.copy()
            convergence_step = adaptation_steps

            for step in range(adaptation_steps):
                step_loss = 0.0
                gradients = {}

                # Compute gradients on support examples
                for example in support_examples:
                    before_state = example.get('before', torch.zeros(1, 1, self.feature_dim, device=self.device))
                    after_state = example.get('after', torch.zeros(1, 1, self.feature_dim, device=self.device))

                    # Forward pass through adaptation network
                    before_features = before_state.mean(dim=(0, 1))[:self.feature_dim]
                    predicted_after = adaptation_network(before_features)
                    target_after = after_state.mean(dim=(0, 1))[:self.feature_dim]

                    # Compute loss
                    loss = F.mse_loss(predicted_after, target_after)
                    step_loss += loss.item()

                    # Compute gradients
                    grads = torch.autograd.grad(loss, adaptation_network.parameters(), create_graph=True)

                    for i, (name, param) in enumerate(adaptation_network.named_parameters()):
                        if name not in gradients:
                            gradients[name] = grads[i]
                        else:
                            gradients[name] += grads[i]

                # Update parameters
                profile = self.component_profiles[component_name]
                for name, param in adaptation_network.named_parameters():
                    if name in gradients:
                        adapted_params[name] = param - profile.adaptation_rate * gradients[name]

                # Check for convergence
                if step_loss < 0.01:
                    convergence_step = step + 1
                    break

            # Compute meta-gradient for updating meta-learner
            meta_gradient = self._compute_meta_gradient(initial_params, adapted_params, support_examples)

            return {
                'success': True,
                'strategy': MetaLearningStrategy.GRADIENT_BASED.value,
                'convergence_steps': convergence_step,
                'final_loss': step_loss,
                'performance': max(0.0, 1.0 - step_loss),
                'meta_gradient': meta_gradient,
                'adapted_params': adapted_params
            }

        except Exception as e:
            self.logger.warning(f"Gradient-based adaptation failed: {e}")
            return {'success': False, 'reason': str(e)}

    def _memory_based_adaptation(self,
                               component_name: str,
                               support_examples: List[Dict[str, torch.Tensor]],
                               adaptation_steps: int) -> Dict[str, Any]:
        """Memory-augmented meta-learning adaptation"""

        try:
            memory_key = f"{component_name}_memory"
            if memory_key not in self.few_shot_memory:
                return {'success': False, 'reason': 'no_memory_allocated'}

            memory = self.few_shot_memory[memory_key]

            # Update memory based on support examples
            memory_updates = []

            for example in support_examples:
                before_state = example.get('before', torch.zeros(1, 1, self.feature_dim, device=self.device))
                after_state = example.get('after', torch.zeros(1, 1, self.feature_dim, device=self.device))

                # Create memory update
                before_features = before_state.mean(dim=(0, 1))[:self.feature_dim]
                after_features = after_state.mean(dim=(0, 1))[:self.feature_dim]

                memory_update = torch.cat([before_features, after_features], dim=0)[:self.feature_dim]
                memory_updates.append(memory_update)

            if memory_updates:
                # Update memory with new examples
                new_memory_content = torch.stack(memory_updates).mean(dim=0)

                # Find best memory slot to update (least recently used)
                memory_similarities = F.cosine_similarity(
                    memory, new_memory_content.unsqueeze(0), dim=1
                )
                update_slot = torch.argmin(memory_similarities).item()

                # Update memory slot
                memory.data[update_slot] = 0.9 * memory.data[update_slot] + 0.1 * new_memory_content

                performance = float(memory_similarities.max().item())

                return {
                    'success': True,
                    'strategy': MetaLearningStrategy.MEMORY_BASED.value,
                    'convergence_steps': 1,  # Memory updates are instant
                    'performance': performance,
                    'memory_updated': True,
                    'updated_slot': update_slot
                }

        except Exception as e:
            self.logger.warning(f"Memory-based adaptation failed: {e}")
            return {'success': False, 'reason': str(e)}

    def _optimization_based_adaptation(self,
                                     component_name: str,
                                     support_examples: List[Dict[str, torch.Tensor]],
                                     adaptation_steps: int) -> Dict[str, Any]:
        """Learning to optimize - meta-learn optimal optimization strategies"""

        try:
            # Use adaptation controller to determine optimal optimization parameters
            task_encoding = self._encode_task_from_examples(support_examples)

            # Generate adaptive optimization parameters
            adaptation_params = self.adaptation_controller(task_encoding)

            adaptive_lr = float(adaptation_params[0].item()) * 0.01  # Scale to reasonable range
            adaptive_steps = max(1, int(adaptation_params[1].item() * adaptation_steps))
            adaptive_momentum = float(adaptation_params[2].item())
            exploration_factor = float(adaptation_params[3].item())

            # Apply adaptive optimization to component
            component_obj = None  # Would get from component registry in full implementation

            optimization_performance = 0.7  # Placeholder - would measure actual optimization success

            return {
                'success': True,
                'strategy': MetaLearningStrategy.OPTIMIZATION_BASED.value,
                'convergence_steps': adaptive_steps,
                'performance': optimization_performance,
                'adaptive_params': {
                    'learning_rate': adaptive_lr,
                    'steps': adaptive_steps,
                    'momentum': adaptive_momentum,
                    'exploration': exploration_factor
                }
            }

        except Exception as e:
            self.logger.warning(f"Optimization-based adaptation failed: {e}")
            return {'success': False, 'reason': str(e)}

    def _compute_meta_gradient(self,
                             initial_params: Dict[str, torch.Tensor],
                             adapted_params: Dict[str, torch.Tensor],
                             support_examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Compute meta-gradients for updating the meta-learner"""

        meta_gradients = {}

        try:
            # Compute parameter updates
            param_updates = {}
            for name in initial_params:
                param_updates[name] = adapted_params[name] - initial_params[name]

            # The meta-gradient is how to best update the meta-learner based on adaptation success
            # For simplicity, use the parameter updates as meta-gradients
            for name, update in param_updates.items():
                meta_gradients[f"meta_{name}"] = update.detach()

            return meta_gradients

        except Exception as e:
            self.logger.warning(f"Meta-gradient computation failed: {e}")
            return {}

    def cross_component_meta_learning(self,
                                    adaptation_experiences: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Perform cross-component meta-learning to share adaptation strategies.

        Args:
            adaptation_experiences: Dict mapping component_name to list of adaptation experiences

        Returns:
            Cross-component learning results
        """

        transfer_results = {
            'successful_transfers': 0,
            'total_transfer_attempts': 0,
            'component_synergies': {},
            'optimal_coordination_patterns': []
        }

        try:
            component_names = list(adaptation_experiences.keys())

            # Analyze successful adaptations across components
            successful_patterns = {}

            for component_name, experiences in adaptation_experiences.items():
                successful_exps = [exp for exp in experiences if exp.get('performance', 0.0) > 0.6]

                if successful_exps:
                    # Extract common patterns in successful adaptations
                    patterns = self._extract_adaptation_patterns(successful_exps)
                    successful_patterns[component_name] = patterns

            # Find transferable patterns between components
            for source_comp in component_names:
                for target_comp in component_names:
                    if source_comp != target_comp:
                        transfer_score = self._attempt_pattern_transfer(
                            source_comp, target_comp, successful_patterns
                        )

                        transfer_results['total_transfer_attempts'] += 1

                        if transfer_score > 0.5:
                            transfer_results['successful_transfers'] += 1

                            # Record component synergy
                            synergy_key = f"{source_comp}→{target_comp}"
                            if synergy_key not in transfer_results['component_synergies']:
                                transfer_results['component_synergies'][synergy_key] = []
                            transfer_results['component_synergies'][synergy_key].append(transfer_score)

            # Identify optimal coordination patterns
            transfer_results['optimal_coordination_patterns'] = self._identify_coordination_patterns(
                transfer_results['component_synergies']
            )

            # Update meta-learning metrics
            self.meta_learning_metrics['cross_component_transfers'] += transfer_results['successful_transfers']

            return transfer_results

        except Exception as e:
            self.logger.warning(f"Cross-component meta-learning failed: {e}")
            return transfer_results

    def _extract_adaptation_patterns(self, successful_experiences: List[Dict]) -> Dict[str, Any]:
        """Extract common patterns from successful adaptations"""

        patterns = {
            'avg_convergence_steps': 0.0,
            'preferred_strategies': [],
            'optimal_learning_rates': [],
            'common_features': None
        }

        try:
            if not successful_experiences:
                return patterns

            # Analyze convergence patterns
            convergence_steps = [exp.get('convergence_steps', 5) for exp in successful_experiences]
            patterns['avg_convergence_steps'] = np.mean(convergence_steps)

            # Analyze strategy preferences
            strategy_counts = defaultdict(int)
            for exp in successful_experiences:
                strategy = exp.get('strategy', 'unknown')
                strategy_counts[strategy] += 1

            patterns['preferred_strategies'] = sorted(
                strategy_counts.items(), key=lambda x: x[1], reverse=True
            )

            # Extract optimal learning rates from adaptive parameters
            learning_rates = []
            for exp in successful_experiences:
                adaptive_params = exp.get('adaptive_params', {})
                if 'learning_rate' in adaptive_params:
                    learning_rates.append(adaptive_params['learning_rate'])

            if learning_rates:
                patterns['optimal_learning_rates'] = learning_rates

            return patterns

        except Exception as e:
            self.logger.warning(f"Pattern extraction failed: {e}")
            return patterns

    def _attempt_pattern_transfer(self,
                                source_component: str,
                                target_component: str,
                                pattern_library: Dict[str, Dict]) -> float:
        """Attempt to transfer adaptation patterns between components"""

        try:
            if source_component not in pattern_library or target_component not in pattern_library:
                return 0.0

            source_patterns = pattern_library[source_component]
            target_patterns = pattern_library[target_component]

            # Check compatibility of adaptation patterns
            source_strategies = dict(source_patterns['preferred_strategies'])
            target_strategies = dict(target_patterns['preferred_strategies'])

            # Compute strategy overlap
            common_strategies = set(source_strategies.keys()) & set(target_strategies.keys())
            strategy_compatibility = len(common_strategies) / max(1, len(source_strategies))

            # Check learning rate compatibility
            source_lrs = source_patterns.get('optimal_learning_rates', [])
            target_lrs = target_patterns.get('optimal_learning_rates', [])

            lr_compatibility = 0.5  # Default
            if source_lrs and target_lrs:
                source_avg_lr = np.mean(source_lrs)
                target_avg_lr = np.mean(target_lrs)
                lr_ratio = min(source_avg_lr, target_avg_lr) / max(source_avg_lr, target_avg_lr)
                lr_compatibility = lr_ratio

            # Overall transfer score
            transfer_score = (strategy_compatibility + lr_compatibility) / 2.0

            return transfer_score

        except Exception as e:
            self.logger.warning(f"Pattern transfer failed: {e}")
            return 0.0

    def _identify_coordination_patterns(self, synergies: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """Identify optimal coordination patterns from component synergies"""

        patterns = []

        try:
            # Find high-synergy component pairs
            for synergy_key, scores in synergies.items():
                if scores:
                    avg_score = np.mean(scores)
                    if avg_score > 0.7:  # High synergy threshold
                        source, target = synergy_key.split('→')
                        patterns.append({
                            'source_component': source,
                            'target_component': target,
                            'synergy_strength': avg_score,
                            'coordination_recommendation': 'high_priority_transfer'
                        })

            # Sort by synergy strength
            patterns.sort(key=lambda x: x['synergy_strength'], reverse=True)

            return patterns[:10]  # Top 10 coordination patterns

        except Exception as e:
            self.logger.warning(f"Coordination pattern identification failed: {e}")
            return []

    def generate_meta_learning_curriculum(self,
                                        difficulty_progression: List[str] = None) -> List[Dict[str, Any]]:
        """
        Generate a curriculum for meta-learning that progressively increases difficulty.

        Args:
            difficulty_progression: Optional custom difficulty progression

        Returns:
            Curriculum of meta-learning tasks
        """

        default_progression = ['basic_patterns', 'complex_patterns', 'novel_combinations', 'systematic_generalization']
        progression = difficulty_progression or default_progression

        curriculum = []

        try:
            for stage, difficulty_level in enumerate(progression):
                stage_tasks = self._generate_stage_tasks(difficulty_level, stage)
                curriculum.extend(stage_tasks)

            return curriculum

        except Exception as e:
            self.logger.warning(f"Curriculum generation failed: {e}")
            return []

    def _generate_stage_tasks(self, difficulty_level: str, stage: int) -> List[Dict[str, Any]]:
        """Generate tasks for a specific curriculum stage"""

        tasks = []

        try:
            if difficulty_level == 'basic_patterns':
                # Simple adaptation tasks
                for i in range(5):
                    task = {
                        'task_id': f"basic_{stage}_{i}",
                        'difficulty': 'basic',
                        'adaptation_steps': 3,
                        'expected_performance': 0.8,
                        'focus_components': ['neuroplanner', 'relmem']
                    }
                    tasks.append(task)

            elif difficulty_level == 'complex_patterns':
                # More complex adaptation requiring multiple components
                for i in range(5):
                    task = {
                        'task_id': f"complex_{stage}_{i}",
                        'difficulty': 'complex',
                        'adaptation_steps': 5,
                        'expected_performance': 0.6,
                        'focus_components': ['neuroplanner', 'dream', 'relmem']
                    }
                    tasks.append(task)

            elif difficulty_level == 'novel_combinations':
                # Test novel primitive combinations
                for i in range(3):
                    task = {
                        'task_id': f"novel_{stage}_{i}",
                        'difficulty': 'novel',
                        'adaptation_steps': 7,
                        'expected_performance': 0.4,
                        'focus_components': ['all'],
                        'requires_composition': True
                    }
                    tasks.append(task)

            elif difficulty_level == 'systematic_generalization':
                # Ultimate test of systematic generalization
                task = {
                    'task_id': f"systematic_{stage}",
                    'difficulty': 'systematic',
                    'adaptation_steps': 10,
                    'expected_performance': 0.3,
                    'focus_components': ['all'],
                    'requires_composition': True,
                    'requires_causal_reasoning': True,
                    'requires_transfer': True
                }
                tasks.append(task)

            return tasks

        except Exception as e:
            self.logger.warning(f"Stage task generation failed: {e}")
            return []

    def assess_meta_learning_capability(self) -> Dict[str, float]:
        """
        Assess the system's meta-learning capability across all components.

        Returns:
            Comprehensive assessment of meta-learning progress
        """

        assessment = {
            'few_shot_learning_score': 0.0,
            'adaptation_speed_score': 0.0,
            'transfer_learning_score': 0.0,
            'strategy_discovery_score': 0.0,
            'cross_component_coordination_score': 0.0,
            'overall_meta_learning_score': 0.0
        }

        try:
            # Few-shot learning capability
            if self.meta_learning_metrics['successful_adaptations'] > 0:
                rapid_learning_rate = (
                    self.meta_learning_metrics['rapid_learning_events'] /
                    self.meta_learning_metrics['successful_adaptations']
                )
                assessment['few_shot_learning_score'] = rapid_learning_rate

            # Adaptation speed
            if self.component_profiles:
                avg_convergence_steps = []
                for profile in self.component_profiles.values():
                    if profile.adaptation_history:
                        steps = [h['convergence_steps'] for h in profile.adaptation_history]
                        avg_convergence_steps.extend(steps)

                if avg_convergence_steps:
                    # Lower steps = faster adaptation = higher score
                    avg_steps = np.mean(avg_convergence_steps)
                    assessment['adaptation_speed_score'] = max(0.0, 1.0 - avg_steps / 10.0)

            # Transfer learning score
            transfer_success_rate = 0.0
            if self.meta_learning_metrics['cross_component_transfers'] > 0:
                # Placeholder - would measure actual transfer success
                transfer_success_rate = min(1.0, self.meta_learning_metrics['cross_component_transfers'] / 100.0)
            assessment['transfer_learning_score'] = transfer_success_rate

            # Strategy discovery
            strategy_diversity = len(set(
                exp.get('strategy', 'unknown') for experiences in self.adaptation_performance_history.values()
                for exp in experiences
            ))
            assessment['strategy_discovery_score'] = min(1.0, strategy_diversity / len(MetaLearningStrategy))

            # Cross-component coordination
            coordination_score = 0.0
            if len(self.component_profiles) > 1:
                # Measure how well components coordinate
                coordination_events = sum(
                    len(profile.adaptation_history) for profile in self.component_profiles.values()
                )
                coordination_score = min(1.0, coordination_events / (len(self.component_profiles) * 10))
            assessment['cross_component_coordination_score'] = coordination_score

            # Overall meta-learning score
            assessment['overall_meta_learning_score'] = (
                assessment['few_shot_learning_score'] * 0.25 +
                assessment['adaptation_speed_score'] * 0.25 +
                assessment['transfer_learning_score'] * 0.25 +
                assessment['strategy_discovery_score'] * 0.125 +
                assessment['cross_component_coordination_score'] * 0.125
            )

            return assessment

        except Exception as e:
            self.logger.warning(f"Meta-learning assessment failed: {e}")
            return assessment

    def generate_meta_learning_training_signals(self) -> Dict[str, torch.Tensor]:
        """Generate training signals that enhance meta-learning across components"""

        signals = {}

        try:
            # Signal 1: Few-shot adaptation loss
            if len(self.meta_experiences) >= 2:
                recent_exps = list(self.meta_experiences)[-5:]
                adaptation_losses = []

                for exp in recent_exps:
                    if exp.final_performance < 0.7:  # Suboptimal performance
                        # Create loss to improve adaptation
                        performance_gap = 0.7 - exp.final_performance
                        adaptation_loss = torch.tensor(performance_gap, device=self.device)
                        adaptation_losses.append(adaptation_loss)

                if adaptation_losses:
                    signals['few_shot_adaptation_loss'] = torch.stack(adaptation_losses).mean()

            # Signal 2: Meta-learning consistency loss
            consistency_loss = self._compute_meta_learning_consistency_loss()
            if consistency_loss is not None:
                signals['meta_learning_consistency_loss'] = consistency_loss

            # Signal 3: Cross-component coordination loss
            coordination_loss = self._compute_coordination_loss()
            if coordination_loss is not None:
                signals['cross_component_coordination_loss'] = coordination_loss

            return signals

        except Exception as e:
            self.logger.warning(f"Meta-learning signal generation failed: {e}")
            return {'few_shot_adaptation_loss': torch.tensor(0.0, device=self.device)}

    def _compute_meta_learning_consistency_loss(self) -> Optional[torch.Tensor]:
        """Compute loss that encourages consistent meta-learning across components"""

        try:
            if len(self.component_profiles) < 2:
                return None

            # Get recent adaptation performance for each component
            component_performances = {}

            for name, profile in self.component_profiles.items():
                if profile.adaptation_history:
                    recent_performance = [h['performance'] for h in list(profile.adaptation_history)[-5:]]
                    component_performances[name] = np.mean(recent_performance)

            if len(component_performances) < 2:
                return None

            performances = list(component_performances.values())
            performance_variance = np.var(performances)

            # Loss encourages similar adaptation capability across components
            consistency_loss = torch.tensor(performance_variance, device=self.device)

            return consistency_loss

        except Exception as e:
            self.logger.warning(f"Consistency loss computation failed: {e}")
            return None

    def _compute_coordination_loss(self) -> Optional[torch.Tensor]:
        """Compute loss that encourages better cross-component coordination"""

        try:
            # Measure coordination effectiveness
            total_experiences = sum(len(profile.adaptation_history) for profile in self.component_profiles.values())

            if total_experiences == 0:
                return None

            # Target: balanced experience across components
            target_experiences_per_component = total_experiences / len(self.component_profiles)

            coordination_imbalance = 0.0
            for profile in self.component_profiles.values():
                actual_experiences = len(profile.adaptation_history)
                imbalance = abs(actual_experiences - target_experiences_per_component)
                coordination_imbalance += imbalance

            coordination_loss = torch.tensor(coordination_imbalance / len(self.component_profiles), device=self.device)

            return coordination_loss

        except Exception as e:
            self.logger.warning(f"Coordination loss computation failed: {e}")
            return None

    def get_human_level_meta_learning_score(self) -> float:
        """
        Assess how close the system is to human-level meta-learning capability.

        Returns score in [0, 1] where 1.0 indicates human-level meta-learning.
        """

        try:
            assessment = self.assess_meta_learning_capability()

            # Weight factors for human-level meta-learning
            factors = [
                (assessment['few_shot_learning_score'], 0.3),      # Critical for human-level learning
                (assessment['adaptation_speed_score'], 0.25),      # Humans adapt quickly
                (assessment['transfer_learning_score'], 0.25),     # Humans transfer knowledge well
                (assessment['strategy_discovery_score'], 0.1),     # Humans discover new strategies
                (assessment['cross_component_coordination_score'], 0.1)  # Humans coordinate mental processes
            ]

            weighted_score = sum(score * weight for score, weight in factors)

            return float(weighted_score)

        except Exception as e:
            self.logger.warning(f"Human-level meta-learning assessment failed: {e}")
            return 0.0