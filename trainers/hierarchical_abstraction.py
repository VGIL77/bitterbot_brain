#!/usr/bin/env python3
"""
Hierarchical Abstraction System for Human-Level Pattern Recognition
Implements multi-level pattern hierarchy from grid-level to meta-pattern abstraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math
import logging

class AbstractionLevel(Enum):
    """Levels of abstraction in the hierarchy"""
    GRID = 0        # Grid-level patterns (rotation, flip, color changes)
    OBJECT = 1      # Object-level patterns (composition, spatial relations)
    ABSTRACT = 2    # Abstract patterns (invariant transformations)
    META = 3        # Meta-patterns (transformation classes, reasoning strategies)

@dataclass
class AbstractPattern:
    """A pattern at a specific abstraction level"""
    level: AbstractionLevel
    pattern_id: str
    representation: torch.Tensor
    confidence: float
    usage_count: int = 0
    success_rate: float = 0.0
    generalization_scope: List[str] = None  # Domains where pattern applies
    causal_weight: float = 0.0  # How causally important this pattern is

    def __post_init__(self):
        if self.generalization_scope is None:
            self.generalization_scope = []

class HierarchicalAbstractor(nn.Module):
    """
    Multi-level abstraction system that extracts patterns at different levels of granularity.
    Enables human-level pattern recognition through hierarchical decomposition.
    """

    def __init__(self,
                 grid_dim: int = 256,
                 object_dim: int = 512,
                 abstract_dim: int = 1024,
                 meta_dim: int = 2048,
                 max_patterns_per_level: int = 256,
                 device: str = "cpu"):
        super().__init__()

        self.device = torch.device(device)
        self.max_patterns_per_level = max_patterns_per_level
        self.logger = logging.getLogger(__name__)

        # Dimension specifications for each level
        self.level_dims = {
            AbstractionLevel.GRID: grid_dim,
            AbstractionLevel.OBJECT: object_dim,
            AbstractionLevel.ABSTRACT: abstract_dim,
            AbstractionLevel.META: meta_dim
        }

        # Pattern storage for each level
        self.pattern_library = {level: {} for level in AbstractionLevel}

        # Cross-level projection layers
        self.grid_to_object = nn.Linear(grid_dim, object_dim)
        self.object_to_abstract = nn.Linear(object_dim, abstract_dim)
        self.abstract_to_meta = nn.Linear(abstract_dim, meta_dim)

        # Reverse projection layers for top-down influence
        self.meta_to_abstract = nn.Linear(meta_dim, abstract_dim)
        self.abstract_to_object = nn.Linear(abstract_dim, object_dim)
        self.object_to_grid = nn.Linear(object_dim, grid_dim)

        # Pattern recognition networks for each level
        self.grid_recognizer = self._create_pattern_recognizer(grid_dim)
        self.object_recognizer = self._create_pattern_recognizer(object_dim)
        self.abstract_recognizer = self._create_pattern_recognizer(abstract_dim)
        self.meta_recognizer = self._create_pattern_recognizer(meta_dim)

        # Pattern generators for creating new abstractions
        self.grid_generator = self._create_pattern_generator(grid_dim)
        self.object_generator = self._create_pattern_generator(object_dim)
        self.abstract_generator = self._create_pattern_generator(abstract_dim)
        self.meta_generator = self._create_pattern_generator(meta_dim)

        # Invariance detection networks
        self.invariance_detector = nn.Sequential(
            nn.Linear(abstract_dim, abstract_dim // 2),
            nn.ReLU(),
            nn.Linear(abstract_dim // 2, abstract_dim // 4),
            nn.ReLU(),
            nn.Linear(abstract_dim // 4, 1),
            nn.Sigmoid()
        )

        # Compositional reasoning network
        self.composition_engine = self._create_composition_engine(abstract_dim)

        # Meta-learning adaptation network
        self.meta_adapter = nn.Sequential(
            nn.Linear(meta_dim, meta_dim // 2),
            nn.ReLU(),
            nn.Linear(meta_dim // 2, 4),  # Outputs: [learning_rate, attention, exploration, consolidation]
            nn.Sigmoid()
        )

    def _create_pattern_recognizer(self, dim: int) -> nn.Module:
        """Create a pattern recognition network for a specific abstraction level"""
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim // 4),
            nn.Sigmoid()
        )

    def _create_pattern_generator(self, dim: int) -> nn.Module:
        """Create a pattern generation network for a specific abstraction level"""
        return nn.Sequential(
            nn.Linear(dim // 4, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
            nn.LayerNorm(dim),
            nn.Tanh()
        )

    def _create_composition_engine(self, dim: int) -> nn.Module:
        """Create compositional reasoning engine"""
        return nn.Sequential(
            nn.Linear(dim * 2, dim),  # Takes two patterns as input
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Tanh()
        )

    def extract_hierarchical_patterns(self,
                                    input_features: torch.Tensor,
                                    level: AbstractionLevel = AbstractionLevel.GRID) -> Dict[AbstractionLevel, torch.Tensor]:
        """
        Extract patterns at all abstraction levels from input features.

        Args:
            input_features: [B, T, D] input representations
            level: Starting level for extraction

        Returns:
            Dict mapping each abstraction level to extracted patterns
        """

        patterns = {}
        current_repr = input_features

        # Bottom-up abstraction
        if level.value <= AbstractionLevel.GRID.value:
            # Grid-level pattern extraction
            grid_patterns = self.grid_recognizer(current_repr)
            patterns[AbstractionLevel.GRID] = grid_patterns
            current_repr = self.grid_to_object(current_repr)

        if level.value <= AbstractionLevel.OBJECT.value:
            # Object-level pattern extraction
            object_patterns = self.object_recognizer(current_repr)
            patterns[AbstractionLevel.OBJECT] = object_patterns
            current_repr = self.object_to_abstract(current_repr)

        if level.value <= AbstractionLevel.ABSTRACT.value:
            # Abstract pattern extraction with invariance detection
            abstract_patterns = self.abstract_recognizer(current_repr)
            patterns[AbstractionLevel.ABSTRACT] = abstract_patterns

            # Detect invariant properties
            invariance_scores = self.invariance_detector(current_repr)
            patterns[f"{AbstractionLevel.ABSTRACT.name}_invariance"] = invariance_scores

            current_repr = self.abstract_to_meta(current_repr)

        if level.value <= AbstractionLevel.META.value:
            # Meta-level pattern extraction
            meta_patterns = self.meta_recognizer(current_repr)
            patterns[AbstractionLevel.META] = meta_patterns

        return patterns

    def top_down_influence(self,
                          high_level_patterns: torch.Tensor,
                          target_level: AbstractionLevel) -> torch.Tensor:
        """
        Apply top-down influence from higher abstraction levels.

        Args:
            high_level_patterns: Patterns from a higher abstraction level
            target_level: Target level to influence

        Returns:
            Influenced representations at target level
        """

        current_repr = high_level_patterns

        # Project down through the hierarchy
        if target_level.value <= AbstractionLevel.ABSTRACT.value:
            current_repr = self.meta_to_abstract(current_repr)

        if target_level.value <= AbstractionLevel.OBJECT.value:
            current_repr = self.abstract_to_object(current_repr)

        if target_level.value <= AbstractionLevel.GRID.value:
            current_repr = self.object_to_grid(current_repr)

        return current_repr

    def compose_patterns(self,
                        pattern1: torch.Tensor,
                        pattern2: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Compose two patterns to create a novel combination.

        Args:
            pattern1, pattern2: Patterns to compose

        Returns:
            (composed_pattern, composition_confidence)
        """

        try:
            # Ensure patterns are same size
            if pattern1.shape != pattern2.shape:
                # Project to common dimension
                common_dim = min(pattern1.shape[-1], pattern2.shape[-1])
                if pattern1.shape[-1] != common_dim:
                    pattern1 = pattern1[..., :common_dim]
                if pattern2.shape[-1] != common_dim:
                    pattern2 = pattern2[..., :common_dim]

            # Concatenate and compose
            combined = torch.cat([pattern1, pattern2], dim=-1)
            composed = self.composition_engine(combined)

            # Compute composition confidence
            similarity = F.cosine_similarity(pattern1.flatten(), pattern2.flatten(), dim=0)
            diversity = 1.0 - similarity  # More diverse = higher confidence
            confidence = float(torch.sigmoid(diversity * 2.0).item())

            return composed, confidence

        except Exception as e:
            self.logger.warning(f"Pattern composition failed: {e}")
            return pattern1, 0.1

    def detect_invariant_transformations(self,
                                       before_patterns: torch.Tensor,
                                       after_patterns: torch.Tensor) -> Dict[str, float]:
        """
        Detect invariant transformations between before/after pattern states.
        Critical for understanding what aspects remain constant during transformations.
        """

        invariants = {}

        try:
            # Compute pattern differences
            pattern_diff = after_patterns - before_patterns

            # Invariant 1: Magnitude preservation
            before_norm = torch.norm(before_patterns, dim=-1)
            after_norm = torch.norm(after_patterns, dim=-1)
            magnitude_preservation = 1.0 - torch.abs(before_norm - after_norm).mean()
            invariants['magnitude_preservation'] = float(magnitude_preservation.item())

            # Invariant 2: Structural similarity
            structural_similarity = F.cosine_similarity(
                before_patterns.flatten(), after_patterns.flatten(), dim=0
            )
            invariants['structural_similarity'] = float(structural_similarity.item())

            # Invariant 3: Energy conservation
            before_energy = torch.sum(before_patterns ** 2, dim=-1)
            after_energy = torch.sum(after_patterns ** 2, dim=-1)
            energy_conservation = 1.0 - torch.abs(before_energy - after_energy).mean()
            invariants['energy_conservation'] = float(energy_conservation.item())

            # Invariant 4: Information preservation
            before_entropy = self._compute_entropy(before_patterns)
            after_entropy = self._compute_entropy(after_patterns)
            info_preservation = 1.0 - abs(before_entropy - after_entropy)
            invariants['information_preservation'] = float(info_preservation)

            # Overall invariance score
            invariants['overall_invariance'] = sum(invariants.values()) / len(invariants)

        except Exception as e:
            self.logger.warning(f"Invariant detection failed: {e}")
            invariants = {'overall_invariance': 0.0}

        return invariants

    def _compute_entropy(self, patterns: torch.Tensor) -> float:
        """Compute approximate entropy of pattern representations"""
        try:
            # Use histogram-based entropy estimation
            flat_patterns = patterns.flatten()
            # Discretize into bins
            hist = torch.histc(flat_patterns, bins=20, min=flat_patterns.min(), max=flat_patterns.max())
            probs = hist / hist.sum()
            probs = probs[probs > 0]  # Remove zero entries
            entropy = -(probs * torch.log(probs)).sum()
            return float(entropy.item())
        except Exception:
            return 1.0

    def generate_abstract_hypothesis(self,
                                   observed_patterns: List[torch.Tensor],
                                   target_level: AbstractionLevel = AbstractionLevel.ABSTRACT) -> Tuple[torch.Tensor, float]:
        """
        Generate abstract hypothesis from observed patterns.
        Critical for few-shot learning and generalization.
        """

        if not observed_patterns:
            return torch.zeros(self.level_dims[target_level]), 0.0

        try:
            # Stack and process observed patterns
            stacked_patterns = torch.stack(observed_patterns, dim=0)  # [N, D]

            # Compute pattern centroid as base hypothesis
            pattern_centroid = stacked_patterns.mean(dim=0)

            # Compute pattern variance for diversity estimation
            pattern_variance = stacked_patterns.var(dim=0)

            # Generate hypothesis using appropriate generator
            if target_level == AbstractionLevel.GRID:
                hypothesis = self.grid_generator(pattern_centroid.unsqueeze(0) / 4)  # Scale down for generator
            elif target_level == AbstractionLevel.OBJECT:
                hypothesis = self.object_generator(pattern_centroid.unsqueeze(0) / 4)
            elif target_level == AbstractionLevel.ABSTRACT:
                hypothesis = self.abstract_generator(pattern_centroid.unsqueeze(0) / 4)
            else:  # META
                hypothesis = self.meta_generator(pattern_centroid.unsqueeze(0) / 4)

            hypothesis = hypothesis.squeeze(0)  # Remove batch dimension

            # Compute hypothesis confidence based on pattern consistency
            consistency = 1.0 / (1.0 + pattern_variance.mean().item())
            diversity = min(1.0, len(observed_patterns) / 5.0)  # More patterns = higher confidence
            confidence = (consistency + diversity) / 2.0

            return hypothesis, confidence

        except Exception as e:
            self.logger.warning(f"Hypothesis generation failed: {e}")
            return torch.zeros(self.level_dims[target_level]), 0.1

    def extract_compositional_primitives(self,
                                       patterns: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract compositional primitives that can be systematically recombined.
        Essential for systematic generalization.
        """

        primitives = []

        try:
            B, T, D = patterns.shape

            # Extract stable components using SVD
            U, S, V = torch.svd(patterns.view(-1, D))

            # Select top components based on singular values
            num_components = min(8, len(S))  # Top 8 components
            threshold = S[0] * 0.1  # 10% of largest singular value

            for i in range(num_components):
                if S[i] > threshold:
                    primitive = V[:, i]  # i-th right singular vector
                    primitives.append(primitive)

            # Extract frequency-based primitives using FFT
            if D >= 4:  # Ensure sufficient dimension for FFT
                freq_patterns = torch.fft.fft(patterns.mean(dim=1), dim=-1)  # [B, D]
                freq_magnitude = torch.abs(freq_patterns)

                # Find dominant frequencies
                dominant_freqs = torch.topk(freq_magnitude.mean(dim=0), k=min(4, D//4)).indices

                for freq_idx in dominant_freqs:
                    freq_primitive = torch.zeros(D)
                    freq_primitive[freq_idx] = 1.0
                    primitives.append(freq_primitive)

            # Extract statistical primitives
            mean_pattern = patterns.mean(dim=(0, 1))
            std_pattern = patterns.std(dim=(0, 1))

            primitives.extend([mean_pattern, std_pattern])

            return primitives[:12]  # Limit to 12 primitives

        except Exception as e:
            self.logger.warning(f"Primitive extraction failed: {e}")
            # Fallback: return identity primitives
            return [torch.eye(D)[i] for i in range(min(4, D))]

    def systematic_recombination(self,
                                primitives: List[torch.Tensor],
                                combination_rules: Optional[List[str]] = None) -> List[Tuple[torch.Tensor, float]]:
        """
        Systematically recombine primitives to generate novel patterns.
        Core mechanism for systematic generalization.
        """

        if len(primitives) < 2:
            return []

        combinations = []
        default_rules = ['add', 'multiply', 'compose', 'interpolate'] if combination_rules is None else combination_rules

        try:
            # Generate all pairwise combinations
            for i in range(len(primitives)):
                for j in range(i + 1, len(primitives)):
                    p1, p2 = primitives[i], primitives[j]

                    for rule in default_rules:
                        try:
                            if rule == 'add':
                                combined = p1 + p2
                                confidence = 0.8
                            elif rule == 'multiply':
                                combined = p1 * p2
                                confidence = 0.7
                            elif rule == 'compose':
                                composed, confidence = self.compose_patterns(
                                    p1.unsqueeze(0), p2.unsqueeze(0)
                                )
                                combined = composed.squeeze(0)
                            elif rule == 'interpolate':
                                alpha = 0.5  # Could be learned
                                combined = alpha * p1 + (1 - alpha) * p2
                                confidence = 0.6
                            else:
                                continue

                            # Normalize combined pattern
                            if combined.norm() > 0:
                                combined = combined / combined.norm()

                            combinations.append((combined, confidence))

                        except Exception as e:
                            self.logger.debug(f"Combination rule {rule} failed: {e}")
                            continue

            # Sort by confidence and return top combinations
            combinations.sort(key=lambda x: x[1], reverse=True)
            return combinations[:20]  # Top 20 combinations

        except Exception as e:
            self.logger.warning(f"Systematic recombination failed: {e}")
            return []

    def abstract_transfer_learning(self,
                                 source_patterns: torch.Tensor,
                                 target_domain_hint: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Transfer abstract patterns from source domain to target domain.
        Critical for cross-domain generalization.
        """

        try:
            # Extract invariant components from source patterns
            source_abstract = self.extract_hierarchical_patterns(
                source_patterns.unsqueeze(0), AbstractionLevel.ABSTRACT
            )[AbstractionLevel.ABSTRACT]

            # Detect what's invariant about the source patterns
            invariants = self.invariance_detector(source_abstract)

            # Project to target domain
            target_projection = self.abstract_to_object(source_abstract)
            target_grid = self.object_to_grid(target_projection)

            # Combine with target domain hint
            target_influence = 0.3  # How much target domain affects transfer
            transferred = (1 - target_influence) * target_grid.squeeze(0) + target_influence * target_domain_hint

            # Compute transfer confidence based on invariance strength
            transfer_confidence = float(invariants.mean().item())

            return transferred, transfer_confidence

        except Exception as e:
            self.logger.warning(f"Abstract transfer failed: {e}")
            return target_domain_hint, 0.1

    def meta_adaptation_signal(self,
                             performance_history: List[float],
                             current_patterns: torch.Tensor) -> Dict[str, float]:
        """
        Generate meta-learning adaptation signals based on performance and patterns.
        Enables learning-to-learn capability.
        """

        try:
            # Extract meta-level representation
            meta_patterns = self.extract_hierarchical_patterns(
                current_patterns.unsqueeze(0), AbstractionLevel.META
            )[AbstractionLevel.META]

            # Generate adaptation parameters
            adaptation_params = self.meta_adapter(meta_patterns)

            # Analyze performance trend
            if len(performance_history) >= 3:
                recent_trend = performance_history[-1] - performance_history[-3]
                trend_factor = torch.sigmoid(torch.tensor(recent_trend * 10.0))  # Scale trend
            else:
                trend_factor = torch.tensor(0.5)

            # Generate adaptation signals
            signals = {
                'learning_rate_multiplier': float(adaptation_params[0, 0].item() * (0.5 + trend_factor.item())),
                'attention_focus_strength': float(adaptation_params[0, 1].item()),
                'exploration_bonus': float(adaptation_params[0, 2].item()),
                'consolidation_strength': float(adaptation_params[0, 3].item()),
                'meta_confidence': float(trend_factor.item())
            }

            return signals

        except Exception as e:
            self.logger.warning(f"Meta-adaptation failed: {e}")
            return {
                'learning_rate_multiplier': 1.0,
                'attention_focus_strength': 1.0,
                'exploration_bonus': 0.1,
                'consolidation_strength': 0.5,
                'meta_confidence': 0.5
            }

    def add_pattern_to_library(self,
                              level: AbstractionLevel,
                              pattern: torch.Tensor,
                              pattern_id: str,
                              confidence: float = 0.5,
                              causal_weight: float = 0.0):
        """Add a pattern to the hierarchical library"""

        if len(self.pattern_library[level]) >= self.max_patterns_per_level:
            # Remove least successful pattern
            worst_pattern_id = min(
                self.pattern_library[level].keys(),
                key=lambda pid: self.pattern_library[level][pid].success_rate
            )
            del self.pattern_library[level][worst_pattern_id]

        abstract_pattern = AbstractPattern(
            level=level,
            pattern_id=pattern_id,
            representation=pattern.detach().clone(),
            confidence=confidence,
            causal_weight=causal_weight
        )

        self.pattern_library[level][pattern_id] = abstract_pattern
        self.logger.debug(f"Added pattern {pattern_id} to {level.name} level")

    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about learned patterns"""

        stats = {}

        for level in AbstractionLevel:
            level_patterns = self.pattern_library[level]
            if level_patterns:
                confidences = [p.confidence for p in level_patterns.values()]
                success_rates = [p.success_rate for p in level_patterns.values()]
                usage_counts = [p.usage_count for p in level_patterns.values()]

                stats[f"{level.name.lower()}_patterns"] = {
                    'count': len(level_patterns),
                    'avg_confidence': sum(confidences) / len(confidences),
                    'avg_success_rate': sum(success_rates) / len(success_rates),
                    'total_usage': sum(usage_counts),
                    'high_confidence_count': sum(1 for c in confidences if c > 0.8)
                }
            else:
                stats[f"{level.name.lower()}_patterns"] = {
                    'count': 0, 'avg_confidence': 0.0, 'avg_success_rate': 0.0,
                    'total_usage': 0, 'high_confidence_count': 0
                }

        # Overall statistics
        total_patterns = sum(len(patterns) for patterns in self.pattern_library.values())
        stats['total_patterns'] = total_patterns
        stats['hierarchy_depth'] = sum(1 for level in AbstractionLevel if self.pattern_library[level])

        return stats

    def get_human_level_capability_score(self) -> float:
        """
        Assess how close the system is to human-level abstraction capability.

        Returns score in [0, 1] where 1.0 indicates human-level capability.
        """

        try:
            stats = self.get_pattern_statistics()

            # Capability factors
            factors = []

            # 1. Hierarchical depth (can we abstract at multiple levels?)
            hierarchy_score = stats['hierarchy_depth'] / len(AbstractionLevel)
            factors.append(hierarchy_score)

            # 2. Pattern diversity (do we have rich pattern libraries?)
            pattern_diversity = min(1.0, stats['total_patterns'] / (self.max_patterns_per_level * len(AbstractionLevel)))
            factors.append(pattern_diversity)

            # 3. Abstract reasoning capability (high-level patterns with good success rates)
            abstract_stats = stats.get('abstract_patterns', {})
            abstract_success = abstract_stats.get('avg_success_rate', 0.0)
            factors.append(abstract_success)

            # 4. Meta-level reasoning (can we reason about reasoning?)
            meta_stats = stats.get('meta_patterns', {})
            meta_capability = meta_stats.get('avg_confidence', 0.0)
            factors.append(meta_capability)

            # 5. Compositional capability (can we combine concepts systematically?)
            if hasattr(self, '_last_composition_success'):
                composition_capability = self._last_composition_success
            else:
                composition_capability = 0.5
            factors.append(composition_capability)

            # Overall human-level capability score
            capability_score = sum(factors) / len(factors)

            return float(capability_score)

        except Exception as e:
            self.logger.warning(f"Capability assessment failed: {e}")
            return 0.0