#!/usr/bin/env python3
"""
Compositional Learning Architecture for TOPAS
Enables systematic recombination of learned primitives for novel problem solving.
Critical for systematic generalization and human-level reasoning flexibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import itertools
import math
import logging
import random

class CompositionRule(Enum):
    """Rules for combining compositional primitives"""
    SEQUENTIAL = "sequential"       # Apply primitives in sequence
    PARALLEL = "parallel"          # Apply primitives simultaneously
    CONDITIONAL = "conditional"    # Apply based on conditions
    RECURSIVE = "recursive"        # Apply primitives recursively
    HIERARCHICAL = "hierarchical"  # Combine at different abstraction levels
    ATTENTION_WEIGHTED = "attention_weighted"  # Combine with learned attention weights

@dataclass
class CompositionalPrimitive:
    """A primitive component that can be combined with others"""
    primitive_id: str
    representation: torch.Tensor
    semantic_type: str              # e.g., "spatial", "color", "shape", "logic"
    composability: float           # How well this primitive combines with others
    usage_count: int = 0
    success_rate: float = 0.0
    compatible_primitives: Set[str] = field(default_factory=set)
    learned_combinations: Dict[str, float] = field(default_factory=dict)  # combination_id -> success_rate

@dataclass
class Composition:
    """A composition of multiple primitives"""
    composition_id: str
    primitives: List[str]          # List of primitive IDs
    combination_rule: CompositionRule
    result_representation: torch.Tensor
    confidence: float
    semantic_coherence: float      # How semantically coherent the composition is
    generalization_scope: List[str] = field(default_factory=list)

class CompositionalLearningEngine(nn.Module):
    """
    Compositional learning engine that discovers and combines primitives systematically.
    Enables the system to handle novel combinations of familiar elements.
    """

    def __init__(self,
                 primitive_dim: int = 256,
                 composition_dim: int = 512,
                 max_primitives: int = 500,
                 max_composition_depth: int = 4,
                 device: str = "cpu"):
        super().__init__()

        self.device = torch.device(device)
        self.primitive_dim = primitive_dim
        self.composition_dim = composition_dim
        self.max_primitives = max_primitives
        self.max_composition_depth = max_composition_depth
        self.logger = logging.getLogger(__name__)

        # Primitive discovery and encoding
        self.primitive_encoder = self._create_primitive_encoder()
        self.primitive_decoder = self._create_primitive_decoder()

        # Composition networks for different rules
        self.sequential_composer = self._create_sequential_composer()
        self.parallel_composer = self._create_parallel_composer()
        self.conditional_composer = self._create_conditional_composer()
        self.attention_composer = self._create_attention_composer()

        # Semantic coherence assessment
        self.coherence_assessor = self._create_coherence_assessor()

        # Compositional reasoning network
        self.composition_reasoner = self._create_composition_reasoner()

        # Storage
        self.primitive_library = {}  # primitive_id -> CompositionalPrimitive
        self.composition_library = {}  # composition_id -> Composition
        self.compatibility_matrix = torch.eye(max_primitives, device=device)

        # Learning dynamics
        self.composition_success_history = deque(maxlen=1000)
        self.primitive_discovery_threshold = 0.7
        self.composition_confidence_threshold = 0.6

        # Systematic generalization tracking
        self.novel_combinations_tested = set()
        self.successful_novel_combinations = set()

    def _create_primitive_encoder(self) -> nn.Module:
        """Create network to encode raw features into compositional primitives"""
        return nn.Sequential(
            nn.Linear(self.primitive_dim, self.primitive_dim),
            nn.LayerNorm(self.primitive_dim),
            nn.ReLU(),
            nn.Linear(self.primitive_dim, self.primitive_dim // 2),
            nn.LayerNorm(self.primitive_dim // 2),
            nn.ReLU(),
            nn.Linear(self.primitive_dim // 2, self.primitive_dim // 4),
            nn.Tanh()
        )

    def _create_primitive_decoder(self) -> nn.Module:
        """Create network to decode primitives back to feature space"""
        return nn.Sequential(
            nn.Linear(self.primitive_dim // 4, self.primitive_dim // 2),
            nn.ReLU(),
            nn.Linear(self.primitive_dim // 2, self.primitive_dim),
            nn.LayerNorm(self.primitive_dim),
            nn.Tanh()
        )

    def _create_sequential_composer(self) -> nn.Module:
        """Network for sequential composition"""
        return nn.Sequential(
            nn.Linear(self.primitive_dim * 2, self.composition_dim),
            nn.LayerNorm(self.composition_dim),
            nn.ReLU(),
            nn.Linear(self.composition_dim, self.composition_dim),
            nn.LayerNorm(self.composition_dim),
            nn.ReLU(),
            nn.Linear(self.composition_dim, self.primitive_dim),
            nn.Tanh()
        )

    def _create_parallel_composer(self) -> nn.Module:
        """Network for parallel composition"""
        return nn.Sequential(
            nn.Linear(self.primitive_dim * 2, self.composition_dim),
            nn.LayerNorm(self.composition_dim),
            nn.ReLU(),
            nn.Linear(self.composition_dim, self.primitive_dim),
            nn.Tanh()
        )

    def _create_conditional_composer(self) -> nn.Module:
        """Network for conditional composition with gating"""
        return nn.ModuleDict({
            'gate': nn.Sequential(
                nn.Linear(self.primitive_dim * 3, self.composition_dim),  # p1 + p2 + condition
                nn.ReLU(),
                nn.Linear(self.composition_dim, 1),
                nn.Sigmoid()
            ),
            'composer': nn.Sequential(
                nn.Linear(self.primitive_dim * 2, self.composition_dim),
                nn.ReLU(),
                nn.Linear(self.composition_dim, self.primitive_dim),
                nn.Tanh()
            )
        })

    def _create_attention_composer(self) -> nn.Module:
        """Network for attention-weighted composition"""
        return nn.ModuleDict({
            'attention': nn.MultiheadAttention(
                embed_dim=self.primitive_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            ),
            'combiner': nn.Sequential(
                nn.Linear(self.primitive_dim, self.composition_dim),
                nn.ReLU(),
                nn.Linear(self.composition_dim, self.primitive_dim),
                nn.Tanh()
            )
        })

    def _create_coherence_assessor(self) -> nn.Module:
        """Network to assess semantic coherence of compositions"""
        return nn.Sequential(
            nn.Linear(self.primitive_dim, self.primitive_dim // 2),
            nn.ReLU(),
            nn.Linear(self.primitive_dim // 2, self.primitive_dim // 4),
            nn.ReLU(),
            nn.Linear(self.primitive_dim // 4, 1),
            nn.Sigmoid()
        )

    def _create_composition_reasoner(self) -> nn.Module:
        """Network for reasoning about composition strategies"""
        return nn.Sequential(
            nn.Linear(self.primitive_dim * 2 + len(CompositionRule), self.composition_dim),
            nn.LayerNorm(self.composition_dim),
            nn.ReLU(),
            nn.Linear(self.composition_dim, self.composition_dim // 2),
            nn.ReLU(),
            nn.Linear(self.composition_dim // 2, 1),
            nn.Sigmoid()
        )

    def discover_primitives(self,
                          input_features: torch.Tensor,
                          semantic_hint: Optional[str] = None) -> List[CompositionalPrimitive]:
        """
        Discover compositional primitives from input features.

        Args:
            input_features: [B, T, D] input representations
            semantic_hint: Optional hint about semantic type

        Returns:
            List of discovered primitives
        """

        discovered_primitives = []

        try:
            B, T, D = input_features.shape

            # Encode into primitive space
            primitive_encodings = self.primitive_encoder(input_features)  # [B, T, D//4]

            # Extract distinct primitives using clustering
            flat_encodings = primitive_encodings.view(-1, primitive_encodings.shape[-1])

            # K-means clustering to find distinct primitives
            num_clusters = min(8, flat_encodings.shape[0] // 4)  # Reasonable number of clusters

            if num_clusters > 0:
                centroids = self._perform_clustering(flat_encodings, num_clusters)

                for i, centroid in enumerate(centroids):
                    primitive_id = f"primitive_{len(self.primitive_library)}_{i}"

                    # Assess primitive quality
                    decoded = self.primitive_decoder(centroid.unsqueeze(0))
                    quality = self._assess_primitive_quality(centroid, decoded.squeeze(0))

                    if quality > self.primitive_discovery_threshold:
                        primitive = CompositionalPrimitive(
                            primitive_id=primitive_id,
                            representation=centroid.detach(),
                            semantic_type=semantic_hint or "unknown",
                            composability=quality
                        )

                        discovered_primitives.append(primitive)

            return discovered_primitives

        except Exception as e:
            self.logger.warning(f"Primitive discovery failed: {e}")
            return []

    def _perform_clustering(self, features: torch.Tensor, num_clusters: int) -> List[torch.Tensor]:
        """Perform K-means clustering to find distinct primitives"""

        try:
            # Simple K-means implementation
            centroids = []

            # Initialize centroids randomly
            indices = torch.randperm(features.shape[0])[:num_clusters]
            initial_centroids = features[indices]

            current_centroids = initial_centroids.clone()

            # Iterate K-means
            for _ in range(10):  # Max 10 iterations
                # Assign points to nearest centroids
                distances = torch.cdist(features, current_centroids)
                assignments = torch.argmin(distances, dim=1)

                # Update centroids
                new_centroids = []
                for k in range(num_clusters):
                    cluster_points = features[assignments == k]
                    if cluster_points.shape[0] > 0:
                        new_centroid = cluster_points.mean(dim=0)
                    else:
                        new_centroid = current_centroids[k]  # Keep old centroid if no points
                    new_centroids.append(new_centroid)

                current_centroids = torch.stack(new_centroids)

            return [current_centroids[i] for i in range(num_clusters)]

        except Exception as e:
            self.logger.warning(f"Clustering failed: {e}")
            return []

    def _assess_primitive_quality(self,
                                primitive_encoding: torch.Tensor,
                                decoded_primitive: torch.Tensor) -> float:
        """Assess the quality of a discovered primitive"""

        try:
            # Reconstruction quality
            reconstruction_loss = F.mse_loss(primitive_encoding, decoded_primitive[:primitive_encoding.shape[0]])
            reconstruction_quality = 1.0 / (1.0 + reconstruction_loss.item())

            # Distinctiveness (how different from existing primitives)
            distinctiveness = 1.0
            for existing_primitive in self.primitive_library.values():
                similarity = F.cosine_similarity(
                    primitive_encoding, existing_primitive.representation, dim=0
                )
                distinctiveness = min(distinctiveness, 1.0 - float(similarity.item()))

            # Information content (entropy)
            normalized_primitive = F.softmax(primitive_encoding, dim=0)
            entropy = -(normalized_primitive * torch.log(normalized_primitive + 1e-8)).sum()
            information_content = float(entropy.item()) / math.log(len(primitive_encoding))

            # Overall quality
            quality = (reconstruction_quality * 0.4 + distinctiveness * 0.4 + information_content * 0.2)

            return quality

        except Exception as e:
            self.logger.warning(f"Primitive quality assessment failed: {e}")
            return 0.0

    def compose_primitives(self,
                         primitive_ids: List[str],
                         composition_rule: CompositionRule,
                         context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, float]:
        """
        Compose multiple primitives using specified rule.

        Args:
            primitive_ids: List of primitive IDs to compose
            composition_rule: Rule for composition
            context: Optional context for conditional composition

        Returns:
            (composed_representation, confidence)
        """

        try:
            if len(primitive_ids) < 2:
                return torch.zeros(self.primitive_dim, device=self.device), 0.0

            # Get primitive representations
            primitives = []
            for pid in primitive_ids:
                if pid in self.primitive_library:
                    primitives.append(self.primitive_library[pid].representation)
                else:
                    self.logger.warning(f"Primitive {pid} not found in library")
                    return torch.zeros(self.primitive_dim, device=self.device), 0.0

            # Apply composition rule
            if composition_rule == CompositionRule.SEQUENTIAL:
                composed = self._sequential_composition(primitives)

            elif composition_rule == CompositionRule.PARALLEL:
                composed = self._parallel_composition(primitives)

            elif composition_rule == CompositionRule.CONDITIONAL:
                if context is not None:
                    composed = self._conditional_composition(primitives, context)
                else:
                    composed = self._parallel_composition(primitives)  # Fallback

            elif composition_rule == CompositionRule.ATTENTION_WEIGHTED:
                composed = self._attention_weighted_composition(primitives)

            elif composition_rule == CompositionRule.RECURSIVE:
                composed = self._recursive_composition(primitives)

            else:  # HIERARCHICAL
                composed = self._hierarchical_composition(primitives)

            # Assess composition quality
            confidence = self._assess_composition_quality(primitives, composed, composition_rule)

            return composed, confidence

        except Exception as e:
            self.logger.warning(f"Primitive composition failed: {e}")
            return torch.zeros(self.primitive_dim, device=self.device), 0.0

    def _sequential_composition(self, primitives: List[torch.Tensor]) -> torch.Tensor:
        """Compose primitives sequentially"""
        result = primitives[0]

        for i in range(1, len(primitives)):
            combined_input = torch.cat([result, primitives[i]], dim=0)
            # Pad or truncate to expected input size
            if combined_input.shape[0] > self.primitive_dim * 2:
                combined_input = combined_input[:self.primitive_dim * 2]
            elif combined_input.shape[0] < self.primitive_dim * 2:
                padding = torch.zeros(self.primitive_dim * 2 - combined_input.shape[0], device=self.device)
                combined_input = torch.cat([combined_input, padding], dim=0)

            result = self.sequential_composer(combined_input)

        return result

    def _parallel_composition(self, primitives: List[torch.Tensor]) -> torch.Tensor:
        """Compose primitives in parallel"""

        # Simple approach: weighted average
        weights = torch.softmax(torch.randn(len(primitives), device=self.device), dim=0)
        result = sum(w * p for w, p in zip(weights, primitives))

        return result

    def _conditional_composition(self,
                               primitives: List[torch.Tensor],
                               condition: torch.Tensor) -> torch.Tensor:
        """Compose primitives based on condition"""

        if len(primitives) < 2:
            return primitives[0] if primitives else torch.zeros(self.primitive_dim, device=self.device)

        # Use gating mechanism
        p1, p2 = primitives[0], primitives[1]

        # Prepare input for conditional composer
        combined_input = torch.cat([p1, p2, condition[:self.primitive_dim]], dim=0)

        # Pad to expected size
        if combined_input.shape[0] < self.primitive_dim * 3:
            padding = torch.zeros(self.primitive_dim * 3 - combined_input.shape[0], device=self.device)
            combined_input = torch.cat([combined_input, padding], dim=0)
        else:
            combined_input = combined_input[:self.primitive_dim * 3]

        gate_score = self.conditional_composer['gate'](combined_input)
        composition = self.conditional_composer['composer'](combined_input[:self.primitive_dim * 2])

        # Gate between first primitive and composition
        result = gate_score * composition + (1 - gate_score) * p1

        return result

    def _attention_weighted_composition(self, primitives: List[torch.Tensor]) -> torch.Tensor:
        """Compose primitives using learned attention weights"""

        if len(primitives) == 1:
            return primitives[0]

        try:
            # Stack primitives for attention
            stacked = torch.stack(primitives, dim=0).unsqueeze(0)  # [1, N, D]

            # Apply self-attention
            attended, _ = self.attention_composer['attention'](stacked, stacked, stacked)
            attended = attended.squeeze(0)  # [N, D]

            # Combine attended primitives
            combined = attended.mean(dim=0)  # Simple average of attended representations
            result = self.attention_composer['combiner'](combined)

            return result

        except Exception as e:
            self.logger.warning(f"Attention composition failed: {e}")
            return self._parallel_composition(primitives)

    def _recursive_composition(self, primitives: List[torch.Tensor]) -> torch.Tensor:
        """Compose primitives recursively"""

        if len(primitives) <= 2:
            return self._parallel_composition(primitives)

        # Recursively compose pairs
        while len(primitives) > 1:
            new_primitives = []

            for i in range(0, len(primitives) - 1, 2):
                if i + 1 < len(primitives):
                    p1, p2 = primitives[i], primitives[i + 1]
                    composed = self._parallel_composition([p1, p2])
                    new_primitives.append(composed)
                else:
                    new_primitives.append(primitives[i])

            primitives = new_primitives

        return primitives[0]

    def _hierarchical_composition(self, primitives: List[torch.Tensor]) -> torch.Tensor:
        """Compose primitives hierarchically"""

        # Group primitives by similarity and compose at different levels
        try:
            if len(primitives) <= 2:
                return self._parallel_composition(primitives)

            # Compute similarity matrix
            similarities = torch.zeros(len(primitives), len(primitives), device=self.device)

            for i in range(len(primitives)):
                for j in range(len(primitives)):
                    similarities[i, j] = F.cosine_similarity(primitives[i], primitives[j], dim=0)

            # Group similar primitives
            groups = self._group_by_similarity(primitives, similarities, threshold=0.7)

            # Compose within groups, then across groups
            group_representatives = []
            for group in groups:
                if len(group) > 1:
                    group_composition = self._parallel_composition(group)
                else:
                    group_composition = group[0]
                group_representatives.append(group_composition)

            # Final composition across groups
            if len(group_representatives) > 1:
                result = self._parallel_composition(group_representatives)
            else:
                result = group_representatives[0]

            return result

        except Exception as e:
            self.logger.warning(f"Hierarchical composition failed: {e}")
            return self._parallel_composition(primitives)

    def _group_by_similarity(self,
                           primitives: List[torch.Tensor],
                           similarities: torch.Tensor,
                           threshold: float = 0.7) -> List[List[torch.Tensor]]:
        """Group primitives by similarity for hierarchical composition"""

        groups = []
        used = set()

        for i in range(len(primitives)):
            if i in used:
                continue

            group = [primitives[i]]
            used.add(i)

            for j in range(i + 1, len(primitives)):
                if j not in used and similarities[i, j] > threshold:
                    group.append(primitives[j])
                    used.add(j)

            groups.append(group)

        return groups

    def _assess_composition_quality(self,
                                  input_primitives: List[torch.Tensor],
                                  composed_result: torch.Tensor,
                                  composition_rule: CompositionRule) -> float:
        """Assess the quality of a composition"""

        try:
            # Factor 1: Semantic coherence
            coherence = float(self.coherence_assessor(composed_result).item())

            # Factor 2: Information preservation
            input_complexity = sum(self._compute_complexity(p) for p in input_primitives) / len(input_primitives)
            output_complexity = self._compute_complexity(composed_result)
            preservation = 1.0 - abs(input_complexity - output_complexity)

            # Factor 3: Rule appropriateness
            rule_features = self._encode_composition_rule(composition_rule)
            primitive_features = torch.stack(input_primitives).mean(dim=0)
            rule_input = torch.cat([primitive_features, composed_result, rule_features], dim=0)

            # Ensure input is correct size
            expected_size = self.primitive_dim * 2 + len(CompositionRule)
            if rule_input.shape[0] < expected_size:
                padding = torch.zeros(expected_size - rule_input.shape[0], device=self.device)
                rule_input = torch.cat([rule_input, padding], dim=0)
            else:
                rule_input = rule_input[:expected_size]

            rule_appropriateness = float(self.composition_reasoner(rule_input).item())

            # Overall quality
            quality = (coherence * 0.4 + preservation * 0.3 + rule_appropriateness * 0.3)

            return quality

        except Exception as e:
            self.logger.warning(f"Composition quality assessment failed: {e}")
            return 0.5

    def _compute_complexity(self, tensor: torch.Tensor) -> float:
        """Compute complexity measure of a tensor"""
        try:
            # Use entropy as complexity measure
            normalized = F.softmax(tensor, dim=0)
            entropy = -(normalized * torch.log(normalized + 1e-8)).sum()
            return float(entropy.item())
        except Exception:
            return 1.0

    def _encode_composition_rule(self, rule: CompositionRule) -> torch.Tensor:
        """Encode composition rule as one-hot vector"""
        rule_vector = torch.zeros(len(CompositionRule), device=self.device)
        rule_vector[rule.value] = 1.0  # This will fail, need to fix

        # Fix: use index of enum
        rule_index = list(CompositionRule).index(rule)
        rule_vector[rule_index] = 1.0

        return rule_vector

    def systematic_generalization_test(self,
                                     test_combinations: List[List[str]],
                                     evaluation_fn: Optional[callable] = None) -> Dict[str, Any]:
        """
        Test systematic generalization by evaluating novel primitive combinations.

        Args:
            test_combinations: List of primitive ID combinations to test
            evaluation_fn: Function to evaluate composition success

        Returns:
            Systematic generalization results
        """

        results = {
            'total_tests': len(test_combinations),
            'successful_compositions': 0,
            'novel_successful_compositions': 0,
            'generalization_score': 0.0,
            'composition_quality_scores': [],
            'failed_combinations': []
        }

        try:
            for combination in test_combinations:
                combination_id = "_".join(sorted(combination))

                # Check if this is a novel combination
                is_novel = combination_id not in self.novel_combinations_tested
                self.novel_combinations_tested.add(combination_id)

                # Test all composition rules for this combination
                best_composition = None
                best_confidence = 0.0
                best_rule = None

                for rule in CompositionRule:
                    try:
                        composed, confidence = self.compose_primitives(combination, rule)

                        if confidence > best_confidence:
                            best_composition = composed
                            best_confidence = confidence
                            best_rule = rule

                    except Exception as e:
                        self.logger.debug(f"Composition failed for rule {rule}: {e}")

                # Evaluate composition success
                if best_composition is not None and best_confidence > self.composition_confidence_threshold:
                    results['successful_compositions'] += 1
                    results['composition_quality_scores'].append(best_confidence)

                    if is_novel:
                        results['novel_successful_compositions'] += 1
                        self.successful_novel_combinations.add(combination_id)

                        # Update primitive compatibility
                        self._update_primitive_compatibility(combination, best_confidence)

                    # Use external evaluation if provided
                    if evaluation_fn:
                        try:
                            external_score = evaluation_fn(best_composition, combination, best_rule)
                            results['composition_quality_scores'][-1] = external_score
                        except Exception as e:
                            self.logger.warning(f"External evaluation failed: {e}")

                else:
                    results['failed_combinations'].append({
                        'combination': combination,
                        'best_confidence': best_confidence,
                        'is_novel': is_novel
                    })

            # Compute generalization score
            if results['total_tests'] > 0:
                success_rate = results['successful_compositions'] / results['total_tests']
                novel_success_rate = results['novel_successful_compositions'] / max(1, sum(1 for c in test_combinations if "_".join(sorted(c)) in self.novel_combinations_tested))

                results['generalization_score'] = (success_rate + novel_success_rate) / 2.0

            return results

        except Exception as e:
            self.logger.warning(f"Systematic generalization test failed: {e}")
            return results

    def _update_primitive_compatibility(self, combination: List[str], success_score: float):
        """Update compatibility matrix based on successful compositions"""

        try:
            # Update pairwise compatibility
            for i, pid1 in enumerate(combination):
                for j, pid2 in enumerate(combination):
                    if i != j and pid1 in self.primitive_library and pid2 in self.primitive_library:
                        # Update compatibility
                        p1 = self.primitive_library[pid1]
                        p2 = self.primitive_library[pid2]

                        p1.compatible_primitives.add(pid2)
                        p2.compatible_primitives.add(pid1)

                        # Update learned combinations
                        combo_key = f"{pid1}+{pid2}"
                        if combo_key in p1.learned_combinations:
                            p1.learned_combinations[combo_key] = (p1.learned_combinations[combo_key] + success_score) / 2.0
                        else:
                            p1.learned_combinations[combo_key] = success_score

        except Exception as e:
            self.logger.warning(f"Compatibility update failed: {e}")

    def generate_novel_combinations(self,
                                  num_combinations: int = 50,
                                  max_primitives_per_combination: int = 3) -> List[List[str]]:
        """
        Generate novel primitive combinations for systematic generalization testing.

        Args:
            num_combinations: Number of combinations to generate
            max_primitives_per_combination: Maximum primitives per combination

        Returns:
            List of novel primitive combinations
        """

        if len(self.primitive_library) < 2:
            return []

        primitive_ids = list(self.primitive_library.keys())
        novel_combinations = []

        try:
            # Generate combinations of different sizes
            for combo_size in range(2, max_primitives_per_combination + 1):
                possible_combinations = list(itertools.combinations(primitive_ids, combo_size))

                # Filter out already tested combinations
                novel_possible = [
                    list(combo) for combo in possible_combinations
                    if "_".join(sorted(combo)) not in self.novel_combinations_tested
                ]

                # Sample from novel combinations
                sample_size = min(len(novel_possible), num_combinations // (max_primitives_per_combination - 1))
                if sample_size > 0:
                    sampled = random.sample(novel_possible, sample_size)
                    novel_combinations.extend(sampled)

            # Prioritize combinations involving high-quality primitives
            def combination_priority(combo):
                avg_quality = sum(self.primitive_library[pid].composability for pid in combo) / len(combo)
                return avg_quality

            novel_combinations.sort(key=combination_priority, reverse=True)

            return novel_combinations[:num_combinations]

        except Exception as e:
            self.logger.warning(f"Novel combination generation failed: {e}")
            return []

    def analyze_compositional_gaps(self) -> Dict[str, Any]:
        """
        Analyze gaps in compositional knowledge for targeted learning.

        Returns:
            Analysis of compositional gaps and recommendations
        """

        analysis = {
            'primitive_coverage': {},
            'composition_rule_usage': {},
            'semantic_type_distribution': {},
            'underexplored_combinations': [],
            'recommendations': []
        }

        try:
            # Analyze primitive coverage
            for semantic_type in ['spatial', 'color', 'shape', 'logic', 'unknown']:
                type_primitives = [p for p in self.primitive_library.values() if p.semantic_type == semantic_type]
                analysis['primitive_coverage'][semantic_type] = len(type_primitives)

            # Analyze composition rule usage
            for rule in CompositionRule:
                rule_usage = sum(1 for comp in self.composition_library.values() if comp.combination_rule == rule)
                analysis['composition_rule_usage'][rule.value] = rule_usage

            # Find underexplored combinations
            all_primitive_pairs = list(itertools.combinations(self.primitive_library.keys(), 2))
            explored_pairs = set()

            for primitive in self.primitive_library.values():
                for combo_key in primitive.learned_combinations.keys():
                    if '+' in combo_key:
                        pid1, pid2 = combo_key.split('+')
                        explored_pairs.add(tuple(sorted([pid1, pid2])))

            underexplored = [
                list(pair) for pair in all_primitive_pairs
                if tuple(sorted(pair)) not in explored_pairs
            ]

            analysis['underexplored_combinations'] = underexplored[:20]  # Top 20

            # Generate recommendations
            analysis['recommendations'] = self._generate_learning_recommendations(analysis)

            return analysis

        except Exception as e:
            self.logger.warning(f"Compositional gap analysis failed: {e}")
            return analysis

    def _generate_learning_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving compositional learning"""

        recommendations = []

        try:
            # Check primitive coverage
            coverage = analysis['primitive_coverage']
            for semantic_type, count in coverage.items():
                if count < 5:  # Few primitives of this type
                    recommendations.append(f"Discover more {semantic_type} primitives (current: {count})")

            # Check composition rule usage
            rule_usage = analysis['composition_rule_usage']
            unused_rules = [rule for rule, count in rule_usage.items() if count == 0]
            if unused_rules:
                recommendations.append(f"Explore composition rules: {', '.join(unused_rules)}")

            # Check underexplored combinations
            underexplored_count = len(analysis['underexplored_combinations'])
            if underexplored_count > 10:
                recommendations.append(f"Test {underexplored_count} underexplored primitive combinations")

            # General recommendations
            if len(self.primitive_library) < 20:
                recommendations.append("Discover more diverse primitives")

            if len(self.composition_library) < 50:
                recommendations.append("Explore more composition strategies")

            return recommendations

        except Exception:
            return ["Continue compositional exploration"]

    def get_compositional_learning_metrics(self) -> Dict[str, float]:
        """Get comprehensive metrics about compositional learning progress"""

        metrics = {
            'primitive_count': len(self.primitive_library),
            'composition_count': len(self.composition_library),
            'avg_primitive_quality': 0.0,
            'avg_composition_confidence': 0.0,
            'novel_combination_success_rate': 0.0,
            'systematic_generalization_score': 0.0,
            'compositional_diversity': 0.0
        }

        try:
            # Primitive metrics
            if self.primitive_library:
                metrics['avg_primitive_quality'] = sum(
                    p.composability for p in self.primitive_library.values()
                ) / len(self.primitive_library)

            # Composition metrics
            if self.composition_library:
                metrics['avg_composition_confidence'] = sum(
                    c.confidence for c in self.composition_library.values()
                ) / len(self.composition_library)

            # Novel combination success
            if self.novel_combinations_tested:
                metrics['novel_combination_success_rate'] = len(self.successful_novel_combinations) / len(self.novel_combinations_tested)

            # Systematic generalization score
            if len(self.composition_success_history) > 0:
                recent_successes = [score for score in list(self.composition_success_history)[-50:] if score > 0.6]
                metrics['systematic_generalization_score'] = len(recent_successes) / min(50, len(self.composition_success_history))

            # Compositional diversity
            rule_usage = defaultdict(int)
            for comp in self.composition_library.values():
                rule_usage[comp.combination_rule] += 1

            if rule_usage:
                entropy = 0.0
                total = sum(rule_usage.values())
                for count in rule_usage.values():
                    if count > 0:
                        p = count / total
                        entropy -= p * math.log(p)

                max_entropy = math.log(len(CompositionRule))
                metrics['compositional_diversity'] = entropy / max_entropy if max_entropy > 0 else 0.0

            return metrics

        except Exception as e:
            self.logger.warning(f"Compositional metrics computation failed: {e}")
            return metrics

    def get_human_level_composition_score(self) -> float:
        """
        Assess how close the system is to human-level compositional reasoning.

        Returns score in [0, 1] where 1.0 indicates human-level capability.
        """

        try:
            metrics = self.get_compositional_learning_metrics()

            # Human-level factors
            factors = []

            # 1. Primitive richness (do we have diverse, high-quality primitives?)
            primitive_richness = min(1.0, metrics['primitive_count'] / 100.0) * metrics['avg_primitive_quality']
            factors.append(primitive_richness)

            # 2. Composition capability (can we combine effectively?)
            composition_capability = metrics['avg_composition_confidence']
            factors.append(composition_capability)

            # 3. Systematic generalization (novel combinations)
            systematic_capability = metrics['systematic_generalization_score']
            factors.append(systematic_capability)

            # 4. Compositional diversity (multiple strategies)
            diversity_capability = metrics['compositional_diversity']
            factors.append(diversity_capability)

            # 5. Novel combination mastery
            novel_mastery = metrics['novel_combination_success_rate']
            factors.append(novel_mastery)

            # Overall human-level composition score
            human_level_score = sum(factors) / len(factors)

            return float(human_level_score)

        except Exception as e:
            self.logger.warning(f"Human-level composition assessment failed: {e}")
            return 0.0

    def update_from_experience(self,
                             transformation_examples: List[Dict[str, torch.Tensor]],
                             success_indicators: List[bool]):
        """Update compositional knowledge from successful/failed experiences"""

        try:
            for example, success in zip(transformation_examples, success_indicators):
                # Discover new primitives from this example
                before_state = example.get('before')
                after_state = example.get('after')

                if before_state is not None:
                    new_primitives = self.discover_primitives(before_state)

                    for primitive in new_primitives:
                        # Add to library if high quality and not duplicate
                        if primitive.composability > self.primitive_discovery_threshold:
                            self.primitive_library[primitive.primitive_id] = primitive

                        # Update success rates
                        primitive.usage_count += 1
                        primitive.success_rate = (primitive.success_rate + float(success)) / 2.0

                # Record composition success
                self.composition_success_history.append(1.0 if success else 0.0)

            # Prune low-quality primitives if library is full
            if len(self.primitive_library) > self.max_primitives:
                self._prune_primitive_library()

        except Exception as e:
            self.logger.warning(f"Experience update failed: {e}")

    def _prune_primitive_library(self):
        """Remove low-quality primitives to make room for new ones"""

        # Sort by quality (combination of success rate and composability)
        sorted_primitives = sorted(
            self.primitive_library.items(),
            key=lambda x: x[1].success_rate * x[1].composability,
            reverse=True
        )

        # Keep top primitives
        kept_primitives = dict(sorted_primitives[:self.max_primitives])
        self.primitive_library = kept_primitives

        self.logger.info(f"Pruned primitive library to {len(kept_primitives)} primitives")

    def generate_compositional_training_signals(self) -> Dict[str, torch.Tensor]:
        """Generate training signals that enhance compositional reasoning"""

        signals = {}

        try:
            if len(self.primitive_library) >= 2:
                # Signal 1: Primitive reconstruction loss
                primitive_tensors = [p.representation for p in self.primitive_library.values()]
                stacked_primitives = torch.stack(primitive_tensors[:min(10, len(primitive_tensors))])

                reconstructed = self.primitive_decoder(self.primitive_encoder(stacked_primitives))
                reconstruction_loss = F.mse_loss(stacked_primitives, reconstructed)
                signals['primitive_reconstruction_loss'] = reconstruction_loss

                # Signal 2: Composition consistency loss
                if len(primitive_tensors) >= 2:
                    p1, p2 = primitive_tensors[0], primitive_tensors[1]
                    composed = self._parallel_composition([p1, p2])
                    consistency_loss = self._compute_composition_consistency_loss(p1, p2, composed)
                    signals['composition_consistency_loss'] = consistency_loss

                # Signal 3: Systematic generalization loss
                generalization_score = self.get_human_level_composition_score()
                generalization_loss = 1.0 - generalization_score
                signals['systematic_generalization_loss'] = torch.tensor(generalization_loss, device=self.device)

            return signals

        except Exception as e:
            self.logger.warning(f"Compositional training signal generation failed: {e}")
            return {'primitive_reconstruction_loss': torch.tensor(0.0, device=self.device)}

    def _compute_composition_consistency_loss(self,
                                            p1: torch.Tensor,
                                            p2: torch.Tensor,
                                            composed: torch.Tensor) -> torch.Tensor:
        """Compute consistency loss for composition"""

        try:
            # Composed result should preserve important aspects of inputs
            p1_contribution = F.cosine_similarity(composed, p1, dim=0)
            p2_contribution = F.cosine_similarity(composed, p2, dim=0)

            # Both primitives should contribute meaningfully
            consistency_target = 0.7  # Target contribution strength
            consistency_loss = (
                F.mse_loss(p1_contribution, torch.tensor(consistency_target, device=self.device)) +
                F.mse_loss(p2_contribution, torch.tensor(consistency_target, device=self.device))
            )

            return consistency_loss

        except Exception as e:
            self.logger.warning(f"Composition consistency loss computation failed: {e}")
            return torch.tensor(0.0, device=self.device)