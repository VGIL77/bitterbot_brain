#!/usr/bin/env python3
"""
Causal Discovery Module for TOPAS
Enables understanding of WHY transformations work, not just that they work.
Critical for human-level causal reasoning and systematic generalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import math
import logging
import numpy as np
from collections import defaultdict, deque

class CausalRelationType(Enum):
    """Types of causal relationships"""
    DIRECT_CAUSE = "direct_cause"           # A directly causes B
    INDIRECT_CAUSE = "indirect_cause"       # A causes B through intermediates
    NECESSARY_CONDITION = "necessary"       # A is necessary for B
    SUFFICIENT_CONDITION = "sufficient"     # A is sufficient for B
    INHIBITORY = "inhibitory"               # A prevents B
    MODERATING = "moderating"               # A modifies the A→B relationship

@dataclass
class CausalHypothesis:
    """A causal hypothesis about a transformation"""
    cause_features: torch.Tensor
    effect_features: torch.Tensor
    relation_type: CausalRelationType
    strength: float
    confidence: float
    supporting_evidence: List[Dict] = None
    refuting_evidence: List[Dict] = None

    def __post_init__(self):
        if self.supporting_evidence is None:
            self.supporting_evidence = []
        if self.refuting_evidence is None:
            self.refuting_evidence = []

@dataclass
class CausalGraph:
    """Represents learned causal relationships"""
    nodes: Dict[str, torch.Tensor]  # feature_id -> feature_vector
    edges: Dict[Tuple[str, str], CausalHypothesis]  # (cause, effect) -> hypothesis
    intervention_history: List[Dict] = None

    def __post_init__(self):
        if self.intervention_history is None:
            self.intervention_history = []

class CausalDiscoveryEngine(nn.Module):
    """
    Causal discovery engine that learns WHY transformations work.
    Implements intervention-based causal learning and counterfactual reasoning.
    """

    def __init__(self,
                 feature_dim: int = 512,
                 max_causal_graph_size: int = 1000,
                 intervention_strength: float = 0.3,
                 device: str = "cpu"):
        super().__init__()

        self.device = torch.device(device)
        self.feature_dim = feature_dim
        self.max_causal_graph_size = max_causal_graph_size
        self.intervention_strength = intervention_strength
        self.logger = logging.getLogger(__name__)

        # Causal discovery networks
        self.cause_detector = self._create_cause_detector()
        self.effect_predictor = self._create_effect_predictor()
        self.confound_detector = self._create_confound_detector()
        self.intervention_generator = self._create_intervention_generator()

        # Causal graph storage
        self.causal_graph = CausalGraph(nodes={}, edges={})
        self.intervention_queue = deque(maxlen=100)

        # Learning dynamics
        self.causal_learning_rate = 0.1
        self.evidence_threshold = 0.7
        self.confidence_decay = 0.95

        # Counterfactual reasoning
        self.counterfactual_memory = {}  # (cause, effect) -> counterfactual examples

    def _create_cause_detector(self) -> nn.Module:
        """Network to detect potential causal features"""
        return nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),  # Before + After features
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.LayerNorm(self.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 2, 1),
            nn.Sigmoid()  # Causal strength score
        )

    def _create_effect_predictor(self) -> nn.Module:
        """Network to predict effects from causes"""
        return nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.Tanh()
        )

    def _create_confound_detector(self) -> nn.Module:
        """Network to detect confounding variables"""
        return nn.Sequential(
            nn.Linear(self.feature_dim * 3, self.feature_dim),  # Cause + Effect + Context
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 2, 1),
            nn.Sigmoid()  # Confounding strength
        )

    def _create_intervention_generator(self) -> nn.Module:
        """Network to generate interventions for causal testing"""
        return nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.Tanh()
        )

    def discover_causal_relationships(self,
                                    before_state: torch.Tensor,
                                    after_state: torch.Tensor,
                                    transformation_context: Optional[Dict[str, Any]] = None) -> List[CausalHypothesis]:
        """
        Discover causal relationships between before and after states.

        Args:
            before_state: [B, T, D] features before transformation
            after_state: [B, T, D] features after transformation
            transformation_context: Additional context about the transformation

        Returns:
            List of causal hypotheses discovered
        """

        hypotheses = []

        try:
            B, T, D = before_state.shape

            # Compute state differences
            state_diff = after_state - before_state

            # Extract potential causal features
            combined_features = torch.cat([before_state, after_state], dim=-1)  # [B, T, 2*D]

            # Detect causal relationships
            causal_scores = self.cause_detector(combined_features)  # [B, T, 1]

            # Find significant causal relationships
            causal_threshold = 0.6
            significant_causals = causal_scores.squeeze(-1) > causal_threshold  # [B, T]

            for b in range(B):
                for t in range(T):
                    if significant_causals[b, t]:
                        # Extract cause and effect features
                        cause_features = before_state[b, t]
                        effect_features = after_state[b, t]

                        # Predict effect from cause
                        predicted_effect = self.effect_predictor(cause_features)

                        # Measure prediction accuracy
                        prediction_accuracy = F.cosine_similarity(
                            predicted_effect, effect_features, dim=0
                        )

                        if prediction_accuracy > 0.5:  # Good prediction = likely causal
                            # Determine causal relation type
                            strength = float(causal_scores[b, t, 0].item())
                            relation_type = self._classify_causal_relation(
                                cause_features, effect_features, state_diff[b, t], strength
                            )

                            hypothesis = CausalHypothesis(
                                cause_features=cause_features.detach(),
                                effect_features=effect_features.detach(),
                                relation_type=relation_type,
                                strength=strength,
                                confidence=float(prediction_accuracy.item())
                            )

                            hypotheses.append(hypothesis)

            # Add context-aware hypotheses if transformation context provided
            if transformation_context:
                context_hypotheses = self._generate_context_hypotheses(
                    before_state, after_state, transformation_context
                )
                hypotheses.extend(context_hypotheses)

            return hypotheses

        except Exception as e:
            self.logger.warning(f"Causal discovery failed: {e}")
            return []

    def _classify_causal_relation(self,
                                cause: torch.Tensor,
                                effect: torch.Tensor,
                                difference: torch.Tensor,
                                strength: float) -> CausalRelationType:
        """Classify the type of causal relationship"""

        try:
            # Analyze the relationship characteristics
            cause_magnitude = torch.norm(cause)
            effect_magnitude = torch.norm(effect)
            diff_magnitude = torch.norm(difference)

            # Simple heuristics for classification
            if strength > 0.8 and diff_magnitude > effect_magnitude * 0.5:
                return CausalRelationType.DIRECT_CAUSE

            elif strength > 0.6 and cause_magnitude > effect_magnitude * 1.2:
                return CausalRelationType.SUFFICIENT_CONDITION

            elif strength > 0.6 and cause_magnitude < effect_magnitude * 0.8:
                return CausalRelationType.NECESSARY_CONDITION

            elif strength > 0.4:
                return CausalRelationType.INDIRECT_CAUSE

            else:
                return CausalRelationType.MODERATING

        except Exception:
            return CausalRelationType.INDIRECT_CAUSE

    def _generate_context_hypotheses(self,
                                   before_state: torch.Tensor,
                                   after_state: torch.Tensor,
                                   context: Dict[str, Any]) -> List[CausalHypothesis]:
        """Generate hypotheses based on transformation context"""

        hypotheses = []

        try:
            # If DSL operations are provided in context
            if 'operations' in context:
                operations = context['operations']

                for op in operations:
                    # Create operation-specific causal hypothesis
                    op_effect = self._compute_operation_effect(before_state, after_state, op)

                    if op_effect is not None:
                        hypothesis = CausalHypothesis(
                            cause_features=torch.tensor([hash(op) % 1000] * self.feature_dim, dtype=torch.float32),
                            effect_features=op_effect,
                            relation_type=CausalRelationType.DIRECT_CAUSE,
                            strength=0.8,
                            confidence=0.7
                        )
                        hypotheses.append(hypothesis)

            return hypotheses

        except Exception as e:
            self.logger.warning(f"Context hypothesis generation failed: {e}")
            return []

    def _compute_operation_effect(self,
                                before_state: torch.Tensor,
                                after_state: torch.Tensor,
                                operation: str) -> Optional[torch.Tensor]:
        """Compute the effect signature of a specific operation"""

        try:
            # Compute the difference attributed to this operation
            state_diff = after_state - before_state

            # Create operation signature
            op_signature = self._create_operation_signature(operation)

            # Correlate with state difference to extract operation effect
            effect = state_diff.mean(dim=(0, 1))  # Average effect across batch and time

            return effect

        except Exception:
            return None

    def _create_operation_signature(self, operation: str) -> torch.Tensor:
        """Create a signature tensor for a DSL operation"""

        # Simple hash-based signature (in practice, could be learned)
        op_hash = hash(operation) % 1000
        signature = torch.zeros(self.feature_dim)

        # Create sparse signature based on operation hash
        for i in range(0, self.feature_dim, 10):
            if (op_hash + i) % 3 == 0:
                signature[i] = 1.0

        return signature

    def perform_intervention(self,
                           target_features: torch.Tensor,
                           intervention_type: str = "random") -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Perform causal intervention for testing causal hypotheses.
        Critical for establishing causality rather than correlation.
        """

        try:
            original_features = target_features.clone()

            if intervention_type == "random":
                # Random intervention
                intervention_mask = torch.rand_like(target_features) < self.intervention_strength
                noise = torch.randn_like(target_features) * 0.5
                intervened_features = torch.where(intervention_mask, noise, target_features)

            elif intervention_type == "ablation":
                # Ablation intervention (set features to zero)
                intervention_mask = torch.rand_like(target_features) < self.intervention_strength
                intervened_features = torch.where(intervention_mask, 0.0, target_features)

            elif intervention_type == "amplification":
                # Amplify existing features
                intervention_mask = torch.rand_like(target_features) < self.intervention_strength
                amplified = target_features * 2.0
                intervened_features = torch.where(intervention_mask, amplified, target_features)

            else:
                # Targeted intervention using learned intervention generator
                intervention_vector = self.intervention_generator(target_features.mean(dim=(0, 1)))
                intervention_mask = torch.rand_like(target_features) < self.intervention_strength
                intervened_features = torch.where(
                    intervention_mask.unsqueeze(0).unsqueeze(0),
                    intervention_vector.unsqueeze(0).unsqueeze(0),
                    target_features
                )

            # Record intervention details
            intervention_record = {
                'type': intervention_type,
                'strength': self.intervention_strength,
                'affected_dimensions': int(intervention_mask.sum().item()) if 'intervention_mask' in locals() else 0,
                'magnitude': float(torch.norm(intervened_features - original_features).item())
            }

            return intervened_features, intervention_record

        except Exception as e:
            self.logger.warning(f"Intervention failed: {e}")
            return target_features, {'type': 'failed', 'error': str(e)}

    def test_causal_hypothesis(self,
                             hypothesis: CausalHypothesis,
                             test_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Test a causal hypothesis using interventions and counterfactuals.

        Args:
            hypothesis: The causal hypothesis to test
            test_data: Test data containing 'before', 'after', 'context'

        Returns:
            Test results with confidence scores
        """

        results = {
            'causal_strength': 0.0,
            'intervention_evidence': 0.0,
            'counterfactual_evidence': 0.0,
            'overall_confidence': 0.0
        }

        try:
            before_features = test_data.get('before')
            after_features = test_data.get('after')

            if before_features is None or after_features is None:
                return results

            # Test 1: Causal strength prediction
            predicted_effect = self.effect_predictor(hypothesis.cause_features)
            actual_effect = after_features.mean(dim=(0, 1))

            prediction_accuracy = F.cosine_similarity(predicted_effect, actual_effect, dim=0)
            results['causal_strength'] = float(prediction_accuracy.item())

            # Test 2: Intervention evidence
            intervened_features, intervention_record = self.perform_intervention(
                before_features, intervention_type="targeted"
            )

            # See if intervention changes the effect as predicted
            intervention_effect = self.effect_predictor(intervened_features.mean(dim=(0, 1)))
            expected_change = intervention_effect - predicted_effect

            # Measure how well intervention matched expectations
            intervention_accuracy = 1.0 - torch.norm(expected_change).item()
            results['intervention_evidence'] = float(max(0.0, intervention_accuracy))

            # Test 3: Counterfactual evidence
            counterfactual_score = self._test_counterfactual(hypothesis, before_features, after_features)
            results['counterfactual_evidence'] = counterfactual_score

            # Overall confidence
            results['overall_confidence'] = (
                results['causal_strength'] * 0.4 +
                results['intervention_evidence'] * 0.3 +
                results['counterfactual_evidence'] * 0.3
            )

            # Update hypothesis confidence
            hypothesis.confidence = (hypothesis.confidence + results['overall_confidence']) / 2.0

            return results

        except Exception as e:
            self.logger.warning(f"Hypothesis testing failed: {e}")
            return results

    def _test_counterfactual(self,
                           hypothesis: CausalHypothesis,
                           before_features: torch.Tensor,
                           after_features: torch.Tensor) -> float:
        """Test counterfactual: 'What if the cause had been different?'"""

        try:
            # Generate counterfactual cause
            counterfactual_cause = hypothesis.cause_features + torch.randn_like(hypothesis.cause_features) * 0.3

            # Predict counterfactual effect
            counterfactual_effect = self.effect_predictor(counterfactual_cause)

            # Measure how different the counterfactual effect would be
            actual_effect = after_features.mean(dim=(0, 1))
            counterfactual_difference = torch.norm(counterfactual_effect - actual_effect)

            # Higher difference = stronger causal evidence
            # (changing cause significantly changes effect)
            evidence_strength = torch.sigmoid(counterfactual_difference * 2.0)

            return float(evidence_strength.item())

        except Exception as e:
            self.logger.debug(f"Counterfactual test failed: {e}")
            return 0.0

    def learn_causal_mechanisms(self,
                              transformation_examples: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """
        Learn causal mechanisms from a set of transformation examples.
        Builds up the causal graph through multiple observations.
        """

        learning_stats = {
            'hypotheses_generated': 0,
            'hypotheses_confirmed': 0,
            'hypotheses_refuted': 0,
            'new_causal_links': 0
        }

        try:
            for example in transformation_examples:
                before_state = example.get('before')
                after_state = example.get('after')
                context = example.get('context', {})

                if before_state is None or after_state is None:
                    continue

                # Generate causal hypotheses for this example
                hypotheses = self.discover_causal_relationships(before_state, after_state, context)
                learning_stats['hypotheses_generated'] += len(hypotheses)

                # Test each hypothesis
                for hypothesis in hypotheses:
                    test_results = self.test_causal_hypothesis(hypothesis, example)

                    if test_results['overall_confidence'] > self.evidence_threshold:
                        # Confirmed hypothesis - add to causal graph
                        self._add_to_causal_graph(hypothesis)
                        learning_stats['hypotheses_confirmed'] += 1
                        learning_stats['new_causal_links'] += 1

                    elif test_results['overall_confidence'] < 0.3:
                        # Refuted hypothesis
                        learning_stats['hypotheses_refuted'] += 1

            # Prune weak causal links
            self._prune_causal_graph()

            return learning_stats

        except Exception as e:
            self.logger.warning(f"Causal mechanism learning failed: {e}")
            return learning_stats

    def _add_to_causal_graph(self, hypothesis: CausalHypothesis):
        """Add a confirmed hypothesis to the causal graph"""

        try:
            # Create feature IDs
            cause_id = f"cause_{len(self.causal_graph.nodes)}"
            effect_id = f"effect_{len(self.causal_graph.nodes) + 1}"

            # Add nodes
            self.causal_graph.nodes[cause_id] = hypothesis.cause_features
            self.causal_graph.nodes[effect_id] = hypothesis.effect_features

            # Add edge
            self.causal_graph.edges[(cause_id, effect_id)] = hypothesis

            self.logger.debug(f"Added causal link: {cause_id} → {effect_id} ({hypothesis.relation_type.value})")

        except Exception as e:
            self.logger.warning(f"Failed to add hypothesis to causal graph: {e}")

    def _prune_causal_graph(self):
        """Remove weak or outdated causal relationships"""

        if len(self.causal_graph.edges) <= self.max_causal_graph_size:
            return

        # Sort edges by confidence and keep top ones
        sorted_edges = sorted(
            self.causal_graph.edges.items(),
            key=lambda x: x[1].confidence,
            reverse=True
        )

        # Keep top edges
        kept_edges = dict(sorted_edges[:self.max_causal_graph_size])

        # Update graph
        self.causal_graph.edges = kept_edges

        # Clean up orphaned nodes
        used_nodes = set()
        for (cause_id, effect_id) in kept_edges.keys():
            used_nodes.add(cause_id)
            used_nodes.add(effect_id)

        self.causal_graph.nodes = {
            node_id: features for node_id, features in self.causal_graph.nodes.items()
            if node_id in used_nodes
        }

    def generate_causal_explanation(self,
                                  transformation: Dict[str, torch.Tensor],
                                  top_k: int = 5) -> Dict[str, Any]:
        """
        Generate a causal explanation for why a transformation works.
        Critical for interpretability and human-level understanding.
        """

        explanation = {
            'primary_causes': [],
            'causal_chain': [],
            'confidence': 0.0,
            'explanation_text': '',
            'causal_mechanisms': []
        }

        try:
            before_state = transformation.get('before')
            after_state = transformation.get('after')

            if before_state is None or after_state is None:
                explanation['explanation_text'] = "Insufficient data for causal explanation"
                return explanation

            # Find matching causal relationships in graph
            matching_edges = []

            for (cause_id, effect_id), hypothesis in self.causal_graph.edges.items():
                # Check similarity to current transformation
                cause_similarity = F.cosine_similarity(
                    hypothesis.cause_features, before_state.mean(dim=(0, 1)), dim=0
                )
                effect_similarity = F.cosine_similarity(
                    hypothesis.effect_features, after_state.mean(dim=(0, 1)), dim=0
                )

                if cause_similarity > 0.6 and effect_similarity > 0.6:
                    overall_match = (cause_similarity + effect_similarity) / 2.0
                    matching_edges.append((hypothesis, float(overall_match.item())))

            # Sort by match quality
            matching_edges.sort(key=lambda x: x[1], reverse=True)

            # Extract top causal explanations
            for hypothesis, match_score in matching_edges[:top_k]:
                causal_mechanism = {
                    'relation_type': hypothesis.relation_type.value,
                    'strength': hypothesis.strength,
                    'confidence': hypothesis.confidence,
                    'match_score': match_score
                }
                explanation['causal_mechanisms'].append(causal_mechanism)

            # Generate explanation text
            if explanation['causal_mechanisms']:
                primary_mechanism = explanation['causal_mechanisms'][0]
                explanation['confidence'] = primary_mechanism['confidence']

                explanation['explanation_text'] = self._generate_explanation_text(
                    explanation['causal_mechanisms']
                )
            else:
                explanation['explanation_text'] = "No matching causal mechanisms found"

            return explanation

        except Exception as e:
            self.logger.warning(f"Causal explanation generation failed: {e}")
            explanation['explanation_text'] = f"Explanation generation failed: {e}"
            return explanation

    def _generate_explanation_text(self, mechanisms: List[Dict[str, Any]]) -> str:
        """Generate human-readable causal explanation"""

        try:
            if not mechanisms:
                return "No causal mechanisms identified"

            primary = mechanisms[0]
            relation_type = primary['relation_type']
            confidence = primary['confidence']

            base_explanations = {
                'direct_cause': f"Direct causal transformation (confidence: {confidence:.2f})",
                'indirect_cause': f"Indirect causal effect through intermediate steps (confidence: {confidence:.2f})",
                'necessary': f"Necessary precondition for transformation (confidence: {confidence:.2f})",
                'sufficient': f"Sufficient condition for transformation (confidence: {confidence:.2f})",
                'inhibitory': f"Inhibitory effect preventing alternative outcomes (confidence: {confidence:.2f})",
                'moderating': f"Moderating influence on transformation strength (confidence: {confidence:.2f})"
            }

            explanation = base_explanations.get(relation_type, f"Unknown causal relation: {relation_type}")

            # Add supporting mechanisms if multiple found
            if len(mechanisms) > 1:
                explanation += f" [+{len(mechanisms)-1} supporting mechanisms]"

            return explanation

        except Exception:
            return "Explanation generation failed"

    def predict_intervention_outcome(self,
                                   current_state: torch.Tensor,
                                   proposed_intervention: Dict[str, Any]) -> Tuple[torch.Tensor, float]:
        """
        Predict the outcome of a proposed intervention using learned causal knowledge.
        Enables counterfactual reasoning and planning.
        """

        try:
            # Find relevant causal mechanisms
            relevant_mechanisms = self._find_relevant_mechanisms(current_state, proposed_intervention)

            if not relevant_mechanisms:
                # No relevant mechanisms - return current state with low confidence
                return current_state, 0.1

            # Simulate intervention effects
            predicted_state = current_state.clone()
            total_confidence = 0.0

            for mechanism in relevant_mechanisms:
                hypothesis = mechanism['hypothesis']
                relevance = mechanism['relevance']

                # Apply causal effect prediction
                predicted_effect = self.effect_predictor(hypothesis.cause_features)

                # Weight by relevance and confidence
                weight = relevance * hypothesis.confidence
                predicted_state = predicted_state + weight * predicted_effect.unsqueeze(0).unsqueeze(0)
                total_confidence += weight

            # Normalize confidence
            final_confidence = min(1.0, total_confidence / len(relevant_mechanisms))

            return predicted_state, final_confidence

        except Exception as e:
            self.logger.warning(f"Intervention prediction failed: {e}")
            return current_state, 0.1

    def _find_relevant_mechanisms(self,
                                current_state: torch.Tensor,
                                intervention: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find causal mechanisms relevant to the proposed intervention"""

        relevant = []

        try:
            state_features = current_state.mean(dim=(0, 1))

            for (cause_id, effect_id), hypothesis in self.causal_graph.edges.items():
                # Check relevance based on feature similarity
                cause_relevance = F.cosine_similarity(
                    hypothesis.cause_features, state_features, dim=0
                )

                if cause_relevance > 0.5:
                    relevant.append({
                        'hypothesis': hypothesis,
                        'relevance': float(cause_relevance.item()),
                        'cause_id': cause_id,
                        'effect_id': effect_id
                    })

            # Sort by relevance
            relevant.sort(key=lambda x: x['relevance'], reverse=True)

            return relevant[:5]  # Top 5 most relevant

        except Exception as e:
            self.logger.warning(f"Mechanism relevance search failed: {e}")
            return []

    def analyze_causal_sufficiency(self,
                                 transformation_examples: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """
        Analyze whether the discovered causal mechanisms are sufficient to explain transformations.
        Critical for assessing causal understanding completeness.
        """

        analysis = {
            'coverage': 0.0,           # How many examples can be explained
            'accuracy': 0.0,           # How accurately we can predict effects
            'completeness': 0.0,       # How complete our causal understanding is
            'consistency': 0.0         # How consistent our causal explanations are
        }

        try:
            explained_count = 0
            total_accuracy = 0.0
            consistency_scores = []

            for example in transformation_examples:
                explanation = self.generate_causal_explanation(example)

                if explanation['confidence'] > 0.5:
                    explained_count += 1

                    # Measure prediction accuracy
                    predicted_effect = self.effect_predictor(example['before'].mean(dim=(0, 1)))
                    actual_effect = example['after'].mean(dim=(0, 1))
                    accuracy = F.cosine_similarity(predicted_effect, actual_effect, dim=0)
                    total_accuracy += float(accuracy.item())

                    # Measure consistency with other similar examples
                    consistency = self._measure_explanation_consistency(explanation, transformation_examples)
                    consistency_scores.append(consistency)

            # Compute analysis metrics
            analysis['coverage'] = explained_count / max(1, len(transformation_examples))
            analysis['accuracy'] = total_accuracy / max(1, explained_count)
            analysis['consistency'] = sum(consistency_scores) / max(1, len(consistency_scores))

            # Completeness based on causal graph richness
            graph_density = len(self.causal_graph.edges) / max(1, len(self.causal_graph.nodes))
            analysis['completeness'] = min(1.0, graph_density)

            return analysis

        except Exception as e:
            self.logger.warning(f"Causal sufficiency analysis failed: {e}")
            return analysis

    def _measure_explanation_consistency(self,
                                       explanation: Dict[str, Any],
                                       all_examples: List[Dict[str, torch.Tensor]]) -> float:
        """Measure how consistent an explanation is with other similar examples"""

        try:
            if not explanation['causal_mechanisms']:
                return 0.0

            primary_mechanism = explanation['causal_mechanisms'][0]
            relation_type = primary_mechanism['relation_type']

            # Count similar explanations
            similar_count = 0
            total_similar_examples = 0

            for other_example in all_examples:
                other_explanation = self.generate_causal_explanation(other_example)

                if other_explanation['causal_mechanisms']:
                    total_similar_examples += 1
                    other_primary = other_explanation['causal_mechanisms'][0]

                    if other_primary['relation_type'] == relation_type:
                        similar_count += 1

            consistency = similar_count / max(1, total_similar_examples)
            return consistency

        except Exception:
            return 0.5

    def get_causal_insights(self) -> Dict[str, Any]:
        """Get insights about learned causal relationships"""

        insights = {
            'total_causal_links': len(self.causal_graph.edges),
            'total_features': len(self.causal_graph.nodes),
            'relation_type_distribution': defaultdict(int),
            'strong_causal_links': 0,
            'causal_complexity': 0.0
        }

        try:
            # Analyze relation types
            for hypothesis in self.causal_graph.edges.values():
                insights['relation_type_distribution'][hypothesis.relation_type.value] += 1

                if hypothesis.confidence > 0.8:
                    insights['strong_causal_links'] += 1

            # Compute causal complexity (average path length in graph)
            if self.causal_graph.edges:
                insights['causal_complexity'] = self._compute_graph_complexity()

            return insights

        except Exception as e:
            self.logger.warning(f"Causal insights computation failed: {e}")
            return insights

    def _compute_graph_complexity(self) -> float:
        """Compute the complexity of the causal graph"""

        try:
            # Simple approximation: ratio of edges to nodes
            num_edges = len(self.causal_graph.edges)
            num_nodes = len(self.causal_graph.nodes)

            if num_nodes == 0:
                return 0.0

            complexity = num_edges / num_nodes
            return min(1.0, complexity)  # Normalize to [0, 1]

        except Exception:
            return 0.0

    def generate_causal_training_signals(self,
                                       current_examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Generate training signals that enhance causal understanding.
        These signals can be used in the main training loop.
        """

        signals = {}

        try:
            # Signal 1: Causal consistency loss
            consistency_losses = []

            for example in current_examples:
                explanation = self.generate_causal_explanation(example)
                if explanation['confidence'] > 0.5:
                    # Generate consistency loss for this explanation
                    predicted_effect = self.effect_predictor(example['before'].mean(dim=(0, 1)))
                    actual_effect = example['after'].mean(dim=(0, 1))
                    consistency_loss = F.mse_loss(predicted_effect, actual_effect)
                    consistency_losses.append(consistency_loss)

            if consistency_losses:
                signals['causal_consistency_loss'] = torch.stack(consistency_losses).mean()

            # Signal 2: Intervention prediction loss
            if len(current_examples) >= 2:
                # Use one example to predict intervention on another
                base_example = current_examples[0]
                test_example = current_examples[1]

                predicted_outcome, confidence = self.predict_intervention_outcome(
                    base_example['before'], {'type': 'transfer_test'}
                )

                intervention_loss = F.mse_loss(
                    predicted_outcome.mean(dim=(0, 1)),
                    test_example['after'].mean(dim=(0, 1))
                )
                signals['intervention_prediction_loss'] = intervention_loss

            # Signal 3: Counterfactual reasoning loss
            counterfactual_losses = []

            for example in current_examples:
                if len(self.causal_graph.edges) > 0:
                    # Generate counterfactual
                    hypothesis = list(self.causal_graph.edges.values())[0]
                    counterfactual_score = self._test_counterfactual(
                        hypothesis, example['before'], example['after']
                    )

                    # Loss should encourage high counterfactual evidence
                    counterfactual_loss = 1.0 - counterfactual_score
                    counterfactual_losses.append(torch.tensor(counterfactual_loss))

            if counterfactual_losses:
                signals['counterfactual_loss'] = torch.stack(counterfactual_losses).mean()

            return signals

        except Exception as e:
            self.logger.warning(f"Causal training signal generation failed: {e}")
            return {'causal_consistency_loss': torch.tensor(0.0)}

    def assess_causal_understanding_quality(self) -> float:
        """
        Assess the quality of learned causal understanding.
        Returns score in [0, 1] where 1.0 indicates strong causal understanding.
        """

        try:
            if not self.causal_graph.edges:
                return 0.0

            # Factor 1: Graph complexity and richness
            graph_richness = min(1.0, len(self.causal_graph.edges) / 50.0)

            # Factor 2: Average confidence of causal hypotheses
            avg_confidence = sum(h.confidence for h in self.causal_graph.edges.values()) / len(self.causal_graph.edges)

            # Factor 3: Diversity of causal relation types
            relation_types = set(h.relation_type for h in self.causal_graph.edges.values())
            relation_diversity = len(relation_types) / len(CausalRelationType)

            # Factor 4: Intervention success rate (from history)
            intervention_success = 0.5  # Placeholder - would track from actual interventions

            # Overall causal understanding quality
            quality_score = (
                graph_richness * 0.3 +
                avg_confidence * 0.3 +
                relation_diversity * 0.2 +
                intervention_success * 0.2
            )

            return float(quality_score)

        except Exception as e:
            self.logger.warning(f"Causal understanding assessment failed: {e}")
            return 0.0