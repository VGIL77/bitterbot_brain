#!/usr/bin/env python3
"""
Cognitive Synchronization Protocol for TOPAS
Coordinates signals between all cognitive components for unified intelligence
"""
import torch
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

class CognitivePhase(Enum):
    """Different phases of cognitive processing"""
    ACTIVE = "active"           # NeuroPlanner reasoning + TOPAS execution
    CONSOLIDATION = "consolidation"  # Dream processing + RelMem binding
    GENERATION = "generation"   # Self-Play synthesis + Dopamine reinforcement
    INTEGRATION = "integration" # Cross-system signal exchange

@dataclass
class CognitiveSignal:
    """Unified signal format for cross-component communication"""
    source: str
    target: str
    signal_type: str
    data: Any
    priority: float = 1.0
    timestamp: float = field(default_factory=time.time)

    def is_urgent(self) -> bool:
        return self.priority > 2.0

@dataclass
class CognitiveState:
    """Unified cognitive state across all components"""
    # Current processing state
    phase: CognitivePhase = CognitivePhase.ACTIVE
    global_step: int = 0

    # Performance metrics
    em_score: float = 0.0
    acc_score: float = 0.0
    breakthrough_threshold: float = 0.33

    # Component activations
    neuroplanner_active: bool = False
    dream_active: bool = False
    relmem_active: bool = False
    selfplay_active: bool = False
    dopamine_active: bool = False

    # Shared patterns and discoveries
    recent_patterns: List[torch.Tensor] = field(default_factory=list)
    breakthrough_signals: List[CognitiveSignal] = field(default_factory=list)
    curiosity_targets: List[str] = field(default_factory=list)

    # Cross-system attention
    attention_focus: Optional[str] = None
    priority_signals: List[CognitiveSignal] = field(default_factory=list)

class CognitiveSynchronizer:
    """Orchestrates unified cognitive processing across all TOPAS components"""

    def __init__(self, device="cuda:0"):
        self.device = device
        self.state = CognitiveState()
        self.signal_queue: List[CognitiveSignal] = []
        self.component_registry: Dict[str, Any] = {}

        # Cognitive timing
        self.cycle_length = 100  # Steps per complete cognitive cycle
        self.phase_transitions = {
            CognitivePhase.ACTIVE: CognitivePhase.CONSOLIDATION,
            CognitivePhase.CONSOLIDATION: CognitivePhase.GENERATION,
            CognitivePhase.GENERATION: CognitivePhase.INTEGRATION,
            CognitivePhase.INTEGRATION: CognitivePhase.ACTIVE
        }

        # Performance tracking
        self.breakthrough_history: List[Tuple[int, float]] = []
        self.pattern_library: Dict[str, torch.Tensor] = {}

        # Unified Experience Buffer for cross-component learning
        self.unified_experiences = {
            "successful_traces": [],        # From STaR bootstrapper
            "reasoning_patterns": [],       # From NeuroPlanner
            "memory_concepts": [],          # From RelMem
            "dream_motifs": [],            # From Dream Engine
            "synthesis_themes": [],         # From emergent patterns
            "curiosity_discoveries": []     # From GCCRF
        }

        # Consciousness cycle coordination
        self.dream_consolidation_queue = []
        self.pattern_synthesis_queue = []
        self.integration_pending = []

    def register_component(self, name: str, component: Any):
        """Register a cognitive component for coordination"""
        self.component_registry[name] = component
        print(f"[CogSync] Registered component: {name}")

    def emit_signal(self, signal: CognitiveSignal):
        """Emit a signal to be processed by the synchronizer"""
        if signal.is_urgent():
            self.state.priority_signals.append(signal)
        else:
            self.signal_queue.append(signal)

    def process_breakthrough(self, em_score: float, acc_score: float,
                           patterns: List[torch.Tensor], global_step: int):
        """Process breakthrough event and coordinate system-wide response"""
        self.state.em_score = em_score
        self.state.acc_score = acc_score
        self.state.global_step = global_step

        is_breakthrough = em_score >= self.state.breakthrough_threshold

        if is_breakthrough:
            # Create breakthrough signal
            breakthrough_signal = CognitiveSignal(
                source="topas_main",
                target="all_systems",
                signal_type="breakthrough",
                data={
                    "em_score": em_score,
                    "acc_score": acc_score,
                    "patterns": patterns,
                    "step": global_step
                },
                priority=3.0  # High priority
            )

            self.emit_signal(breakthrough_signal)
            self.breakthrough_history.append((global_step, em_score))

            # Store patterns for cross-system learning
            for i, pattern in enumerate(patterns):
                if pattern.numel() > 0:
                    self.pattern_library[f"breakthrough_{global_step}_{i}"] = pattern.detach().cpu()

            print(f"[CogSync] 🎯 BREAKTHROUGH at step {global_step}: EM={em_score:.1%}")

        return is_breakthrough

    def should_transition_phase(self) -> bool:
        """Determine if we should transition to the next cognitive phase"""
        steps_in_phase = self.state.global_step % self.cycle_length

        # Transition every cycle_length/4 steps (4 phases per cycle)
        phase_duration = self.cycle_length // 4
        return steps_in_phase % phase_duration == 0

    def transition_phase(self):
        """Transition to the next cognitive phase"""
        old_phase = self.state.phase
        self.state.phase = self.phase_transitions[old_phase]

        # Deactivate components from previous phase
        self._deactivate_phase_components(old_phase)

        # Activate components for new phase
        self._activate_phase_components(self.state.phase)

        print(f"[CogSync] Phase transition: {old_phase.value} → {self.state.phase.value}")

    def _activate_phase_components(self, phase: CognitivePhase):
        """Activate components appropriate for the current phase"""
        if phase == CognitivePhase.ACTIVE:
            self.state.neuroplanner_active = True
            self.state.attention_focus = "reasoning"

        elif phase == CognitivePhase.CONSOLIDATION:
            self.state.dream_active = True
            self.state.relmem_active = True
            self.state.attention_focus = "memory_formation"

        elif phase == CognitivePhase.GENERATION:
            self.state.selfplay_active = True
            self.state.dopamine_active = True
            self.state.attention_focus = "pattern_synthesis"

        elif phase == CognitivePhase.INTEGRATION:
            self.state.attention_focus = "cross_system_integration"
            # All systems briefly active for signal exchange

    def _deactivate_phase_components(self, phase: CognitivePhase):
        """Deactivate components from the previous phase"""
        if phase == CognitivePhase.ACTIVE:
            self.state.neuroplanner_active = False

        elif phase == CognitivePhase.CONSOLIDATION:
            self.state.dream_active = False
            self.state.relmem_active = False

        elif phase == CognitivePhase.GENERATION:
            self.state.selfplay_active = False
            self.state.dopamine_active = False

    def get_active_components(self) -> List[str]:
        """Get list of currently active components"""
        active = []
        if self.state.neuroplanner_active:
            active.append("neuroplanner")
        if self.state.dream_active:
            active.append("dream")
        if self.state.relmem_active:
            active.append("relmem")
        if self.state.selfplay_active:
            active.append("selfplay")
        if self.state.dopamine_active:
            active.append("dopamine")
        return active

    def process_signals(self) -> Dict[str, List[CognitiveSignal]]:
        """Process queued signals and route them to appropriate components"""
        # Handle priority signals first
        priority_signals = self.state.priority_signals.copy()
        self.state.priority_signals.clear()

        # Regular signals
        regular_signals = self.signal_queue.copy()
        self.signal_queue.clear()

        # Route signals by target
        routed_signals = {}
        for signal in priority_signals + regular_signals:
            if signal.target not in routed_signals:
                routed_signals[signal.target] = []
            routed_signals[signal.target].append(signal)

        return routed_signals

    def get_cross_system_guidance(self) -> Dict[str, Any]:
        """Provide guidance for cross-system learning"""
        guidance = {
            "current_phase": self.state.phase.value,
            "active_components": self.get_active_components(),
            "attention_focus": self.state.attention_focus,
            "recent_breakthroughs": self.breakthrough_history[-5:],  # Last 5 breakthroughs
            "available_patterns": list(self.pattern_library.keys())[-10:],  # Last 10 patterns
            "performance_trend": self._calculate_performance_trend()
        }
        return guidance

    def _calculate_performance_trend(self) -> str:
        """Calculate recent performance trend"""
        if len(self.breakthrough_history) < 2:
            return "insufficient_data"

        recent_scores = [score for _, score in self.breakthrough_history[-5:]]
        if len(recent_scores) >= 2:
            trend = recent_scores[-1] - recent_scores[0]
            if trend > 0.05:
                return "improving"
            elif trend < -0.05:
                return "declining"
            else:
                return "stable"
        return "unknown"

    def add_experience(self, experience_type: str, data: Any):
        """Add experience to unified buffer for cross-component learning"""
        if experience_type in self.unified_experiences:
            self.unified_experiences[experience_type].append(data)

            # Maintain buffer size
            max_size = 100
            if len(self.unified_experiences[experience_type]) > max_size:
                self.unified_experiences[experience_type] = self.unified_experiences[experience_type][-max_size:]

    def trigger_consciousness_cycle(self):
        """Trigger enhanced processing during consciousness cycles"""
        print(f"[CogSync] 🌙 CONSCIOUSNESS CYCLE at step {self.state.global_step}")

        # Dream consolidation phase
        if self.state.phase == CognitivePhase.CONSOLIDATION and self.component_registry.get("dream"):
            self._coordinate_dream_consolidation()

        # Memory integration phase
        if self.state.phase == CognitivePhase.CONSOLIDATION and self.component_registry.get("relmem"):
            self._coordinate_memory_integration()

        # Pattern synthesis phase
        if self.state.phase == CognitivePhase.GENERATION:
            self._coordinate_pattern_synthesis()

    def _coordinate_dream_consolidation(self):
        """Coordinate dream processing with other components"""
        dream_engine = self.component_registry.get("dream")
        if not dream_engine:
            return

        # Share recent breakthrough patterns with dream engine
        recent_patterns = self.unified_experiences["successful_traces"][-10:]
        if recent_patterns:
            self.emit_signal(CognitiveSignal(
                source="cognitive_sync",
                target="dream",
                signal_type="pattern_consolidation",
                data={"patterns": recent_patterns},
                priority=2.0
            ))

    def _coordinate_memory_integration(self):
        """Coordinate relational memory with other cognitive components"""
        relmem = self.component_registry.get("relmem")
        if not relmem:
            return

        # Share reasoning patterns with relational memory
        reasoning_patterns = self.unified_experiences["reasoning_patterns"][-5:]
        if reasoning_patterns:
            self.emit_signal(CognitiveSignal(
                source="cognitive_sync",
                target="relmem",
                signal_type="concept_formation",
                data={"patterns": reasoning_patterns},
                priority=2.0
            ))

    def _coordinate_pattern_synthesis(self):
        """Coordinate pattern synthesis across components"""
        # Synthesize patterns from all experience types
        all_patterns = []
        for exp_type, experiences in self.unified_experiences.items():
            if experiences:
                all_patterns.extend(experiences[-3:])  # Recent experiences from each type

        if all_patterns:
            self.pattern_synthesis_queue.extend(all_patterns)

    def assess_emergent_intelligence(self) -> Dict[str, float]:
        """Assess emergent intelligence properties across the unified system"""
        assessment = {
            "pattern_diversity": 0.0,
            "cross_component_coherence": 0.0,
            "adaptive_learning_rate": 0.0,
            "transfer_capability": 0.0,
            "curiosity_effectiveness": 0.0,
            "memory_consolidation": 0.0,
            "overall_intelligence": 0.0
        }

        try:
            # Pattern diversity: How many unique patterns are actively maintained
            total_patterns = sum(len(experiences) for experiences in self.unified_experiences.values())
            assessment["pattern_diversity"] = min(1.0, total_patterns / 500.0)  # Normalize to [0,1]

            # Cross-component coherence: How well components are sharing information
            signal_activity = len(self.signal_queue) + len(self.state.priority_signals)
            assessment["cross_component_coherence"] = min(1.0, signal_activity / 20.0)

            # Transfer capability: How well patterns generalize across tasks
            breakthrough_frequency = len(self.breakthrough_history) / max(1, self.state.global_step / 100)
            assessment["transfer_capability"] = min(1.0, breakthrough_frequency)

            # Memory consolidation: How effectively experiences are being retained
            consolidation_score = len(self.pattern_library) / max(1, len(self.breakthrough_history))
            assessment["memory_consolidation"] = min(1.0, consolidation_score)

            # Overall intelligence: Composite score
            assessment["overall_intelligence"] = (
                assessment["pattern_diversity"] * 0.25 +
                assessment["cross_component_coherence"] * 0.25 +
                assessment["transfer_capability"] * 0.30 +
                assessment["memory_consolidation"] * 0.20
            )

        except Exception as e:
            print(f"[CogSync] Intelligence assessment failed: {e}")

        return assessment

    def generate_intelligence_report(self) -> str:
        """Generate a comprehensive intelligence report"""
        assessment = self.assess_emergent_intelligence()
        guidance = self.get_cross_system_guidance()

        report = f"""
🧠 UNIFIED COGNITIVE INTELLIGENCE REPORT (Step {self.state.global_step})
================================================================

📊 Emergent Intelligence Metrics:
  • Pattern Diversity: {assessment['pattern_diversity']:.2%}
  • Cross-Component Coherence: {assessment['cross_component_coherence']:.2%}
  • Transfer Capability: {assessment['transfer_capability']:.2%}
  • Memory Consolidation: {assessment['memory_consolidation']:.2%}
  • Overall Intelligence: {assessment['overall_intelligence']:.2%}

🔄 Cognitive State:
  • Current Phase: {self.state.phase.value.upper()}
  • Active Components: {', '.join(guidance['active_components'])}
  • Attention Focus: {guidance.get('attention_focus', 'distributed')}
  • Performance Trend: {guidance.get('performance_trend', 'unknown').upper()}

📈 Learning Progress:
  • Total Breakthroughs: {len(self.breakthrough_history)}
  • Pattern Library Size: {len(self.pattern_library)}
  • Recent EM Score: {self.state.em_score:.2%}
  • System Integration: {"ACTIVE" if len(self.unified_experiences) > 0 else "INACTIVE"}

💡 Recommendations:
"""

        # Add dynamic recommendations based on assessment
        if assessment['overall_intelligence'] < 0.3:
            report += "  • LOW INTELLIGENCE: Increase cross-component signal frequency\n"
        elif assessment['overall_intelligence'] > 0.7:
            report += "  • HIGH INTELLIGENCE: System showing emergent properties\n"

        if assessment['transfer_capability'] < 0.2:
            report += "  • POOR TRANSFER: Enhance pattern generalization mechanisms\n"

        if assessment['cross_component_coherence'] < 0.3:
            report += "  • LOW COHERENCE: Improve component communication protocols\n"

        return report

    def step(self, global_step: int) -> Dict[str, Any]:
        """Execute one step of cognitive synchronization"""
        self.state.global_step = global_step

        # Check for phase transitions
        if self.should_transition_phase():
            self.transition_phase()

        # Trigger consciousness cycles at phase boundaries
        if self.state.phase == CognitivePhase.CONSOLIDATION:
            self.trigger_consciousness_cycle()

        # Process signals
        routed_signals = self.process_signals()

        # Return coordination info for components
        return {
            "cognitive_state": self.state,
            "routed_signals": routed_signals,
            "guidance": self.get_cross_system_guidance(),
            "active_components": self.get_active_components(),
            "phase_info": {
                "current_phase": self.state.phase.value,
                "step_in_cycle": global_step % self.cycle_length,
                "next_transition": self.cycle_length - (global_step % self.cycle_length)
            }
        }