"""
Strict Hippocampal Sharp-Wave Ripple Substrate
- No dummy coherence (1.0) for insufficient samples
- No silent skips on phase update
- No warnings-only on forced events
- Deterministic amplitude in strict mode
- Phase entropy must be computed from real events
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

STRICT_RIPPLE = True

def _ensure(cond: bool, msg: str):
    if not cond:
        raise RuntimeError(msg)


@dataclass
class RippleConfig:
    """Configuration for hippocampal ripple substrate"""
    event_rate_hz: float = 0.8
    center_freq_hz: float = 170.0
    burst_duration_ms: tuple = (50.0, 150.0)
    dt_ms: float = 2.0
    stdp_gain: float = 3.0
    phase_lock: bool = True
    phase_bins: int = 8
    coherence_window_ms: float = 200.0
    metric_kappa: Optional[float] = None
    qref_threshold: float = 0.5
    buffer_size: int = 1000
    coherence_history: int = 50
    min_event_rate: float = 0.1
    max_event_rate: float = 5.0
    min_center_freq: float = 120.0
    max_center_freq: float = 250.0

    def __post_init__(self):
        if self.dt_ms <= 0:
            raise ValueError("dt_ms must be positive")
        fs_hz = 1000.0 / self.dt_ms
        nyquist_freq = 0.45 * fs_hz
        if self.center_freq_hz > nyquist_freq:
            raise ValueError(f"center_freq {self.center_freq_hz} exceeds Nyquist {nyquist_freq}")
        if not (self.min_event_rate <= self.event_rate_hz <= self.max_event_rate):
            raise ValueError("event_rate outside valid range")
        if not (self.min_center_freq <= self.center_freq_hz <= self.max_center_freq):
            raise ValueError("center_freq outside valid range")
        if self.burst_duration_ms[0] >= self.burst_duration_ms[1]:
            raise ValueError("burst_duration_ms min must be less than max")
        if self.stdp_gain <= 0:
            raise ValueError("stdp_gain must be positive")


class RippleContext:
    __slots__ = [
        'is_active','phase','amplitude','frequency','burst_start_time',
        'burst_duration','stdp_multiplier','coherence','phase_bin',
        'qref_warp_factor','replay_alignment','event_id'
    ]
    def __init__(self):
        self.is_active=False; self.phase=0.0; self.amplitude=0.0
        self.frequency=0.0; self.burst_start_time=0.0; self.burst_duration=0.0
        self.stdp_multiplier=1.0; self.coherence=0.0; self.phase_bin=0
        self.qref_warp_factor=1.0; self.replay_alignment=0.0; self.event_id=0
    def reset(self): self.__init__()


class RippleSubstrate:
    def __init__(self, config: RippleConfig):
        self.config=config; self.context=RippleContext()
        self._fs_hz=1000.0/config.dt_ms
        self._validate_nyquist()
        self._current_time=0.0; self._next_event_time=0.0; self._event_counter=0
        self._phase_accumulator=0.0; self._last_phase_time=0.0
        self._phase_history=[]; self._amplitude_history=[]
        self._coherence_buffer=np.zeros(config.coherence_history)
        self._coherence_index=0
        self._event_buffer=[]
        self.stats={'total_events':0,'total_duration':0.0,
                    'avg_coherence':0.0,'phase_distribution':np.zeros(config.phase_bins),
                    'stdp_activations':0}

        # === SPIKE PATTERN CONSOLIDATION STATE ===
        self._last_consolidation_event = None
        self._consolidation_count = 0

        self._schedule_next_event()

    # --- Readiness gate ---
    def is_ready(self) -> bool:
        """Require a minimal history to compute meaningful coherence."""
        return len(self._phase_history) >= 10 and self.context.is_active

    # Convenience accessors expected by DreamEngine (valid only when ready)
    def get_coherence_metrics(self) -> dict:
        _ensure(self.is_ready(), "[Ripple] coherence requested before sufficient samples")
        return {"current_coherence": float(self.context.coherence)}

    def get_phase_info(self) -> dict:
        _ensure(self.is_ready(), "[Ripple] phase requested before sufficient samples")
        # normalize phase to [0,1)
        return {"phase_normalized": float(self.context.phase / (2*np.pi))}

    def get_stdp_gain(self) -> float:
        _ensure(self.context.is_active, "[Ripple] stdp gain requested outside ripple")
        return float(self.context.stdp_multiplier)

    def _validate_nyquist(self):
        safe_freq=0.45*self._fs_hz
        if self.config.center_freq_hz>safe_freq:
            raise ValueError(f"center {self.config.center_freq_hz} exceeds safe Nyquist {safe_freq}")

    def update(self,current_time:float):
        _ensure(current_time>=0,"[Ripple] current_time must be non-negative")
        self._current_time=current_time
        if current_time>=self._next_event_time and not self.context.is_active:
            self._trigger_ripple_event()
        if self.context.is_active: self._update_active_ripple()

    def _schedule_next_event(self):
        interval=np.random.exponential(1.0/self.config.event_rate_hz)
        self._next_event_time=self._current_time+interval

    def _trigger_ripple_event(self):
        self._event_counter+=1
        min_dur,max_dur=self.config.burst_duration_ms
        dur=np.random.uniform(min_dur,max_dur)/1000.0
        self.context.is_active=True; self.context.burst_start_time=self._current_time
        self.context.burst_duration=dur; self.context.event_id=self._event_counter
        self.context.frequency=self.config.center_freq_hz+np.random.normal(0,5.0)
        self.context.stdp_multiplier=self.config.stdp_gain
        self._phase_accumulator=0.0; self._last_phase_time=self._current_time
        self.context.amplitude=1.0
        self.stats['total_events']+=1; self.stats['stdp_activations']+=1

    def _update_active_ripple(self):
        elapsed=self._current_time-self.context.burst_start_time
        if elapsed>=self.context.burst_duration: return self._end_ripple_burst()
        self._update_phase(); self._update_amplitude(elapsed); self._update_coherence()

    def _update_phase(self):
        dt=self._current_time-self._last_phase_time
        if dt<=0:
            if STRICT_RIPPLE: raise RuntimeError("[Ripple] Non-positive dt in phase update")
            return
        self._phase_accumulator+=2*np.pi*self.context.frequency*dt
        self.context.phase=self._phase_accumulator%(2*np.pi)
        self.context.phase_bin=min(int(self.context.phase/(2*np.pi)*self.config.phase_bins),
                                   self.config.phase_bins-1)
        self.stats['phase_distribution'][self.context.phase_bin]+=1
        self._last_phase_time=self._current_time

    def _update_amplitude(self,elapsed:float):
        midpoint=self.context.burst_duration/2.0; sigma=self.context.burst_duration/4.0
        envelope=np.exp(-0.5*((elapsed-midpoint)/sigma)**2)
        hf_mod=1.0+0.1*np.sin(2*np.pi*40*elapsed)
        base_amp=1.0 if STRICT_RIPPLE else 1.0+0.2*np.random.randn()
        self.context.amplitude=max(0.1,base_amp*envelope*hf_mod)

    def _update_coherence(self):
        self._phase_history.append(self.context.phase)
        self._amplitude_history.append(self.context.amplitude)
        if len(self._phase_history)>int(self.config.coherence_window_ms):
            self._phase_history.pop(0); self._amplitude_history.pop(0)
        if len(self._phase_history)<10:
            if STRICT_RIPPLE: raise RuntimeError("[Ripple] Not enough samples for coherence")
            self.context.coherence=1.0
        else:
            diffs=np.diff(np.array(self._phase_history))
            diffs=np.angle(np.exp(1j*diffs))
            expected=2*np.pi*self.context.frequency/1000.0
            phase_consistency=np.exp(-np.var(diffs-expected))
            amp_stability=np.exp(-np.var(self._amplitude_history)/np.mean(self._amplitude_history)**2)
            self.context.coherence=float(np.clip(0.7*phase_consistency+0.3*amp_stability,0.0,1.0))
        self._coherence_buffer[self._coherence_index]=self.context.coherence
        self._coherence_index=(self._coherence_index+1)%self.config.coherence_history
        if self.stats['total_events']>0:
            self.stats['avg_coherence']=np.mean(self._coherence_buffer[:min(self.stats['total_events'],
                                                                           self.config.coherence_history)])

    def _end_ripple_burst(self):
        duration=self._current_time-self.context.burst_start_time
        coherence = self.context.coherence
        self.stats['total_duration']+=duration

        # Store event info
        event_info = {'id':self.context.event_id,'duration':duration,
                     'freq':self.context.frequency,'coherence':coherence}
        self._event_buffer.append(event_info)
        if len(self._event_buffer)>self.config.buffer_size: self._event_buffer.pop(0)

        # === SPIKE PATTERN CONSOLIDATION TRIGGER ===
        # High-coherence ripples (>0.8) trigger memory consolidation
        consolidation_triggered = False
        if coherence > 0.8 and duration > 0.08:  # High coherence + sufficient duration
            consolidation_triggered = True
            # Store consolidation event for external systems to respond to
            self._last_consolidation_event = {
                'triggered_at': self._current_time,
                'coherence': coherence,
                'duration': duration,
                'phase_at_end': self.context.phase,
                'event_id': self.context.event_id
            }
            logger.info(f"[Ripple] Consolidation triggered: coherence={coherence:.3f}, duration={duration:.3f}s")

        self.context.reset(); self._phase_history.clear(); self._amplitude_history.clear()
        self._schedule_next_event()

        # Update consolidation stats
        if consolidation_triggered:
            self._consolidation_count += 1

        return consolidation_triggered

    def check_consolidation_event(self) -> Optional[Dict]:
        """
        Check for pending consolidation events and return details.
        Returns event info and clears the trigger (consume once).
        """
        if self._last_consolidation_event is not None:
            event = self._last_consolidation_event.copy()
            self._last_consolidation_event = None  # Clear after consumption
            return event
        return None

    def get_consolidation_stats(self) -> Dict[str, Any]:
        """Get consolidation-related statistics."""
        return {
            'total_consolidations': self._consolidation_count,
            'consolidation_rate': (self._consolidation_count / max(1, self.stats['total_events'])),
            'avg_consolidation_coherence': np.mean([
                event['coherence'] for event in self._event_buffer
                if event['coherence'] > 0.8
            ]) if any(event['coherence'] > 0.8 for event in self._event_buffer) else 0.0
        }

    def get_statistics(self)->Dict[str,Any]:
        stats=self.stats.copy()
        if stats['total_events']>0:
            stats['avg_event_duration']=stats['total_duration']/stats['total_events']
        else:
            if STRICT_RIPPLE: raise RuntimeError("[Ripple] No events yet; statistics undefined")
            stats['avg_event_duration']=0.0
        return stats

    def force_ripple_event(self,duration_ms:Optional[float]=None):
        if self.context.is_active:
            if STRICT_RIPPLE: raise RuntimeError("Cannot force ripple event while active")
            return
        self._next_event_time=self._current_time
        self.update(self._current_time)

    # Legacy compatibility methods for existing dream engine integration
    def is_ripple_active(self) -> bool:
        """Check if ripple is currently active"""
        return self.context.is_active

    def get_current_context(self):
        """Get current ripple context for consolidation"""
        if not self.context.is_active:
            return None

        # Create a simple context object with required attributes
        class SimpleContext:
            def __init__(self, gain, phase_peak, active):
                self.gain = gain
                self.phase_peak = phase_peak
                self.active = active

        return SimpleContext(
            gain=self.context.stdp_multiplier,
            phase_peak=self.context.phase,
            active=self.context.is_active
        )