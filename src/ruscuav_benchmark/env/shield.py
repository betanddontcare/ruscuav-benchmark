"""Urban safety shields.

Action shield
    Predicts the next horizontal position for the commanded acceleration. If it
    is unsafe, the command is replaced by the best safe candidate among the
    command, its two perpendiculars, a push along the clearance gradient and zero,
    scored by distance gained to the target, clearance gained and risk.
Geometric shield
    Samples the motion segment of the step. If it enters a hazard, the position
    stops ``backoff_m`` before the first unsafe sample (or stays at the previous
    position if that is still unsafe).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from .dynamics import PointMassDynamics
from .hazards import UrbanHazardModel


class UrbanShield:
    def __init__(self, hazards: UrbanHazardModel, dynamics: PointMassDynamics, action_shield: bool,
                 geometric_shield: bool, segment_check_step_m: float, backoff_m: float) -> None:
        self.hazards = hazards
        self.dynamics = dynamics
        self.action_shield = bool(action_shield)
        self.geometric_shield = bool(geometric_shield)
        self.segment_check_step_m = float(segment_check_step_m)
        self.backoff_m = float(max(0.0, backoff_m))
        self.a_max_xy = dynamics.config.a_max_xy
        self.bounds_low = hazards.bounds_low
        self.bounds_high = hazards.bounds_high

    # ------------------------------------------------------------------ action shield
    def _predicted_pos(self, prev_pos, prev_vel, acc_xy) -> np.ndarray:
        p_xy = self.dynamics.predict_next_pos_xy(prev_pos, prev_vel, acc_xy.astype(np.float32))
        return np.array([p_xy[0], p_xy[1], prev_pos[2]], dtype=np.float32)

    def filter_action(self, prev_pos, prev_vel, desired_acc_xy,
                      dist_to_target: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, Optional[str]]:
        """Return (acceleration to apply, reason if the command was replaced)."""
        acc0 = desired_acc_xy.astype(np.float32)
        if not self.action_shield:
            return acc0, None
        hazards = self.hazards
        pred0 = self._predicted_pos(prev_pos, prev_vel, acc0)
        if not hazards.is_unsafe(pred0, include_clearance_violation=True):
            return acc0, None

        ax, ay = float(acc0[0]), float(acc0[1])
        a_left = np.array([-ay, ax], dtype=np.float32)
        a_right = np.array([ay, -ax], dtype=np.float32)
        grad = hazards.clearance_gradient_xy(prev_pos)
        acc_mag = float(np.linalg.norm(acc0) + 1e-6)
        a_away = grad * acc_mag
        candidates = [acc0, a_left, a_right, a_away, np.zeros(2, dtype=np.float32)]

        prev_dist = float(dist_to_target(prev_pos))
        prev_clear = float(hazards.signed_clearance_m(prev_pos))
        risk_model = hazards.risk_model

        def score(acc_xy):
            acc_xy = np.clip(acc_xy, -self.a_max_xy, self.a_max_xy).astype(np.float32)
            pred = self._predicted_pos(prev_pos, prev_vel, acc_xy)
            if hazards.is_unsafe(pred, include_clearance_violation=True):
                return -1e12
            dist_gain = prev_dist - float(dist_to_target(pred))
            clear_gain = float(hazards.signed_clearance_m(pred) - prev_clear)
            risk = float(risk_model.risk_at(pred[0], pred[1], pred[2]))
            return 1.00 * dist_gain + 0.35 * clear_gain - 0.02 * risk

        best = max(candidates, key=score)
        best_score = score(best)
        if best_score < -1e11:
            return np.zeros(2, dtype=np.float32), "urban_action_no_feasible_candidate"
        reason = None if np.allclose(best, acc0, atol=1e-6) else "urban_action_candidate_replaced"
        return np.clip(best, -self.a_max_xy, self.a_max_xy).astype(np.float32), reason

    # ------------------------------------------------------------------ geometric shield
    def first_hit_on_segment(self, A, B) -> Optional[float]:
        """Fraction t in (0, 1] of the first unsafe sample on segment AB, or None."""
        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)
        d = B - A
        seg_len = float(np.linalg.norm(d) + 1e-9)
        if seg_len < 1e-9:
            if self.hazards.is_unsafe(B, include_clearance_violation=True):
                return 0.0
            return None
        n = max(2, int(np.ceil(seg_len / self.segment_check_step_m)))
        for k in range(1, n + 1):
            t = float(k / n)
            P = A + t * d
            if self.hazards.is_unsafe(P, include_clearance_violation=True):
                return t
        return None

    def project_motion(self, prev_pos, desired_pos) -> Tuple[np.ndarray, bool]:
        """Return (accepted position, whether the shield changed it)."""
        if not self.geometric_shield:
            return desired_pos.astype(np.float32), False
        A = np.asarray(prev_pos, dtype=np.float32)
        B = np.asarray(desired_pos, dtype=np.float32)
        # Already unsafe: do not project; the termination logic handles it.
        if self.hazards.is_unsafe(A, include_clearance_violation=True):
            return B.astype(np.float32), False
        hit_t = self.first_hit_on_segment(A, B)
        if hit_t is None:
            return B.astype(np.float32), False
        seg_len = float(np.linalg.norm(B - A) + 1e-9)
        backoff_t = min(hit_t, self.backoff_m / seg_len)
        safe_t = max(0.0, hit_t - backoff_t)
        safe_pos = A + safe_t * (B - A)
        safe_pos = np.clip(safe_pos, self.bounds_low, self.bounds_high).astype(np.float32)
        # If the sampling still leaves the position unsafe, stay at the previous one.
        if self.hazards.is_unsafe(safe_pos, include_clearance_violation=True):
            safe_pos = A.copy().astype(np.float32)
        return safe_pos.astype(np.float32), True
