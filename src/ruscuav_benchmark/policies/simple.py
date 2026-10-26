"""Simple non-learned policies for tests and sanity checks.

They act on the observation alone, like a trained policy, and work with every method
preset: the geometry part of the observation, which every preset starts with, holds
the direction to the target (entries 6-7) and, after the ``n_rays`` LiDAR readings,
the normalised velocity.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = ["TargetSeekingPolicy", "ConstantPolicy"]


class TargetSeekingPolicy:
    """Accelerates towards the target and damps the velocity:
    ``clip(gain * direction_to_target - damping * velocity, -1, 1)``. It ignores the
    LiDAR and the semantic layers, so it flies straight lines; the shields keep it out
    of buildings."""

    def __init__(self, gain: float = 1.0, damping: float = 0.5, n_rays: int = 64) -> None:
        self.gain = float(gain)
        self.damping = float(damping)
        self.n_rays = int(n_rays)

    def predict(self, obs: np.ndarray, deterministic: bool = True, **_: object) -> Tuple[np.ndarray, Optional[object]]:
        obs = np.asarray(obs, dtype=np.float64)
        direction = obs[6:8]
        velocity = obs[11 + self.n_rays: 13 + self.n_rays]
        action = np.clip(self.gain * direction - self.damping * velocity, -1.0, 1.0)
        return action.astype(np.float32), None


class ConstantPolicy:
    """Always the same action."""

    def __init__(self, action: Sequence[float]) -> None:
        self.action = np.asarray(action, dtype=np.float32).reshape(2)

    def predict(self, obs: np.ndarray, deterministic: bool = True, **_: object) -> Tuple[np.ndarray, Optional[object]]:
        return self.action.copy(), None
