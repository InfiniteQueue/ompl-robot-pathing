"""Per-joint motion limits, and the time a move takes under them.

Each joint runs bang-bang: it accelerates at its limit until it either reaches its velocity
limit or has to start braking.  So the velocity profile is a **triangle** for a short move
and becomes a **trapezoid** only once the velocity limit is actually reached, at
``d = v^2 / a``::

    t = 2 * sqrt(d / a)        d <= v^2 / a     (triangular, never reaches v)
        d / v + v / a          d >  v^2 / a     (trapezoidal, cruises at v)

The two agree at the crossover, both giving ``2v/a``.

The consequence that matters for path quality is that **time is not linear in distance**.
Splitting one move into two costs an extra ``v/a`` when both halves still reach cruise, and
up to 41% more when they do not -- a triangular move of ``d`` takes ``2*sqrt(d/a)``, so two
moves of ``d/2`` take ``2*sqrt(2)`` times ``sqrt(d/a)`` against ``2*sqrt(d/a)`` for one.
Every waypoint is a full stop, and short hops are where that hurts most.

Joint 3 is coupled to joint 2
-----------------------------
A linkage holds link 3 at a fixed angle to the floor as joint 2 moves, so the number the
robot's J3 register carries is not the relative rotation the URDF models.  Measured against
this cell's own kinematics, link 3's elevation is a function of ``q3 - q2`` alone: it reads
the same at ``(q2, q3)`` of ``(-0.3, -0.3)``, ``(0, 0)`` and ``(0.3, 0.3)``, and rises from
-22.9 to +35.2 degrees as ``q3 - q2`` runs from -0.6 to +0.6 rad.  So the register reads
``q3 - q2``, increasing as the arm points up, and that is what :func:`commanded` converts to
for the output file.

Timing is a separate question.  The drive still moves joint 3 through the URDF's own relative
rotation, so its motion profile runs on ``q3`` unmodified, exactly like every other joint.
"""
from __future__ import annotations

import math

import numpy as np

# Defaults quoted per joint number: joint 6 is the fast wrist axis, everything else is
# the base figure.
DEFAULT_VELOCITY = 2.0 * math.pi / 3.0          # rad/s  (120 deg/s)
DEFAULT_VELOCITY_LAST = 11.0 * math.pi / 9.0    # rad/s  (220 deg/s)
DEFAULT_ACCELERATION = 2.5                      # rad/s^2
DEFAULT_ACCELERATION_LAST = 11.0                # rad/s^2

# The coupled pair, as joint numbers: joint 3's register value is measured against joint 2.
COUPLED_JOINT = 3
COUPLING_SOURCE = 2


def commanded(q: np.ndarray, coupled: bool = True) -> np.ndarray:
    """Kinematic joint values as the robot's own registers read them.

    Only joint 3 differs: its register is measured against the floor rather than against
    link 2, so it reads ``q3 - q2``.  See the module docstring for the measurement.  This is
    a presentation step for the output file -- planning, collision checking and timing all
    work in the kinematic values, which are what the URDF describes.
    """
    out = np.array(q, dtype=float)
    if coupled and len(out) >= COUPLED_JOINT:
        out[COUPLED_JOINT - 1] = out[COUPLED_JOINT - 1] - out[COUPLING_SOURCE - 1]
    return out


def default_velocity(n: int) -> list[float]:
    out = [DEFAULT_VELOCITY] * n
    if n >= 6:
        out[5] = DEFAULT_VELOCITY_LAST
    return out


def default_acceleration(n: int) -> list[float]:
    out = [DEFAULT_ACCELERATION] * n
    if n >= 6:
        out[5] = DEFAULT_ACCELERATION_LAST
    return out


def parse_limits(text: str | None, n: int, defaults: list[float], label: str) -> np.ndarray:
    """Read a comma-separated per-joint limit list.

    Accepts nothing (use the defaults), a single value applied to every joint, or exactly
    one value per joint in joint order.
    """
    if text is None or not str(text).strip():
        return np.array(defaults, dtype=float)
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    try:
        values = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"{label}: {exc}") from None
    if len(values) == 1:
        values = values * n
    if len(values) != n:
        raise ValueError(
            f"{label}: expected 1 or {n} comma-separated values, got {len(values)}")
    if any(v <= 0.0 for v in values):
        raise ValueError(f"{label}: every value must be positive")
    return np.array(values, dtype=float)


class JointDynamics:
    """Velocity and acceleration limits per joint, and the move times they imply."""

    def __init__(self, velocity, acceleration):
        self.velocity = np.asarray(velocity, dtype=float)
        self.acceleration = np.asarray(acceleration, dtype=float)
        if self.velocity.shape != self.acceleration.shape:
            raise ValueError("velocity and acceleration limits must cover the same joints")
        if np.any(self.velocity <= 0) or np.any(self.acceleration <= 0):
            raise ValueError("velocity and acceleration limits must be positive")

    def joint_times(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Time each joint needs to make this move on its own.

        Every joint, joint 3 included, is timed on the rotation the URDF gives it: the J2
        linkage changes what joint 3's register *reads*, not how far its drive turns.
        """
        d = np.abs(np.asarray(b, dtype=float) - np.asarray(a, dtype=float))
        v, acc = self.velocity, self.acceleration
        cruise = v * v / acc                     # distance at which the trapezoid starts
        triangular = 2.0 * np.sqrt(np.maximum(d, 0.0) / acc)
        trapezoid = d / v + v / acc
        return np.where(d <= cruise, triangular, trapezoid)

    def move_time(self, a: np.ndarray, b: np.ndarray) -> float:
        """Time for a coordinated move: the slowest joint governs, all starting together."""
        t = self.joint_times(a, b)
        return float(np.max(t)) if t.size else 0.0

    def slowest_joint(self, a: np.ndarray, b: np.ndarray) -> int:
        return int(np.argmax(self.joint_times(a, b)))

    def describe(self, names: list[str] | None = None) -> str:
        names = names or [f"j{i + 1}" for i in range(len(self.velocity))]
        parts = ", ".join(
            f"{n} {v:.3f} rad/s, {a:g} rad/s^2"
            for n, v, a in zip(names, self.velocity, self.acceleration))
        return f"joint limits: {parts}"
