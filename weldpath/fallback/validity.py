"""Whether a candidate point is somewhere the robot can actually be parked.

A fallback point is a tool pose.  Turning one into a fallback *pose* -- a configuration --
takes an inverse-kinematics solve, and which solution comes back is the whole question:
the same point is reachable elbow-up and elbow-down, and one of those may be buried in the
fixture while the other is in clear air.

The rule is deliberately strict.  The solution taken is the one nearest in joint space to
whichever end of the transit is nearest in Cartesian space, and *that* solution has to
survive both tests.  The alternative -- searching the branches for any solution that
passes -- would find more points, but the point of a fallback is that the robot can get to
it and back without contortion, and a solution the arm has to turn itself inside out to
reach is not that.  Nearest-to-the-nearest-end is what keeps the detour a detour.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Why a candidate was rejected.  Kept as plain strings rather than an enum: they are only
# ever logged, and the log is the audience.
NO_IK = "no inverse-kinematics solution"
IN_COLLISION = "the nearest solution is in collision"
TOO_CLOSE = "the nearest solution comes within the fallback distance"
OK = ""


@dataclass
class Verdict:
    """The outcome of testing one candidate point."""
    q: np.ndarray | None
    reason: str
    clearance_mm: float = float("nan")

    @property
    def valid(self) -> bool:
        return self.q is not None and self.reason is OK


class Validator:
    """Tests candidate points against one transit's two ends.

    Constructed per transit because both ends enter the test: which one seeds the solve
    depends on where the candidate is, so a validator built for one transit says nothing
    useful about another.
    """

    def __init__(self, cell, qa: np.ndarray, qb: np.ndarray, fallback_mm: float):
        self.cell = cell
        self.qa = np.asarray(qa, dtype=float)
        self.qb = np.asarray(qb, dtype=float)
        self.fallback_mm = float(fallback_mm)
        # Tool positions of the two ends, in metres, for the "nearest in Cartesian space"
        # comparison.  Measured once: the ends do not move while the search runs.
        self.pa = cell.fk(self.qa)[:3, 3]
        self.pb = cell.fk(self.qb)[:3, 3]
        # The clearance query answers nothing it cannot see.  Asking for exactly the
        # fallback distance is enough: a reading saturates at the probe when nothing is
        # within it, and a saturated reading is precisely the case that passes.
        cell.require_proximity(self.fallback_mm)

    def seed_for(self, position_world: np.ndarray) -> np.ndarray:
        """The end configuration to solve from, for a candidate at ``position_world`` (m)."""
        p = np.asarray(position_world, dtype=float)
        return self.qa if (np.linalg.norm(p - self.pa)
                           <= np.linalg.norm(p - self.pb)) else self.qb

    def check(self, pose_world_mm: np.ndarray) -> Verdict:
        """Test one candidate tool pose, given in manifest units."""
        T = np.asarray(pose_world_mm, dtype=float)
        seed = self.seed_for(T[:3, 3] * self.cell.man.scale)
        # require_collision_free=False on purpose: the rule is that the *nearest* solution
        # is clear, not that some clear solution exists.  Letting the solver skip past a
        # blocked branch would answer the weaker question and quietly widen the rule.
        q = self.cell.solve_pose(T, [seed], require_collision_free=False)
        if q is None:
            return Verdict(None, NO_IK)
        if self.cell.in_collision(q):
            return Verdict(None, IN_COLLISION)
        got = self.cell.clearance_mm(q)
        if got < self.fallback_mm:
            return Verdict(None, TOO_CLOSE, got)
        return Verdict(q, OK, got)
