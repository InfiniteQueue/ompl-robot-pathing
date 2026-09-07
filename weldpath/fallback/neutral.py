"""What counts as a neutral pose, and how to find the nearest one.

A neutral pose is a *set* of joint configurations rather than a single one: it fixes some
joints, constrains one to stay out of a band, and says nothing at all about the rest.  So
"the nearest neutral pose" to a given configuration is a projection onto that set, not a
lookup, and it is cheap -- each condition touches its own joint and nothing couples them,
so the nearest point is found joint by joint with no search.

Everything about the definition lives here, in ``CONDITIONS``.  Changing what neutral
means is meant to be an edit to that list and nothing else: the projection, the validity
test and the three search methods all read it rather than restating it.

Joint numbers are the robot's own, counting from 1, and are converted to indices at the
point of use.  Angles are in degrees here because that is how the conditions were
specified and how they will be re-specified; the planner works in radians and the
conversion happens at the boundary.

**Joint 3 is quoted against the floor.**  A linkage holds link 3 at a fixed angle to the
floor as joint 2 moves, so the robot's J3 register reads ``q3 - q2`` rather than the
URDF's relative rotation -- see :mod:`weldpath.profile`.  A condition on joint 3 is
therefore a condition on the register value, and the projection converts it back.  Getting
this the wrong way round is a 50-degree error that still looks plausible on paper, which is
why the coupling is applied here rather than left to the caller.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..profile import COUPLED_JOINT, COUPLING_SOURCE


@dataclass(frozen=True)
class Fixed:
    """Joint ``number`` sits at ``degrees``."""
    number: int
    degrees: float

    def describe(self) -> str:
        return f"J{self.number} = {self.degrees:g} deg"


@dataclass(frozen=True)
class Outside:
    """Joint ``number`` is anywhere except strictly between ``low`` and ``high``.

    Two answers are equally neutral -- the low edge and the high edge -- so the projection
    takes whichever the configuration is already nearer, which is what makes it the
    nearest neutral pose rather than merely a neutral one.
    """
    number: int
    low: float
    high: float

    def describe(self) -> str:
        return f"J{self.number} outside {self.low:g}..{self.high:g} deg"


# ---------------------------------------------------------------------------
# The definition.  Edit this, not the code below it.
# ---------------------------------------------------------------------------
CONDITIONS: tuple = (
    Fixed(2, -50.0),
    Fixed(3, +10.0),            # against the floor: the register value, i.e. q3 - q2
    Outside(5, -15.0, +15.0),
)

# Joints whose condition is quoted as a register value rather than a kinematic one.  Only
# joint 3 is coupled, and only to joint 2, but naming it this way keeps the fact in one
# place and out of the projection's arithmetic.
REGISTER_QUOTED = {COUPLED_JOINT: COUPLING_SOURCE}


def describe() -> str:
    """The definition in one line, for the run log."""
    return "; ".join(c.describe() for c in CONDITIONS)


def _kinematic(number: int, value_rad: float, q: np.ndarray) -> float:
    """A condition's target as a kinematic joint value.

    ``profile.commanded`` converts kinematic values to register values by subtracting the
    coupling source; this is that step inverted, so a condition quoted in register terms
    lands on the joint value the planner and the collision checker actually use.
    """
    source = REGISTER_QUOTED.get(number)
    if source is None:
        return value_rad
    return value_rad + float(q[source - 1])


def project(q: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """The neutral configuration nearest ``q``, clamped into the joint limits.

    Nearest in plain joint space rather than under the cell's TCP-travel weights: the
    conditions name individual joints, and each one is satisfied by moving that joint
    alone, so no weighting can change which value is chosen -- only how the cost of
    getting there would be reported.  A weighted metric would matter if the conditions
    ever traded one joint against another; none of them do.

    Order matters where a condition is register-quoted.  Joint 3's target depends on joint
    2, so joint 2 is placed first and joint 3 read off the result; ``CONDITIONS`` is
    walked in the order given and the coupling source is fixed before its dependant.
    """
    out = np.array(q, dtype=float)
    for cond in CONDITIONS:
        i = cond.number - 1
        if i >= len(out):
            continue                    # a shorter arm than the conditions describe
        if isinstance(cond, Fixed):
            out[i] = _kinematic(cond.number, np.deg2rad(cond.degrees), out)
        elif isinstance(cond, Outside):
            low = _kinematic(cond.number, np.deg2rad(cond.low), out)
            high = _kinematic(cond.number, np.deg2rad(cond.high), out)
            if low < out[i] < high:
                # Strictly inside the forbidden band: step to whichever edge is nearer.
                out[i] = low if (out[i] - low) <= (high - out[i]) else high
    return np.clip(out, lower, upper)


def satisfied(q: np.ndarray, tol_deg: float = 1e-6) -> bool:
    """Whether ``q`` is a neutral pose.

    Used to check the projection rather than to filter candidates -- nothing in the search
    produces a configuration that ought to be neutral without having been projected.  It
    is the one place the conditions are read forwards rather than backwards, so a
    projection that quietly disagrees with its own definition shows up here.
    """
    tol = np.deg2rad(tol_deg)
    for cond in CONDITIONS:
        i = cond.number - 1
        if i >= len(q):
            continue
        if isinstance(cond, Fixed):
            if abs(float(q[i]) - _kinematic(cond.number, np.deg2rad(cond.degrees), q)) > tol:
                return False
        elif isinstance(cond, Outside):
            low = _kinematic(cond.number, np.deg2rad(cond.low), q)
            high = _kinematic(cond.number, np.deg2rad(cond.high), q)
            if low + tol < float(q[i]) < high - tol:
                return False
    return True
