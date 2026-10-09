"""Where to look for one end's fallback point, and in what order.

**A fallback belongs to one end of the transit.**  The detour is
``w1 -> f1 -> f2 -> w2``, so ``f1`` is the start locator's own fallback and ``f2`` is the
end locator's: each is a place to stand just clear of the pocket that end sits in, and the
leg joining a locator to its own fallback is the extraction from that pocket.  That is why
the methods here are paired per end rather than held in one list of preference.  The older
arrangement walked three methods -- retreat from the start, retreat from the end, then a
line out of the midpoint -- and handed the planner whichever poses they produced as
interchangeable candidates for one via, which made "retreat from the end locator" a
fallback for the *start* locator's failure.  It never was: those answer two different
questions.

Each end has two methods, tried in order:

  ``primary``    step out along the locator's own retreat axis -- the direction that
                 points into the gun's bulk, which is the way out of the pocket.
  ``secondary``  step from the locator towards where the nearest neutral pose puts the
                 tool.  This is the walk to use when the retreat axis is blocked, or when
                 the locator has no usable axis at all, and it heads somewhere the arm is
                 known to be able to stand rather than in a direction read off the gun.

Both walk outward from the locator a step at a time, so the first candidate that passes
:mod:`weldpath.fallback.validity` is the nearest one that works -- which is what keeps a
detour a detour.  A method decides *where* to look and nothing else; whether a candidate
is any good is validity's question, and a method that answered it too would have to be
rewritten every time the rule changed.

Adding a method means writing a generator and putting it in an ``End``.  Nothing else in
the package needs to know it exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from . import neutral, retreat

# A backstop, not a working limit.  What normally ends a retreat walk is running out of
# arm: the finder stops as soon as a step has no inverse-kinematics solution, which is the
# robot itself saying the direction is exhausted and is a far better answer than any step
# count guessed in advance.  This only catches the case where that never happens -- a
# direction that stays reachable forever, which a redundant or unbounded joint can produce
# -- so it is set high enough never to bind on a real arm.
MAX_STEPS = 2000


@dataclass
class Context:
    """Everything a method needs to know about the transit it is finding a pose for."""
    cell: object
    man: object
    qa: np.ndarray                  # configuration at the start locator
    qb: np.ndarray                  # configuration at the end locator
    pose_a: np.ndarray              # start locator pose, 4x4, manifest units
    pose_b: np.ndarray              # end locator pose, 4x4, manifest units
    step_mm: float


def _walk_axis(ctx: Context, pose: np.ndarray, q_at: np.ndarray) -> Iterator[np.ndarray]:
    """Candidates stepping away from ``pose`` along its retreat axis."""
    found = retreat.retreat_axis(ctx.cell, ctx.man, pose, q_at)
    if found is None:
        return
    axis_world, _label = found
    # The axis is a unit direction, so it carries no units of its own and the step is
    # simply ``step_mm`` of it.  World and manifest frames share an orientation and differ
    # only in scale, which is why a direction found in metres can be applied to a position
    # held in millimetres without converting anything.
    step = ctx.step_mm * axis_world
    for k in range(1, MAX_STEPS + 1):
        out = np.array(pose, dtype=float)
        out[:3, 3] = out[:3, 3] + k * step
        yield out


def _walk_towards(ctx: Context, pose: np.ndarray,
                  q_at: np.ndarray) -> Iterator[np.ndarray]:
    """Candidates stepping from ``pose`` towards where the nearest neutral pose stands.

    The orientation is the locator's own and is held for the whole walk.  What is being
    searched is a line of *positions*; turning the tool as it goes would search a
    different curve at every step, which is a harder thing to reason about when it fails.
    Holding the locator's orientation rather than the neutral pose's also keeps the first
    few candidates close to a pose the arm demonstrably reaches -- the locator itself --
    which is where the useful answers are.

    Each end projects *its own* configuration onto the neutral set, so the two ends walk
    towards their own nearest neutral pose rather than sharing one.  Both the start of the
    line and its direction are then per end, which is the point of pairing the methods
    this way: two ends sharing one line would often return the same pose for both
    fallbacks, and there would be nothing left for the middle leg to join.
    """
    target = ctx.cell.pose_mm(
        neutral.project(np.asarray(q_at, dtype=float),
                        ctx.cell.lower, ctx.cell.upper))[:3, 3]
    start = np.asarray(pose, dtype=float)[:3, 3]
    span = target - start
    dist = float(np.linalg.norm(span))
    if dist < 1e-9:
        return                      # already standing where neutral puts the tool
    # Not capped at MAX_STEPS: this walk has an end of its own.  The cap exists for the
    # retreat walks, which head off in a direction and would otherwise go forever; here
    # the number of steps is fixed by how far the neutral pose is, and truncating it would
    # drop the far end of the line -- which is the neutral pose's own tool position.
    n = int(np.ceil(dist / max(ctx.step_mm, 1e-9)))
    direction = span / dist
    for k in range(1, n + 1):
        out = np.array(pose, dtype=float)
        # Clamped to the target so the last step lands on it rather than past it: the
        # neutral pose's own tool position is the most promising candidate on this line
        # and stepping over it would be the one place the walk could miss.
        out[:3, 3] = start + direction * min(k * ctx.step_mm, dist)
        yield out


def retreat_from_start(ctx: Context) -> Iterator[np.ndarray]:
    """Back the start locator out along the axis that points into the gun's bulk."""
    return _walk_axis(ctx, ctx.pose_a, ctx.qa)


def retreat_from_end(ctx: Context) -> Iterator[np.ndarray]:
    """The same, from the far end of the transit."""
    return _walk_axis(ctx, ctx.pose_b, ctx.qb)


def neutral_from_start(ctx: Context) -> Iterator[np.ndarray]:
    """Out of the start locator towards its own nearest neutral pose."""
    return _walk_towards(ctx, ctx.pose_a, ctx.qa)


def neutral_from_end(ctx: Context) -> Iterator[np.ndarray]:
    """Out of the end locator towards its own nearest neutral pose."""
    return _walk_towards(ctx, ctx.pose_b, ctx.qb)


@dataclass(frozen=True)
class Method:
    name: str
    generate: Callable[[Context], Iterator[np.ndarray]]


@dataclass(frozen=True)
class End:
    """One end of the transit, and the two places its fallback is looked for.

    ``name`` is how the end is spoken of in the log and in a failure message, and it is
    also the label a gun opening is screened against -- an opening has to leave every pose
    the detour is pinned to clear, and a fallback pose is one of those poses.
    """
    name: str
    primary: Method
    secondary: Method


# The two ends, in route order: the start locator's fallback comes first in the detour.
ENDS: tuple[End, ...] = (
    End("start",
        Method("retreat from the start locator", retreat_from_start),
        Method("the start locator towards its nearest neutral pose", neutral_from_start)),
    End("end",
        Method("retreat from the end locator", retreat_from_end),
        Method("the end locator towards its nearest neutral pose", neutral_from_end)),
)
