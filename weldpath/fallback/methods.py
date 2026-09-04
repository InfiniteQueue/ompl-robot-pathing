"""Where to look for a fallback point, in the order the methods are tried.

Each method is a generator of candidate tool poses, most promising first.  It decides
*where* to look and nothing else: whether a candidate is any good is
:mod:`weldpath.fallback.validity`'s question, and a method that answered it too would
have to be rewritten every time the rule changed.

Adding a method means writing a generator and putting it in ``METHODS``.  Nothing else in
the package needs to know it exists -- the finder walks the list, times each one and logs
what it found, and the order of the list is the order of preference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from ..planning import interpolate_pose
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

    def tcp_pose_mm(self, q: np.ndarray) -> np.ndarray:
        """The tool pose at ``q`` as a 4x4 in manifest units."""
        T = np.array(self.cell.fk(q), dtype=float)
        T[:3, 3] /= self.man.scale
        return T


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


def retreat_from_start(ctx: Context) -> Iterator[np.ndarray]:
    """Back the start locator out along the axis that points into the gun's bulk."""
    return _walk_axis(ctx, ctx.pose_a, ctx.qa)


def retreat_from_end(ctx: Context) -> Iterator[np.ndarray]:
    """The same, from the far end of the transit."""
    return _walk_axis(ctx, ctx.pose_b, ctx.qb)


def nearest_neutral(ctx: Context) -> np.ndarray:
    """The neutral configuration nearest either end of the transit.

    Both ends are projected and the closer projection wins -- "nearest" is a property of
    the transit, and taking the start end by convention would ignore that one end is
    often already half way to neutral while the other is buried in the fixture.
    """
    cell = ctx.cell
    best, best_cost = None, np.inf
    for q in (ctx.qa, ctx.qb):
        candidate = neutral.project(q, cell.lower, cell.upper)
        cost = cell.distance(q, candidate)
        if cost < best_cost:
            best, best_cost = candidate, cost
    return best


def midpoint_towards_neutral(ctx: Context) -> Iterator[np.ndarray]:
    """From half way between the two ends, walk towards where neutral puts the tool.

    The orientation is the one half way between the two ends and it is held for the whole
    walk.  What is being searched is a line of *positions* -- that is what the method was
    specified as -- and turning the tool as it goes would search a different curve at
    every step, which is a harder thing to reason about when it fails.
    """
    mid = interpolate_pose(np.asarray(ctx.pose_a, dtype=float),
                           np.asarray(ctx.pose_b, dtype=float), 0.5)
    target = ctx.tcp_pose_mm(nearest_neutral(ctx))[:3, 3]
    span = target - mid[:3, 3]
    dist = float(np.linalg.norm(span))
    if dist < 1e-9:
        return
    # Not capped at MAX_STEPS: this walk has an end of its own.  The cap exists for the
    # retreat walks, which head off in a direction and would otherwise go forever; here
    # the number of steps is fixed by how far apart the two points are, and truncating it
    # would drop the far end of the line -- which is the neutral pose's own tool position,
    # the most promising candidate on it.
    n = int(np.ceil(dist / max(ctx.step_mm, 1e-9)))
    direction = span / dist
    for k in range(1, n + 1):
        out = np.array(mid, dtype=float)
        # Clamped to the target so the last step lands on it rather than past it: the
        # neutral pose's own tool position is the most promising candidate on this line
        # and stepping over it would be the one place the walk could miss.
        out[:3, 3] = mid[:3, 3] + direction * min(k * ctx.step_mm, dist)
        yield out


@dataclass(frozen=True)
class Method:
    name: str
    generate: Callable[[Context], Iterator[np.ndarray]]


# The order of preference.  Append to extend it.
METHODS: tuple[Method, ...] = (
    Method("retreat from the start locator", retreat_from_start),
    Method("retreat from the end locator", retreat_from_end),
    Method("midpoint towards the nearest neutral pose", midpoint_towards_neutral),
)
