"""Cartesian-space tree search: a route whose every edge is a straight tool move.

OMPL cannot do this.  Its only move profile is ``OMPLRealVectorMoveProfile``, whose state
space is the joint vector, so every edge it produces is a joint-space straight line and no
setting changes that.  The bindings expose no hook for a custom motion validator either,
so a linear-profile tree has to be its own planner rather than a configuration of theirs.

What is *not* different is the state space.  A tool pose does not determine an arm
configuration -- one pose has several inverse kinematics branches, and an edge that is
clear on one is blocked on another.  A tree whose nodes were poses would therefore keep
finding routes that no single configuration can execute, and joining two of its edges
could demand a wrist flip in the middle of a move the controller drives as a straight
line.  So the nodes here are joint states, exactly as OMPL's are.  Only the **edge**
changes: the joint chord becomes the Cartesian line, checked along the curve the tool
actually follows.

The route that comes out is a polyline of straight moves, and a polyline is not itself
straight.  That is the point of the search -- the shape is free, sampled rather than
assumed, and nothing here presumes a detour is an offset from the chord, an arc, or any
other family a parametric search would have to be told about in advance.

Two things this module holds to, both learned the hard way elsewhere in the planner:

* **Edges carry their own states.**  ``plan_linear`` seeds each station from the previous
  solution and takes what comes back nearest, so it is greedy and its answer depends on
  where it started.  Asking it the same question twice can give a different chain, or no
  chain.  A tree that stored only endpoints would hand on routes that re-derive
  differently downstream and get thrown out by a verifier that is right to throw them out.
  Every edge here keeps the states it was validated on, and those are the states that
  ship.
* **No branch changes inside an edge.**  A station is refused if it lands further than
  ``jump_rad`` from the one before it on any joint, whatever the inverse kinematics says.
  Scattered seeds are still tried, so other configurations remain reachable, but only by
  starting a new edge from a node that already sits in one -- never by flipping halfway
  along a line the robot will drive without stopping.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .cell import Cell
from .planning import PlanningError, interpolate_pose

__all__ = ["CartesianBudget", "plan_cartesian"]


@dataclass
class CartesianBudget:
    """How hard the tree searches, and how far it reaches at a time.

    ``extend_mm`` is the one setting worth understanding before the others.  It is how far
    a single extend drives the tool, and it trades the two failure modes against each
    other: short extends explore reliably but grow the tree slowly and pay a
    nearest-neighbour scan per millimetre of progress, while long ones cover ground but
    are refused whole the moment any station on them fails, wasting the inverse kinematics
    already spent on the stations before it.
    """
    seconds: float = 20.0           # wall clock for one solve
    max_iters: int = 4000           # sample budget, whichever runs out first
    extend_mm: float = 120.0        # furthest one extend drives the tool
    extend_deg: float = 25.0        # ...and furthest it turns the tool, whichever binds
    margin_mm: float = 150.0        # how far outside the endpoints sampling may stray
    tilt_deg: float = 30.0          # how far off the endpoint slerp an orientation is drawn
    goal_bias: float = 0.10         # share of samples aimed at the far tree's root
    jump_rad: float = 0.50          # largest per-joint step allowed between two stations
    orient_weight_mm_per_rad: float = 200.0     # a radian of turn as this much tool travel
    branch_seeds: int = 8           # scattered seeds tried when the continuing one fails
    seed: int = 0

    @property
    def enabled(self) -> bool:
        return self.seconds > 0.0 and self.max_iters > 0


# ---------------------------------------------------------------------------
# pose helpers
# ---------------------------------------------------------------------------
def _rot_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Shortest rotation angle, in radians, taking ``Ra`` onto ``Rb``."""
    R = Ra.T @ Rb
    return float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))


def _random_rotation(rng: np.random.Generator, max_angle: float) -> np.ndarray:
    """A rotation of at most ``max_angle`` about a uniformly random axis."""
    axis = rng.normal(size=3)
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        return np.eye(3)
    axis /= n
    th = float(rng.uniform(-max_angle, max_angle))
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def pose_mm(cell: Cell, q: np.ndarray) -> np.ndarray:
    """TCP pose in manifest units.

    ``Cell.fk`` answers in environment units -- metres -- while the inverse kinematics
    behind ``solve_pose`` takes manifest units and applies the scale itself.  Handing the
    raw transform over asks for a pose a millimetre from the base, which has no solution,
    so every line would read as unreachable.  Same conversion as ``MotionModel._pose_mm``.
    """
    T = cell.fk(q).copy()
    T[:3, 3] /= cell.man.scale
    return T


# ---------------------------------------------------------------------------
# the tree
# ---------------------------------------------------------------------------
@dataclass
class _Node:
    q: np.ndarray
    pose: np.ndarray                # manifest units
    parent: int                     # -1 at the root
    chain: list[np.ndarray]         # states from the parent to here, parent's q pinned first


class _Tree:
    """Nodes and the edges that reached them, with a nearest-neighbour scan over poses.

    Distance is tool travel plus turn, the turn converted to millimetres by
    ``orient_weight_mm_per_rad`` so the two are commensurate.  The rotation part is scored
    by the Frobenius distance between the rotation matrices rather than by the angle
    itself: the two are monotonically related -- ``|Ra - Rb|_F = 2 sqrt(2) sin(theta/2)``
    -- and the Frobenius form is one vectorised subtraction over the whole tree, where the
    angle needs a trace and an arccos per node.  Nearest is a ranking, so a monotone
    substitute for the metric ranks identically.
    """

    def __init__(self, root_q: np.ndarray, root_pose: np.ndarray, w_mm_per_rad: float):
        self.nodes: list[_Node] = [_Node(np.asarray(root_q, dtype=float),
                                         np.asarray(root_pose, dtype=float), -1, [])]
        self.w = w_mm_per_rad / np.sqrt(2.0)     # see the note on the Frobenius form
        self._pos = np.empty((16, 3), dtype=float)
        self._rot = np.empty((16, 9), dtype=float)
        self._n = 0
        self._push(root_pose)

    def _push(self, pose: np.ndarray) -> None:
        if self._n == len(self._pos):
            self._pos = np.vstack([self._pos, np.empty_like(self._pos)])
            self._rot = np.vstack([self._rot, np.empty_like(self._rot)])
        self._pos[self._n] = pose[:3, 3]
        self._rot[self._n] = pose[:3, :3].reshape(9)
        self._n += 1

    def add(self, q: np.ndarray, pose: np.ndarray, parent: int,
            chain: list[np.ndarray]) -> int:
        self.nodes.append(_Node(np.asarray(q, dtype=float),
                                np.asarray(pose, dtype=float), parent, chain))
        self._push(pose)
        return len(self.nodes) - 1

    def nearest(self, pose: np.ndarray) -> int:
        dp = np.linalg.norm(self._pos[:self._n] - pose[:3, 3], axis=1)
        dr = np.linalg.norm(self._rot[:self._n] - pose[:3, :3].reshape(9), axis=1)
        return int(np.argmin(dp + self.w * dr))

    def route_from_root(self, index: int) -> list[np.ndarray]:
        """Every state from this tree's root down to ``index``, in that order."""
        edges: list[list[np.ndarray]] = []
        i = index
        while self.nodes[i].parent >= 0:
            edges.append(self.nodes[i].chain)
            i = self.nodes[i].parent
        out = [self.nodes[i].q]
        for chain in reversed(edges):
            out.extend(chain[1:])           # chain[0] is the state already emitted
        return out


# ---------------------------------------------------------------------------
# the extend
# ---------------------------------------------------------------------------
def _steer(cell: Cell, step_mm: float, max_step: float, budget: CartesianBudget,
           q_from: np.ndarray, pose_from: np.ndarray, pose_to: np.ndarray, *,
           reach_mm: float, rng: np.random.Generator
           ) -> tuple[list[np.ndarray], np.ndarray] | None:
    """Drive the tool along the straight line towards ``pose_to``, as far as it will go.

    Returns the states it reached and the pose it stopped at, or ``None`` when it could
    not leave ``q_from`` at all.  Stopping early is a result, not a failure: the prefix is
    collision free and reachable, so it is a real place for the tree to stand, and
    throwing it away because the rest of the line was blocked is what makes a
    single-shot linear check useless as a search primitive.

    ``plan_linear`` is not reused here for exactly that reason -- it raises on the first
    station it cannot solve, discarding the ones before it -- and because seeding policy
    matters at this level.  The continuing configuration is tried first and alone; only if
    it fails are scattered seeds brought in, and any answer landing further than
    ``jump_rad`` from the previous station on any joint is refused however it was found.
    """
    pa = np.asarray(pose_from, dtype=float)
    pb = np.asarray(pose_to, dtype=float)
    dist = float(np.linalg.norm(pb[:3, 3] - pa[:3, 3]))
    turn = _rot_angle(pa[:3, :3], pb[:3, :3])

    # How much of the line this extend takes: whichever of travel and turn binds first.
    t = 1.0
    if dist > 1e-9:
        t = min(t, reach_mm / dist)
    if turn > 1e-9:
        t = min(t, np.radians(budget.extend_deg) / turn)
    if t <= 1e-9:
        return None

    span = dist * t
    n = max(1, int(np.ceil(span / max(step_mm, 1e-6))))

    chain = [np.asarray(q_from, dtype=float)]
    current = chain[0]
    reached = pa
    for k in range(1, n + 1):
        pose = interpolate_pose(pa, pb, t * k / n)
        q = cell.solve_pose(pose, [current], branch_seeds=0)
        if q is None and budget.branch_seeds:
            q = cell.solve_pose(pose, [current], branch_seeds=budget.branch_seeds, rng=rng)
        if q is None:
            break                           # no collision-free inverse kinematics here
        if float(np.max(np.abs(q - current))) > budget.jump_rad:
            break                           # a branch flip; not executable inside one move
        if cell.segment_collides(current, q, max_step=max_step):
            break                           # clear at both stations, blocked between them
        chain.append(q)
        current = q
        reached = pose

    if len(chain) < 2:
        return None
    return chain, reached


# ---------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------
def _sample(rng: np.random.Generator, pa: np.ndarray, pb: np.ndarray,
            budget: CartesianBudget) -> np.ndarray:
    """A pose to grow towards: a point near the endpoints, turned a little off their slerp.

    Neither half is uniform over what the robot could reach, and both bounds are the
    search's own admission of what it expects.  Position is drawn from the box the two
    endpoints span, inflated by ``margin_mm`` -- a detour has to leave the chord to be
    worth finding, but one that leaves it by more than the fixture is deep is not a detour.
    Orientation is drawn near the interpolation between the two endpoint orientations and
    perturbed by at most ``tilt_deg``, because a tool that arrives at a weld square to the
    panel is square to it most of the way in, and sampling SO(3) freely would spend nearly
    the whole budget on orientations no route uses.
    """
    lo = np.minimum(pa[:3, 3], pb[:3, 3]) - budget.margin_mm
    hi = np.maximum(pa[:3, 3], pb[:3, 3]) + budget.margin_mm
    out = interpolate_pose(pa, pb, float(rng.uniform(0.0, 1.0)))
    out[:3, 3] = rng.uniform(lo, hi)
    out[:3, :3] = out[:3, :3] @ _random_rotation(rng, np.radians(budget.tilt_deg))
    return out


def plan_cartesian(cell: Cell, qa: np.ndarray, qb: np.ndarray, *,
                   max_step: float,
                   budget: CartesianBudget | None = None, log=None) -> list[np.ndarray]:
    """A route from ``qa`` to ``qb`` whose every move is a straight line of the tool.

    Bidirectional, because both ends are given and growing from one of them alone throws
    away half of what is known.  The trees swap each iteration, so whichever is having the
    easier time of it keeps being the one that explores.

    Stations along each edge are spaced by ``Cell.tcp_check_mm`` -- the same tool-space
    step the joint-space check subdivides on, read from the cell rather than passed in so
    that there is one number and no way for a caller to hand this a coarser one.

    The states come back dense -- every station the search validated -- and consecutive
    states are pairwise checked along the Cartesian line between them.  Reducing that to
    the few points a controller needs is the reduction passes' job, not this one's, and
    they already work under the linear profile.
    """
    budget = budget or CartesianBudget()
    rng = np.random.default_rng(budget.seed)
    deadline = time.perf_counter() + budget.seconds
    step_mm = float(getattr(cell, "tcp_check_mm", 0.0)) or float("inf")

    pa, pb = pose_mm(cell, qa), pose_mm(cell, qb)
    if cell.in_collision(qa) or cell.in_collision(qb):
        raise PlanningError("cartesian tree: an endpoint is already in collision")

    start = _Tree(qa, pa, budget.orient_weight_mm_per_rad)
    goal = _Tree(qb, pb, budget.orient_weight_mm_per_rad)
    a, b = start, goal
    forward = True                          # whether `a` is still the start tree

    iters = 0
    while iters < budget.max_iters and time.perf_counter() < deadline:
        iters += 1
        target = (b.nodes[0].pose if rng.uniform() < budget.goal_bias
                  else _sample(rng, pa, pb, budget))

        i = a.nearest(target)
        grown = _steer(cell, step_mm, max_step, budget, a.nodes[i].q, a.nodes[i].pose,
                       target, reach_mm=budget.extend_mm, rng=rng)
        if grown is None:
            a, b, forward = b, a, not forward
            continue
        chain, pose = grown
        new = a.add(chain[-1], pose, i, chain)

        # Reach for the other tree from where we just landed, repeatedly -- this is the
        # "connect" half, and it is what makes the two trees meet in a handful of extends
        # rather than waiting for a sample to fall between them.
        j = b.nearest(pose)
        cursor, cursor_pose, cursor_index = a.nodes[new].q, pose, new
        while time.perf_counter() < deadline:
            step = _steer(cell, step_mm, max_step, budget, cursor, cursor_pose,
                          b.nodes[j].pose, reach_mm=budget.extend_mm, rng=rng)
            if step is None:
                break
            link, cursor_pose = step
            cursor_index = a.add(link[-1], cursor_pose, cursor_index, link)
            cursor = link[-1]
            gap = float(np.linalg.norm(cursor_pose[:3, 3] - b.nodes[j].pose[:3, 3]))
            turn = _rot_angle(cursor_pose[:3, :3], b.nodes[j].pose[:3, :3])
            if gap > 1e-6 or turn > 1e-9:
                continue
            # The two trees are at the same pose.  They are not yet at the same joint
            # state: `link` ended wherever the inverse kinematics landed, and the far
            # tree's node is its own solution for that pose.  Closing that gap is a joint
            # move at a standstill, so it is checked as one.
            far = b.nodes[j].q
            if float(np.max(np.abs(cursor - far))) > budget.jump_rad:
                break                       # met in space, on different branches
            if cell.segment_collides(cursor, far, max_step=max_step):
                break
            route = _join(a, cursor_index, b, j, forward)
            if log:
                log(f"      cartesian tree: joined after {iters} samples, "
                    f"{len(start.nodes) + len(goal.nodes)} nodes, {len(route)} states")
            return _checked(cell, route, max_step, qa, qb)

        a, b, forward = b, a, not forward

    raise PlanningError(
        f"cartesian tree: no straight-line route after {iters} samples and "
        f"{budget.seconds:g}s ({len(start.nodes)} and {len(goal.nodes)} nodes)")


def _join(a: _Tree, ia: int, b: _Tree, ib: int, forward: bool) -> list[np.ndarray]:
    """The one state list the two trees describe, in start-to-goal order.

    An edge is stored in the direction it was grown, and the far tree grew backwards from
    the goal, so its half is reversed on the way out.  Reversing is sound here in a way it
    would not be for a joint-space edge that had only been sampled: the line from b to a
    is the same line as from a to b, and the states are kept rather than re-solved, so
    what ships is what was checked.
    """
    left = a.route_from_root(ia)
    right = b.route_from_root(ib)
    right.reverse()
    joined = left + right
    return joined if forward else list(reversed(joined))


def _checked(cell: Cell, route: list[np.ndarray], max_step: float,
             qa: np.ndarray, qb: np.ndarray) -> list[np.ndarray]:
    """The assembled route, with the assembly itself checked rather than assumed.

    Every pair here was validated as it was grown, so this can only fail on a bookkeeping
    mistake -- an edge stitched in the wrong order, or a chain whose ends were not the
    states its nodes claimed.  That is precisely the class of error worth catching at the
    seam rather than three passes downstream, where it reads as a planner that produces
    untraversable routes.
    """
    route[0] = np.asarray(qa, dtype=float)
    route[-1] = np.asarray(qb, dtype=float)
    for k, (x, y) in enumerate(zip(route, route[1:])):
        if cell.segment_collides(x, y, max_step=max_step):
            raise PlanningError(
                f"cartesian tree: assembled route is blocked at move {k} -> {k + 1}, "
                f"which means the tree and the route disagree")
    return route
