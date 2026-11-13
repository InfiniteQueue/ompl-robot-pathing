"""Joint-space tree search: the Cartesian tree's framework with the joint chord as its edge.

:mod:`weldpath.cartesian` exists because OMPL cannot plan linear motion.  This module
exists for the opposite reason: OMPL *can* plan joint motion, and does, and we cannot see
inside it.  What the profiling found is that a sampling run's time is not going where the
obvious story says it is -- ``contactTest`` is **0.013 ms** with the robot 150 mm clear
against **4.7 ms** 5 mm off a panel, so collision checking in open air is nearly free, and
yet an OMPL run in open air is not.  The cost is inside the planner, it is not the
geometry, and no flag reaches it.  Putting the same state space and the same edge in a
loop of our own makes that cost attributable and the sampling ours to change.

**The same state space, deliberately.**  Nodes are joint vectors, as OMPL's are and as the
Cartesian tree's are.  Nothing here is a different problem; it is the same problem solved
where we can read the clock.

**Three differences that are not incidental.**

* **The edge is checked by the cell, not by the planner.**  Every extend is
  :meth:`Cell.segment_collides` at ``--check-step-deg`` with the tool-space bisection
  under it -- the *same* test ``planning._route_fault`` applies to a finished route.  So a
  route from here cannot be refused by it.  OMPL's can and is: it bounds
  ``longest_valid_segment_length``, one distance over the whole joint vector in its own
  metric, which is not commensurable with the largest-single-joint rule the cell uses, and
  counted over the shipped logs of nine studies **47 of 1017 sampler runs came back with a
  route this cell then refused**.  Those are whole searches thrown away.
* **A blocked extend keeps its clear prefix.**  The extend is that same
  ``segment_collides`` cut into ``--check-step-deg`` pieces so that the part which cleared
  is still a place for the tree to stand.  This is the one thing a single-shot check
  cannot give a search, and it is the trick :mod:`weldpath.cartesian` is built on.
* **No inverse kinematics anywhere.**  The Cartesian tree pays a ``solve_pose`` per
  station and refuses any station that lands further than ``jump_rad`` from the last, both
  because its edge is a pose path that an arm has to be found for.  A joint chord is
  already a joint path.  That removes the per-station IK, the branch-flip refusal, and --
  the part that matters for what this is *for* -- the constraint that every edge be a
  straight tool move.  Measured on twelve real failing extractions, the two Cartesian
  trees ended 400 to 600 mm apart after 250 samples with no sampling policy closing the
  gap: those legs are not short of samples, they are short of a *shape*, and a joint edge
  is the shape they were missing.

**Edges do not carry their states, and that is a real difference from the Cartesian
tree.**  That rule is there because ``plan_linear`` is greedy and seeded, so asking it the
same question twice can give a different chain or none; a route that re-derived
differently downstream would be thrown out by a verifier right to throw it out.  A joint
chord has nothing to re-derive -- ``a + t * (b - a)`` is the same points every time -- so
only the two ends need keeping.  What is returned is nevertheless the **dense** station
list rather than the waypoints, for a different reason: ``planning._split_runs`` may label
a near-panel move linear, and a linear move is flown along the tool's straight line rather
than along the joint chord this validated.  At one check step apart the two curves
coincide, so handing back the stations is what makes the route clear under the cell's own
check *as shipped* rather than only as planned.  Returning the waypoints would also have
been sound -- it is exactly the footing OMPL's four-point routes are already on, and
``_verify_runs`` is what catches a run that will not fly -- but it would be sound by the
grace of a downstream pass rather than by construction.

**One metric, and it is the max-norm.**  ``segment_collides`` lays its grid down by the
*largest single joint's* delta, so the max-norm is what an edge's price is proportional to.
Ranking neighbours by anything else would rank by one quantity and pay for another: an L2
metric prefers a node that is close in the aggregate over one that is close in the joint
that will decide the check count.  So ``extend_rad`` is a max-norm bound, ``nearest`` is a
max-norm scan, and the two are the same number throughout.  Per-joint weights are offered
for the case where that is the wrong call, and default to none.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .cell import Cell
from .planning import PlanningError

__all__ = ["JointBudget", "plan_joint"]


@dataclass
class JointBudget:
    """How hard the tree searches, and how far one extend drives the joint vector.

    The time and count budgets are per gun opening for the reason
    :class:`weldpath.cartesian.CartesianBudget`'s are: the gun is part of the shape being
    steered around, so a search that failed is evidence about the opening it ran at and
    about no other.

    ``extend_rad`` is the setting worth understanding first.  It is the furthest any one
    joint moves in a single extend, so an extend costs at most ``extend_rad / max_step``
    collision tests and that is the whole of what it costs -- there is no inverse
    kinematics behind it.  The trade is the usual one: short extends grow the tree slowly
    and pay a nearest-neighbour scan per unit of progress, long ones cover ground but
    spend more of their tests past the point where the chord is already blocked.  Unlike
    the Cartesian tree's ``extend_mm`` there is no wasted IK in that overrun, which is
    what makes a long extend cheaper here than there.

    ``margin_rad`` is the joint-space analogue of ``CartesianBudget.margin_mm`` and is
    **off by default**, because the thing being compared against samples the full
    kinematic limits.  Set, it confines samples to the box the two endpoints span inflated
    by this much, intersected with the limits.  It is a hard wall on where the arm may be
    put, not a preference: an extend travels at most ``extend_rad`` towards a sample, so
    every node lies in the convex hull of the endpoints and the box, and the box contains
    them.

    ``weights`` scales each joint before the metric is taken, for a cell where a radian of
    one joint is not worth a radian of another.  ``None`` is unit weights, which is what
    the max-norm argument in the module docstring assumes; anything else makes ``nearest``
    rank by something other than what an edge costs, so it is an override rather than a
    tuning knob.
    """
    seconds: float = 20.0           # wall clock before one gun opening is abandoned
    process_seconds: float = 20.0   # wall clock on better routes, once one is found
    gun_openings: int = 3           # least openings attempted, where that many exist
    max_iters: int = 20000          # sample budget, whichever runs out first
    extend_rad: float = 0.50        # furthest any one joint moves in one extend
    goal_bias: float = 0.10         # share of samples aimed at the far tree's root
    margin_rad: float | None = None  # confine sampling to the endpoints' box, inflated
    weights: tuple | None = None    # per-joint scaling before the metric; None is unit
    seed: int = 0

    @property
    def enabled(self) -> bool:
        return self.seconds > 0.0 and self.max_iters > 0


@dataclass
class _Node:
    q: np.ndarray
    parent: int                     # -1 at the root


class _Tree:
    """Nodes and the edges between them, with a max-norm nearest-neighbour scan.

    The scan is one vectorised reduction over a contiguous array that grows by doubling,
    which is the same arrangement :class:`weldpath.cartesian._Tree` uses and for the same
    reason: it runs once per sample and once per connect pass, so it is on the hot path
    even though each one is cheap.
    """

    def __init__(self, root_q: np.ndarray, weights: np.ndarray | None):
        root_q = np.asarray(root_q, dtype=float)
        self.nodes: list[_Node] = [_Node(root_q, -1)]
        self.w = weights
        self._q = np.empty((16, len(root_q)), dtype=float)
        self._n = 0
        self._push(root_q)

    def _push(self, q: np.ndarray) -> None:
        if self._n == len(self._q):
            self._q = np.vstack([self._q, np.empty_like(self._q)])
        self._q[self._n] = q
        self._n += 1

    def add(self, q: np.ndarray, parent: int) -> int:
        q = np.asarray(q, dtype=float)
        self.nodes.append(_Node(q, parent))
        self._push(q)
        return len(self.nodes) - 1

    def nearest(self, q: np.ndarray) -> int:
        d = np.abs(self._q[:self._n] - q)
        if self.w is not None:
            d = d * self.w
        return int(np.argmin(np.max(d, axis=1)))

    def route_from_root(self, index: int) -> list[np.ndarray]:
        """The waypoints from this tree's root down to ``index``, in that order."""
        out: list[np.ndarray] = []
        i = index
        while i >= 0:
            out.append(self.nodes[i].q)
            i = self.nodes[i].parent
        out.reverse()
        return out


def _span(a: np.ndarray, b: np.ndarray, w: np.ndarray | None) -> float:
    """The distance the metric reports: the largest single joint's delta, weighted."""
    d = np.abs(np.asarray(b, dtype=float) - np.asarray(a, dtype=float))
    if w is not None:
        d = d * w
    return float(np.max(d))


# ---------------------------------------------------------------------------
# the extend
# ---------------------------------------------------------------------------
def _steer(cell: Cell, max_step: float, budget: JointBudget, w: np.ndarray | None,
           q_from: np.ndarray, q_to: np.ndarray,
           reach_rad: float) -> list[np.ndarray] | None:
    """Drive the joint vector along the chord towards ``q_to``, as far as it will go.

    Returns the stations it reached, ``q_from`` pinned first, or ``None`` when it could not
    leave ``q_from`` at all.  Stopping early is a result and not a failure.

    This is :meth:`Cell.segment_collides` over the whole extend, cut into pieces one
    ``max_step`` long so that a blocked extend still yields the part that cleared.  Each
    piece is asked of ``segment_collides`` rather than of ``in_collision`` on purpose: the
    bisection under the joint grid is what catches the gun stepping clean through a fixture
    between two samples, and a loop over bare state tests would not have it.  The station
    shared by two consecutive pieces is tested twice and costs nothing the second time --
    ``Cell.in_collision`` remembers its verdict per state.
    """
    q_from = np.asarray(q_from, dtype=float)
    d = np.asarray(q_to, dtype=float) - q_from
    span = _span(q_from, q_to, w)
    if span < 1e-12:
        return None

    t = min(1.0, reach_rad / span)
    # Pieces are measured in the *unweighted* largest joint, which is what the grid
    # ``segment_collides`` lays down is measured in; the weights belong to the metric that
    # chose this neighbour, not to the check that validates the edge.
    raw = float(np.max(np.abs(d))) * t
    n = max(1, int(np.ceil(raw / max_step)))

    chain = [q_from]
    current = q_from
    for k in range(1, n + 1):
        q = q_from + d * (t * k / n)
        if not cell.within_limits(q):
            break
        if cell.segment_collides(current, q, max_step=max_step):
            break
        chain.append(q)
        current = q

    return chain if len(chain) > 1 else None


# ---------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------
def _sample(rng: np.random.Generator, cell: Cell, qa: np.ndarray, qb: np.ndarray,
            budget: JointBudget) -> np.ndarray:
    """A configuration to grow towards: uniform over the limits, or over a box in them.

    Uniform over the limits is what OMPL's ``RealVectorStateSpace`` does, and it is the
    default here so that the comparison is of the search and not of the prior put on it.
    ``margin_rad`` is the other option and it is the Cartesian tree's: the arm is expected
    somewhere between its two endpoints, and a detour that leaves that region by more than
    this is not a detour.
    """
    lo, hi = cell.lower, cell.upper
    if budget.margin_rad is not None:
        lo = np.maximum(lo, np.minimum(qa, qb) - budget.margin_rad)
        hi = np.minimum(hi, np.maximum(qa, qb) + budget.margin_rad)
    return rng.uniform(lo, hi)


def plan_joint(cell: Cell, qa: np.ndarray, qb: np.ndarray, *,
               max_step: float, label: str = "",
               budget: JointBudget | None = None, log=None) -> list[np.ndarray]:
    """A route of joint moves from ``qa`` to ``qb``, clear under the cell's own check.

    Bidirectional RRT-Connect: the trees swap each iteration, so whichever is having the
    easier time of it keeps being the one that explores, and after every extend the tree
    that just grew reaches greedily for the other.  That is the same arrangement
    :func:`weldpath.cartesian.plan_cartesian` uses, which is the point -- the framework is
    held fixed and only the edge changes, so a difference in what the two solve is a
    difference between a straight tool move and a joint chord rather than between two
    searches.

    The states come back dense, every station the search validated, as the Cartesian
    tree's do.  See the module docstring for why that and not the waypoints.

    ``label`` names this search in the log line and is used for nothing else.  Nothing is
    offered to :mod:`weldpath.treetrace`: the failure view bakes a tool pose and a
    clearance reading per node, and this search never computes a pose, so offering it
    would make every run pay forward kinematics for a diagnostic only a failed segment
    reads.
    """
    budget = budget or JointBudget()
    rng = np.random.default_rng(budget.seed)
    deadline = time.perf_counter() + budget.seconds
    w = None if budget.weights is None else np.asarray(budget.weights, dtype=float)

    qa = np.asarray(qa, dtype=float)
    qb = np.asarray(qb, dtype=float)
    if cell.in_collision(qa) or cell.in_collision(qb):
        raise PlanningError("joint tree: an endpoint is already in collision")

    start = _Tree(qa, w)
    goal = _Tree(qb, w)
    a, b = start, goal
    forward = True                          # whether `a` is still the start tree

    iters = 0
    while iters < budget.max_iters and time.perf_counter() < deadline:
        iters += 1
        target = (b.nodes[0].q if rng.uniform() < budget.goal_bias
                  else _sample(rng, cell, qa, qb, budget))

        i = a.nearest(target)
        chain = _steer(cell, max_step, budget, w, a.nodes[i].q, target,
                       budget.extend_rad)
        if chain is None:
            a, b, forward = b, a, not forward
            continue
        new = a.add(chain[-1], i)

        # The connect half: reach for the other tree from where we just landed, over and
        # over, which is what makes the two meet in a handful of extends rather than
        # waiting for a sample to fall between them.  In joint space the meeting is
        # unambiguous -- the far node *is* a configuration, so arriving at it is arriving,
        # with no pose that two branches could both answer and no standstill move left to
        # check.
        j = b.nearest(a.nodes[new].q)
        cursor, cursor_index = a.nodes[new].q, new
        while time.perf_counter() < deadline:
            far = b.nodes[j].q
            step = _steer(cell, max_step, budget, w, cursor, far, budget.extend_rad)
            if step is None:
                break
            cursor_index = a.add(step[-1], cursor_index)
            cursor = step[-1]
            if _span(cursor, far, w) > 1e-9:
                continue
            route = _join(a, cursor_index, b, j, forward, max_step)
            if log:
                log(f"      joint tree: joined after {iters} samples, "
                    f"{len(start.nodes) + len(goal.nodes)} nodes, {len(route)} states")
            return _checked(cell, route, max_step, qa, qb)

        a, b, forward = b, a, not forward

    raise PlanningError(
        f"joint tree: no route after {iters} samples and {budget.seconds:g}s "
        f"({len(start.nodes)} and {len(goal.nodes)} nodes)")


def _join(a: _Tree, ia: int, b: _Tree, ib: int, forward: bool,
          max_step: float) -> list[np.ndarray]:
    """The one state list the two trees describe, dense, in start-to-goal order.

    The waypoints are what the trees hold; the stations between them are re-derived here
    rather than stored, a joint chord being the same points every time it is asked for --
    see the module docstring.  ``a`` and ``b`` meet at the same configuration, so the far
    tree's copy of it is dropped on the way in.
    """
    waypoints = a.route_from_root(ia) + b.route_from_root(ib)[::-1][1:]
    if not forward:
        waypoints.reverse()

    out = [waypoints[0]]
    for x, y in zip(waypoints, waypoints[1:]):
        span = float(np.max(np.abs(y - x)))
        n = max(1, int(np.ceil(span / max_step)))
        for k in range(1, n + 1):
            out.append(x + (y - x) * (k / n))
    return out


def _checked(cell: Cell, route: list[np.ndarray], max_step: float,
             qa: np.ndarray, qb: np.ndarray) -> list[np.ndarray]:
    """The assembled route, with the assembly itself checked rather than assumed.

    Every pair here was validated as it was grown, so this can only fail on a bookkeeping
    mistake -- an edge stitched in the wrong order, or a tree reversed that should not have
    been.  That is the class of error worth catching at the seam rather than three passes
    downstream, where it reads as a planner producing untraversable routes.  It is nearly
    free: the cell remembers a contact test per state, and these are the states it just
    made.
    """
    route[0] = np.asarray(qa, dtype=float)
    route[-1] = np.asarray(qb, dtype=float)
    for k, (x, y) in enumerate(zip(route, route[1:])):
        if cell.segment_collides(x, y, max_step=max_step):
            raise PlanningError(
                f"joint tree: assembled route is blocked at move {k} -> {k + 1}, "
                f"which means the tree and the route disagree")
    return route
