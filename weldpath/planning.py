"""Motion planning: freespace transits, linear weld approach/depart, and path reduction.

Freespace transits go through OMPL, but only after a direct joint-space move has been
ruled out -- most transits in a fixtured cell are clear, and a single joint move is both
faster to execute and cleaner on the controller than a sampled path.

OMPL's default ``longest_valid_segment_length`` is 0.005 rad.  With convex collision
geometry a discrete check costs a few milliseconds, so that setting spends about a second
of collision checking per tree edge and the planner exhausts its time budget before it
finds anything.  Coarsening it to a few hundredths of a radian is what makes the sampling
planner usable here; it stays well inside the manifest's contact tolerance.
"""
from __future__ import annotations

import ctypes
import struct
import time
from dataclasses import dataclass

import numpy as np

from tesseract_robotics import tesseract_command_language as cl
from tesseract_robotics.tesseract_common import ProfileDictionary
from tesseract_robotics.tesseract_motion_planners import PlannerRequest
from tesseract_robotics.tesseract_motion_planners_ompl import (
    OMPLMotionPlanner, OMPLRealVectorMoveProfile)

from .cell import Cell

OMPL_NAMESPACE = "OMPLMotionPlanner"


class PlanningError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# path post-processing
# ---------------------------------------------------------------------------
class MotionModel:
    """Which profile governs a move, and what that profile forbids and costs.

    The profile is **not** stored against a waypoint.  It is derived from the two states a
    move runs between: the move is linear when either end is inside the panel band.  That
    is what makes it survive the optimisation passes -- shortcut and polish move points
    around, and a waypoint relocated out of the band changes the profile of its own two
    moves with it, with nothing to invalidate and no bookkeeping to get wrong.

    It also gives the marking the output wants for free.  A waypoint is reached by exactly
    one move, and that move is linear if the waypoint is in the band or the one before it
    was -- "ends in the region, or was moved to from a point in the region".

    With no zone in force every move is ``PTP`` and every method here reduces to the plain
    joint-space one it wraps, so the un-zoned path through the planner is unchanged.
    """

    def __init__(self, cell: Cell, *, max_step: float, zone: "LinearZone | None" = None):
        self.cell = cell
        self.max_step = max_step
        self.zone = zone if (zone is not None and zone.enabled) else None
        # One clearance query per distinct state: the query loads a joint state into the
        # environment, which costs far more than everything else these passes do.
        self._near: dict[bytes, bool] = {}

    # -- which profile ------------------------------------------------------
    def near(self, q: np.ndarray) -> bool:
        if self.zone is None:
            return False
        key = np.asarray(q, dtype=float).tobytes()
        hit = self._near.get(key)
        if hit is None:
            hit = bool(self.cell.clearance_mm(q) <= self.zone.near_mm)
            self._near[key] = hit
        return hit

    def motion(self, a: np.ndarray, b: np.ndarray) -> str:
        return LIN if (self.near(a) or self.near(b)) else PTP

    def demotes(self, a: np.ndarray, b: np.ndarray, replaced) -> bool:
        """Would replacing ``replaced`` with the move a->b drop it out of the profile?

        The optimisation passes exist to delete waypoints, and deleting is nearly always
        quicker: every retained hop pays its own pair of ramps, so a chord across twenty
        dense points beats them on time before the clearance penalty is even consulted.
        Left alone they would therefore dissolve exactly the near-panel stretches this is
        here to create, and hand back a single long move whose two ends sit outside the
        band -- which by the endpoint rule is joint motion, sweeping an arc through the
        very region the linear profile was chosen for.

        So a replacement that takes points out of the band without inheriting the band
        itself is refused.  Everything else the passes want to do is still allowed,
        including reshaping and thinning the stretch, since a chord between two points
        that are themselves in the band stays linear and stays checked as one.
        """
        if self.zone is None or self.motion(a, b) == LIN:
            return False
        return any(self.near(q) for q in replaced)

    # -- what it forbids ----------------------------------------------------
    def blocked(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Whether this move is unusable, checked along the path it will really take."""
        if self.motion(a, b) == PTP:
            return self.cell.segment_collides(a, b, max_step=self.max_step)
        return not self.linear_ok(a, b)

    def _pose_mm(self, q: np.ndarray) -> np.ndarray:
        """TCP pose in manifest units.

        ``fk`` answers in environment units, which are metres, while ``plan_linear`` and
        the inverse kinematics behind it take manifest units and apply the scale
        themselves.  Handing the raw transform over asks for a pose a millimetre from the
        base, which has no solution, so every linear move reads as unreachable.
        """
        T = self.cell.fk(q).copy()
        T[:3, 3] /= self.cell.man.scale
        return T

    def linear_ok(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Can the tool run straight from ``a`` to ``b``?

        Under a linear profile the controller drives the tool along the straight line and
        solves inverse kinematics as it goes, so that line -- not the joint chord between
        the same two states -- is what has to be reachable and clear.  A joint-space check
        here would be checking a path the robot does not take, and for the long chords the
        shortcut pass proposes the two are nowhere near each other.
        """
        try:
            chain = plan_linear(self.cell, self._pose_mm(a), self._pose_mm(b), a,
                                step_mm=self.zone.step_mm)
        except PlanningError:
            return False                    # no inverse kinematics somewhere along it
        # Inverse kinematics returns the solution nearest its seed rather than the state
        # asked for, so pin the ends back before checking the gaps between the samples.
        chain[0] = np.asarray(a, dtype=float)
        chain[-1] = np.asarray(b, dtype=float)
        return not _chain_collides(self.cell, chain, self.max_step)

    # -- what it costs ------------------------------------------------------
    def _crosses(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Whether this move reaches into the band from outside it, or back out."""
        return self.zone is not None and self.near(a) != self.near(b)

    def _crossing_floor(self, a: np.ndarray, b: np.ndarray) -> float:
        """Seconds a move that reaches into the band from outside it is held to.

        Joint motion is wanted outside the band, but the profile rule makes a move linear
        when *either* end is near, so a chord reaching in from open space is linear over
        its whole length.  ``demotes`` does not object -- it refuses replacements that
        come out ``PTP``, and this one does not -- so nothing stops the optimisation
        passes deleting the waypoint at the edge of the band and handing back one long
        straight sweep in from far outside.  What the route should do instead is stop at
        the edge: joint motion out to it, linear from there in.

        Charged as a speed the move is costed at, and only on a move that crosses the
        edge.  Both of those are load bearing:

        * **On the crossing only.**  A move wholly inside the band pays nothing, which is
          the point -- that motion is wanted.  A tax on every linear move cannot decide
          this question at all: ``simplify`` weighs the chord against the polyline it
          would replace, whose own near-panel hops are linear too, so both sides scale
          together, and since each retained hop pays its own pair of ramps the polyline's
          linear part outweighs the chord's.  The chord's share of such a tax is the
          smaller one, so the collapse survives any multiplier whatsoever.
        * **As a speed, not a multiplier.**  A multiplier is a ratio that both sides of
          the comparison carry, so it converges: raising it shortens the sweep and then
          stops responding.  A speed makes the cost an absolute quantity set by how far
          the move actually runs, so the long reach in from open space is charged for its
          length while the short hop across the edge -- the move the route is supposed to
          keep -- is charged for almost none of it.  That difference does not cancel, so
          the setting keeps biting however far it is pushed.

        The figure is used for costing and nothing else: the robot is never scheduled by
        it, and it is deliberately not a speed the cell obeys anywhere.
        """
        if self.zone.crossing_speed_mm_s <= 0.0 or not self._crosses(a, b):
            return 0.0
        return _tcp_travel(self.cell, [a, b]) / self.zone.crossing_speed_mm_s

    def _time_floor(self, a: np.ndarray, b: np.ndarray) -> float:
        """Seconds this linear move is held to, over and above the joint limits.

        Two floors can apply and the higher wins, which is what keeps them independent of
        one another; either may be off without disturbing the other.  ``linear_speed_mm_s``
        is a prediction -- the same figure schedules the exported program -- while the
        crossing speed is a preference and reaches nothing outside these passes.
        """
        if self.zone is None or self.motion(a, b) == PTP:
            return 0.0
        floor = self._crossing_floor(a, b)
        if self.zone.linear_speed_mm_s > 0.0:
            floor = max(floor, _tcp_travel(self.cell, [a, b]) / self.zone.linear_speed_mm_s)
        return floor

    def cruise_time(self, a: np.ndarray, b: np.ndarray) -> float:
        return max(self.cell.cruise_time(a, b), self._time_floor(a, b))

    def move_time(self, a: np.ndarray, b: np.ndarray) -> float:
        return max(self.cell.move_time(a, b), self._time_floor(a, b))

    def _known_near(self, q: np.ndarray) -> bool | None:
        """``near(q)`` if it has already been measured, else None.  Never measures."""
        if self.zone is None:
            return False
        return self._near.get(np.asarray(q, dtype=float).tobytes())

    def bound_time(self, a: np.ndarray, b: np.ndarray) -> float:
        """A lower bound on the stop-to-stop cost, guaranteed not to measure clearance.

        Bounds ``cost(a, b, stops=True)`` specifically.  It is built on the full move time,
        ramps included, so it is *not* a bound on the cruise-only cost -- a caller
        comparing against cruise figures wants ``cell.cruise_time`` instead, which is what
        ``shortcut`` already uses.

        The optimisation passes draw candidates at random and most of them lose, so they
        reject on a bound before paying for the real thing.  For that to be worth doing the
        bound has to be cheap, and :meth:`move_time` is not: it asks :meth:`motion` which
        profile governs the move, and a state nothing has measured yet answers that with a
        contact test over every convex piece within the probe -- on a candidate that is
        about to be thrown away.

        So this asks only what is already known.  Every clearance factor is at least 1, so
        the joint-limit time alone is always a valid floor.  The tool speed cap can be
        added on top of it whenever one end is *known* to be near, since the move is then
        linear whatever the other end turns out to be -- and in a cell where the route
        hugs the panels that is the common case, so the bound is usually as tight as the
        real one for nothing.  Where neither end has been measured the cap is left out
        rather than measured for: a looser bound rejects fewer candidates, which is a
        cost, but a wrong one would reject a candidate that should have won.

        The crossing surcharge is always left out.  ``_crosses`` compares both ends and so
        cannot be answered from one of them, and since it only ever raises the cost,
        omitting it keeps this below the true figure.  It is applied in full by
        :meth:`cost`, on the candidates that get that far.
        """
        base = self.cell.move_time(a, b)
        if self.zone is None or self.zone.linear_speed_mm_s <= 0.0:
            return base
        if not (self._known_near(a) or self._known_near(b)):
            return base
        return max(base, _tcp_travel(self.cell, [a, b]) / self.zone.linear_speed_mm_s)

    def cost(self, a: np.ndarray, b: np.ndarray, *, fa: float | None = None,
             fb: float | None = None, stops: bool = False) -> float:
        """Penalised time, with the tool speed cap folded in.

        The cap raises the base time; the clearance penalty on top of it is unchanged, so
        the two are combined by scaling rather than by replacement.  Optimising without
        this would let the passes trade a joint move for a linear one that is quicker on
        the joint limits and slower once the tool speed governs it.

        The crossing speed rides on that same floor, so it too compounds with the
        clearance penalty rather than adding to it: a move that both reaches into the band
        from outside and runs hard against a panel is charged for both.
        """
        penalised = self.cell.segment_cost(a, b, max_step=self.max_step,
                                           fa=fa, fb=fb, stops=stops)
        floor = self._time_floor(a, b)
        if floor <= 0.0:
            return penalised
        raw = self.cell.move_time(a, b) if stops else self.cell.cruise_time(a, b)
        if raw <= 0.0:
            return max(penalised, floor)
        return penalised * max(1.0, floor / raw)

    # -- pass-through -------------------------------------------------------
    def penalty_factor(self, q: np.ndarray) -> float:
        return self.cell.penalty_factor(q)

    @property
    def penalised(self) -> bool:
        return self.cell.penalty is not None and self.cell.penalty.enabled


def simplify(model: MotionModel, path: list[np.ndarray]) -> list[np.ndarray]:
    """Reduce a path to the fewest waypoints that still traverse it without collision.

    From each kept waypoint, reach for the furthest later waypoint the robot can get to
    on a straight joint-space move, and drop everything in between.  Because the test is
    a collision check on the replacement move -- not a geometric deviation tolerance --
    the result is guaranteed to be traversable, which a Douglas-Peucker style reduction
    is not: dropping a point that merely lies close to the chord can cut the corner
    straight through an obstacle.

    A sampled path wanders and the controller does not need the wandering, so this is
    also what keeps the output down to the handful of points a FANUC program wants.

    Every point kept is a point the robot comes to a full stop at, so the comparison is
    made in **penalised time under the real joint dynamics**: keeping ``i..j`` costs the
    sum of the individual bang-bang moves between them, against one move from ``i`` to
    ``j``.  Deleting is always the faster of the two -- the direct move is no further on
    any joint than the detour, and it pays one pair of ramps instead of many -- so what
    decides it is the clearance penalty.  Reductions that push the route nearer the parts
    are refused, which is what stops this pass, running last, from quietly undoing the
    standoff the shortcut pass just bought: the chord across a corner the robot took wide
    is quicker, collision free, and hard against the panel.
    """
    if len(path) < 3:
        return [p.copy() for p in path]
    factors: dict[int, float] = {}

    def factor(k: int) -> float:
        if k not in factors:
            factors[k] = model.penalty_factor(path[k])
        return factors[k]

    def polyline_cost(i: int, j: int) -> float:
        # Each retained hop is its own stop-to-stop move.  The hops are short, so the
        # endpoint factors describe them well enough without sampling their interiors.
        return sum(model.move_time(path[k], path[k + 1]) * max(factor(k), factor(k + 1))
                   for k in range(i, j))

    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if model.demotes(path[i], path[j], path[i + 1:j]):
                j -= 1
                continue
            if model.blocked(path[i], path[j]):
                j -= 1
                continue
            chord = model.cost(path[i], path[j],
                               fa=factor(i), fb=factor(j), stops=True)
            if chord > polyline_cost(i, j) + 1e-9:
                j -= 1
                continue
            break
        out.append(path[j])
        i = j
    return out


def _hop(model: "MotionModel", a: np.ndarray, b: np.ndarray,
         fa: float, fb: float) -> float:
    """Penalised time of one emitted move: full bang-bang, ramps included."""
    return model.cost(a, b, fa=fa, fb=fb, stops=True)


def polish(model: "MotionModel", path: list[np.ndarray], *, time_budget: float = 5.0,
           rng: np.random.Generator | None = None,
           log=None) -> list[np.ndarray]:
    """Remove and relocate waypoints of the reduced path, judged on penalised time.

    The passes before this one work on a densified path, where a point is a sampling
    artefact and the honest measure of a change is cruise time.  Here every point is one
    the robot will really stop at, so the measure is the full stop-to-stop time, ramps
    included -- and a waypoint that buys nothing is now visibly expensive rather than free.
    Two moves, applied to interior waypoints only, since the endpoints belong to the
    locators either side:

    * **remove** -- drop a waypoint when going straight past it is quicker under the
      penalty than stopping at it.  Ignoring the penalty this is always true, so what it
      really tests is whether the corner was bought for clearance or is just left over.
    * **relocate** -- displace a waypoint and keep it if the pair of moves through it gets
      quicker.  This is the move that can unpick a cluster the removal test alone cannot:
      shifting a point away from a panel can be what makes its neighbour droppable, so a
      removal sweep follows every accepted relocation.

    ``simplify`` already applies the same removal test greedily, so this is a second,
    non-greedy opinion on it rather than the first -- what is new here is the relocation,
    and the sweep that follows it.  Every replacement is collision checked before it is
    kept, so the result stays traversable.
    """
    if len(path) < 3 or time_budget <= 0:
        return [np.asarray(p, dtype=float).copy() for p in path]

    cell = model.cell
    rng = rng or np.random.default_rng(1)
    pts = [np.asarray(p, dtype=float).copy() for p in path]
    fac = [model.penalty_factor(p) for p in pts]
    costs = [_hop(model, pts[i], pts[i + 1], fac[i], fac[i + 1])
             for i in range(len(pts) - 1)]
    before = sum(costs)
    deadline = time.time() + time_budget
    dropped = moved = tried = 0

    def drop_sweep() -> None:
        nonlocal dropped
        k = 1
        while k < len(pts) - 1 and time.time() < deadline:
            if model.demotes(pts[k - 1], pts[k + 1], [pts[k]]):
                k += 1
                continue
            if model.blocked(pts[k - 1], pts[k + 1]):
                k += 1
                continue
            direct = _hop(model, pts[k - 1], pts[k + 1], fac[k - 1], fac[k + 1])
            if direct >= costs[k - 1] + costs[k] - 1e-9:
                k += 1
                continue
            del pts[k], fac[k]
            costs[k - 1:k + 1] = [direct]
            dropped += 1
            # Deliberately not advancing: the point that has just moved into k is now next
            # to a different neighbour and deserves its own test.

    drop_sweep()
    while time.time() < deadline and len(pts) > 2:
        tried += 1
        k = int(rng.integers(1, len(pts) - 1))
        # Drawn in joint space but scaled through the cell's joint weights, so an attempt
        # moves the tool about as far whichever joints it happens to use.
        direction = rng.normal(size=len(pts[k]))
        reach = float(np.linalg.norm(direction * cell.weights))
        if reach <= 0.0:
            continue
        candidate = np.clip(pts[k] + direction * (float(rng.uniform(0.005, 0.15)) / reach),
                            cell.lower, cell.upper)
        if not cell.within_limits(candidate):
            continue
        # Unpenalised time is a lower bound on penalised time, so this rejects most
        # candidates before paying for a collision check or a clearance query.  It has to
        # be ``bound_time`` rather than ``move_time`` to keep that promise: the latter
        # settles the move's profile, which measures the candidate's clearance -- the
        # dearest query there is, spent on a point that is usually about to be discarded.
        budget = costs[k - 1] + costs[k]
        if (model.bound_time(pts[k - 1], candidate)
                + model.bound_time(candidate, pts[k + 1])) >= budget - 1e-9:
            continue
        if (model.demotes(pts[k - 1], candidate, [pts[k]])
                or model.demotes(candidate, pts[k + 1], [pts[k]])):
            continue
        if model.blocked(pts[k - 1], candidate):
            continue
        if model.blocked(candidate, pts[k + 1]):
            continue
        f = model.penalty_factor(candidate)
        first = _hop(model, pts[k - 1], candidate, fac[k - 1], f)
        second = _hop(model, candidate, pts[k + 1], f, fac[k + 1])
        if first + second >= budget - 1e-9:
            continue
        pts[k], fac[k] = candidate, f
        costs[k - 1], costs[k] = first, second
        moved += 1
        drop_sweep()

    if log:
        after = sum(costs)
        gain = 100.0 * (1.0 - after / before) if before > 0 else 0.0
        log(f"      polish: dropped {dropped} and relocated {moved} waypoints from "
            f"{tried} attempts, penalised time {before:.2f} -> {after:.2f} s "
            f"({gain:.0f}% better), {len(pts)} points")
    return pts


def _refine(model: "MotionModel", path: list[np.ndarray], *, shortcut_seconds: float,
            polish_seconds: float, log) -> list[np.ndarray]:
    """The whole post-processing chain, in the order the three passes need to run.

    Shortcutting reshapes the route while it is still dense, reduction picks which of those
    points are actually worth stopping at, and polishing then judges those stops under the
    time they really cost.
    """
    log(f"      refining {len(path)} points: up to {shortcut_seconds:g}s shortcutting "
        f"then {polish_seconds:g}s polishing, both spent in full")
    improved = shortcut(model, path, time_budget=shortcut_seconds, log=log)
    reduced = simplify(model, improved)
    return polish(model, reduced, time_budget=polish_seconds, log=log)


def _resample(cell: Cell, a: np.ndarray, b: np.ndarray, step: float) -> list[np.ndarray]:
    """Points strictly between a and b, spaced no further apart than ``step``."""
    n = max(1, int(np.ceil(float(np.max(np.abs(b - a))) / max(step, 1e-9))))
    return [a + (b - a) * (k / n) for k in range(1, n)]


def _densify(cell: Cell, path: list[np.ndarray], step: float) -> list[np.ndarray]:
    """A sampling planner's handful of states, filled in at collision-check resolution.

    OMPL returns very few points -- four is typical -- and every pass downstream wants the
    route rather than the tree's nodes.  The spacing is a joint-space one, the same
    ``--check-step-deg`` the checking uses, so consecutive points differ by less than the
    resolution anything here can resolve.
    """
    out: list[np.ndarray] = [np.asarray(path[0], dtype=float)]
    for a, b in zip(path, path[1:]):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        out.extend(_resample(cell, a, b, step))
        out.append(b)
    return out


def shortcut(model: "MotionModel", path: list[np.ndarray], *, time_budget: float = 2.0,
             rng: np.random.Generator | None = None,
             log=None) -> list[np.ndarray]:
    """Reshape a path to take less time, penalised for running close to the parts.

    ``simplify`` can only delete waypoints that a straight move already bypasses, so it
    never changes the route: a sampled path that swings the arm around the base to reach a
    point beside it keeps that swing.  This pass reshapes the route instead, using two
    moves that between them can both shorten and stand off:

    * **cut** -- replace a stretch of path with the straight move between its ends, when
      that move is collision free and cheaper.  This is what removes gross detours, but it
      can only ever remove path, so on its own it cannot move away from an obstacle.
    * **relocate** -- displace a single waypoint and keep it if both neighbouring moves
      stay clear and the pair gets cheaper.  This is what gives the clearance penalty
      teeth: pushing a waypoint away from a panel costs a little travel and saves a lot of
      penalty, so it wins.  Every waypoint is eligible except the two endpoints, which are
      the states handed in and belong to the locators either side.

    Cost is **time**, under the same joint velocity limits the output is scheduled with, so
    a wide J1 excursion is cut before a wrist rotation that covers the same angle far
    faster; time spent near a panel is multiplied by
    :class:`~weldpath.penalty.ClearancePenalty`.

    The path is densified first, so changes can land between the planner's own waypoints
    instead of only at them -- which is also why this pass counts *cruise* time and leaves
    the ramps to :func:`simplify`.  A densified point is a sampling artefact, not a stop
    the robot will make, so charging it a full acceleration ramp would score the route by
    how finely it happened to be sampled and make every cut look good regardless of where
    it went.  Cruise time is unchanged by subdivision, so this pass judges the route's
    shape and ``simplify`` judges how many stops it needs.

    Work is bounded by ``time_budget`` seconds; the result is always collision free, since
    every replacement is checked before it is kept.
    """
    if len(path) < 3 or time_budget <= 0:
        return [np.asarray(p, dtype=float).copy() for p in path]

    cell = model.cell
    max_step = model.max_step
    dense = _densify(cell, path, max_step)

    # One clearance query per waypoint, cached: the geometry query dominates, so scoring a
    # candidate has to be arithmetic over remembered factors rather than fresh queries.
    penalised = model.penalised
    fac = [model.penalty_factor(p) if penalised else 1.0 for p in dense]

    def step_cost(x: np.ndarray, y: np.ndarray, fx: float, fy: float) -> float:
        return model.cruise_time(x, y) * max(fx, fy)

    def span_cost(i: int, j: int) -> float:
        return sum(step_cost(dense[k], dense[k + 1], fac[k], fac[k + 1])
                   for k in range(i, j))

    rng = rng or np.random.default_rng(0)
    before = span_cost(0, len(dense) - 1)
    deadline = time.time() + time_budget
    tried = cuts = moves = 0

    while time.time() < deadline and len(dense) > 2:
        tried += 1
        if rng.random() < 0.5:
            cuts += _try_cut(model, dense, fac, rng, penalised, span_cost)
        else:
            moves += _try_relocate(model, dense, fac, rng, penalised, step_cost)

    if log:
        after = span_cost(0, len(dense) - 1)
        gain = 100.0 * (1.0 - after / before) if before > 0 else 0.0
        detail = "penalised cruise time" if penalised else "cruise time"
        log(f"      shortcut: {cuts} cuts and {moves} relocations kept from {tried} "
            f"attempts, {detail} {before:.2f} -> {after:.2f} s ({gain:.0f}% better)")
    return dense


def _try_cut(model: "MotionModel", dense, fac, rng, penalised, span_cost) -> int:
    """Replace dense[i..j] with the straight move between the ends, if that is cheaper."""
    i, j = sorted(rng.integers(0, len(dense), size=2))
    if j - i < 2:
        return 0
    span = span_cost(i, j)
    # Every factor is at least 1 and cruise time is additive, so the unpenalised cruise
    # time of the direct move is a valid lower bound on what the replacement can cost.
    # Rejecting on that first keeps the expensive checks off the many candidates that
    # were never going to win.
    if model.cell.cruise_time(dense[i], dense[j]) >= span - 1e-9:
        return 0
    if model.demotes(dense[i], dense[j], dense[i + 1:j]):
        return 0
    if model.blocked(dense[i], dense[j]):
        return 0
    points = _resample(model.cell, dense[i], dense[j], model.max_step)
    if penalised:
        factors = [model.penalty_factor(p) for p in points]
        chain = [dense[i]] + points + [dense[j]]
        chain_f = [fac[i]] + factors + [fac[j]]
        direct = sum(model.cruise_time(x, y) * max(fx, fy)
                     for x, y, fx, fy in zip(chain, chain[1:], chain_f, chain_f[1:]))
        if direct >= span - 1e-9:
            return 0
    else:
        factors = [1.0] * len(points)
    dense[i + 1:j] = points
    fac[i + 1:j] = factors
    return 1


def _try_relocate(model: "MotionModel", dense, fac, rng, penalised, step_cost) -> int:
    """Displace one interior waypoint and keep the move if it lowers the local cost.

    The displacement is drawn in joint space but scaled by the cell's joint weights, so a
    given attempt moves the tool about as far whichever joints it uses -- otherwise almost
    every sample would be a wrist twiddle that changes nothing.
    """
    if len(dense) < 3:
        return 0
    cell = model.cell
    k = int(rng.integers(1, len(dense) - 1))        # endpoints are fixed by the caller
    target = float(rng.uniform(0.005, 0.15))        # metres of tool travel
    direction = rng.normal(size=len(dense[k]))
    reach = float(np.linalg.norm(direction * cell.weights))
    if reach <= 0.0:
        return 0
    candidate = dense[k] + direction * (target / reach)
    candidate = np.clip(candidate, cell.lower, cell.upper)
    if not cell.within_limits(candidate):
        return 0

    before = (step_cost(dense[k - 1], dense[k], fac[k - 1], fac[k])
              + step_cost(dense[k], dense[k + 1], fac[k], fac[k + 1]))
    # Cheapest possible replacement, ignoring any penalty, as an early reject.
    floor = (cell.cruise_time(dense[k - 1], candidate)
             + cell.cruise_time(candidate, dense[k + 1]))
    if floor >= before - 1e-9:
        return 0
    if (model.demotes(dense[k - 1], candidate, [dense[k]])
            or model.demotes(candidate, dense[k + 1], [dense[k]])):
        return 0
    if model.blocked(dense[k - 1], candidate):
        return 0
    if model.blocked(candidate, dense[k + 1]):
        return 0

    f = model.penalty_factor(candidate) if penalised else 1.0
    after = (step_cost(dense[k - 1], candidate, fac[k - 1], f)
             + step_cost(candidate, dense[k + 1], f, fac[k + 1]))
    if after >= before - 1e-9:
        return 0
    dense[k] = candidate
    fac[k] = f
    return 1


# ---------------------------------------------------------------------------
# freespace
# ---------------------------------------------------------------------------
def _path_cost(cell: Cell, path: list[np.ndarray], max_step: float
               ) -> tuple[float, float]:
    """Penalised time for a whole path, and its plain time.

    Cruise time, not stop-to-stop time: this scores a raw sampling-planner solution, whose
    waypoint count is an artefact of how the tree happened to grow rather than a decision
    anyone made.  Ranking solutions on stop-to-stop time would mostly rank them on how many
    nodes each one took, which says nothing about the route.

    With no penalty in force the two figures are equal, so ranking on the first still ranks
    on time and the choice degrades to "quickest raw solution" rather than to nothing.
    Endpoint factors are threaded from one segment to the next so each waypoint costs one
    clearance query rather than two.
    """
    if len(path) < 2:
        return 0.0, 0.0
    penalised = cell.penalty is not None and cell.penalty.enabled
    total = plain = 0.0
    prev_f = cell.penalty_factor(path[0]) if penalised else None
    for a, b in zip(path, path[1:]):
        next_f = cell.penalty_factor(b) if penalised else None
        total += cell.segment_cost(a, b, max_step=max_step, fa=prev_f, fb=next_f)
        plain += cell.cruise_time(a, b)
        prev_f = next_f
    return total, plain


def _make_program(cell: Cell, qa: np.ndarray, qb: np.ndarray) -> cl.CompositeInstruction:
    program = cl.CompositeInstruction("DEFAULT")
    program.setManipulatorInfo(cell.manip_info)
    start = cl.MoveInstruction(
        cl.WaypointPoly_wrap_StateWaypoint(cl.StateWaypoint(cell.joint_names, qa)),
        cl.MoveInstructionType_FREESPACE, "DEFAULT")
    program.push_back(cl.InstructionPoly_wrap_MoveInstruction(start))
    goal = cl.MoveInstruction(
        cl.WaypointPoly_wrap_JointWaypoint(cl.JointWaypoint(cell.joint_names, qb)),
        cl.MoveInstructionType_FREESPACE, "DEFAULT")
    program.push_back(cl.InstructionPoly_wrap_MoveInstruction(goal))
    return program


# How long one OMPL solve may run.  Tesseract's own default, and what this used to be
# stuck with; see _set_planning_time for why changing it is awkward.
DEFAULT_PLANNING_TIME = 5.0
# The rest of OMPLSolverConfig's documented defaults, used to recognise the struct.
_SOLVER_SIGNATURE = (10, 0, 1)          # max_solutions, simplify, optimize
_planning_time_warned = False


@dataclass
class OmplBudget:
    """How many sampling-planner runs a transit gets, and how long each may search.

    Two phases, because the two things a run can be for are not the same job.

    **Phase one is a choice between routes.**  RRTConnect returns the first path it finds
    and which homotopy class that lands in is luck; no later pass can move a route to the
    other side of an obstacle, so solving several times and keeping the cheapest is the
    only stage that can choose at all.  Every run is spent whether or not earlier ones
    succeeded -- stopping at the first success is exactly what this phase exists not to do.

    **Phase two is a search for any route at all**, and only runs when phase one came back
    empty.  It stops at the first solution, there being nothing to choose between, and it
    carries its own per-run budget: a transit that beat phase one is usually one where
    restarting the tree is the problem, so the time that helps is a longer single search
    rather than another shake of the sampler.
    """
    phase_one_runs: int = 5
    phase_one_seconds: float = DEFAULT_PLANNING_TIME
    phase_two_max_runs: int = 15
    phase_two_seconds: float = DEFAULT_PLANNING_TIME

    @property
    def worst_case_runs(self) -> int:
        return max(self.phase_one_runs, 0) + max(self.phase_two_max_runs, 0)


def _set_planning_time(profile: OMPLRealVectorMoveProfile, seconds: float) -> bool:
    """Set the solver's per-run time limit.  Returns False if it could not be done.

    ``OMPLSolverConfig`` is **not wrapped** by these bindings.  ``profile.solver_config``
    comes back as a bare ``SwigPyObject`` -- a typed pointer with no members exposed -- and
    the module defines no constructor, accessor or factory for the type, so there is no
    supported way to reach ``planning_time`` from Python.  Nor is there a way round it:
    ``OMPLMotionPlanner.terminate()`` exists but only ever *shortens* a solve, and the
    binding's own warning says even that is unimplemented.

    So this writes the field through the pointer, and earns the right to by proving it is
    the right field first.  Rather than trusting a hard-coded offset, it scans for the
    documented default layout -- ``planning_time`` immediately followed by
    ``max_solutions=10``, ``simplify=false``, ``optimize=true`` -- and refuses to write
    unless exactly one candidate matches.  It then reads the value back.  A build that
    reorders the struct or changes its defaults produces no match, so the failure mode is
    "declines to act", not "corrupts the neighbouring field".
    """
    try:
        address = int(profile.solver_config)
    except Exception:
        return False

    blob = ctypes.string_at(address, 128)
    matches = []
    for offset in range(0, len(blob) - 16, 8):
        value = struct.unpack_from("<d", blob, offset)[0]
        if abs(value - DEFAULT_PLANNING_TIME) > 1e-12:
            continue
        rest = (struct.unpack_from("<i", blob, offset + 8)[0],
                blob[offset + 12], blob[offset + 13])
        if rest == _SOLVER_SIGNATURE:
            matches.append(offset)
    if len(matches) != 1:
        return False

    ctypes.memmove(address + matches[0], struct.pack("<d", float(seconds)), 8)
    # Re-fetch the pointer rather than reusing it: this also confirms the write landed on
    # the profile's own member and not on a temporary copy handed out by the getter.
    check = ctypes.string_at(int(profile.solver_config) + matches[0], 8)
    return abs(struct.unpack("<d", check)[0] - float(seconds)) < 1e-12


def _ompl_profile(segment_length: float,
                  planning_time: float = DEFAULT_PLANNING_TIME,
                  log=None) -> OMPLRealVectorMoveProfile:
    global _planning_time_warned
    profile = OMPLRealVectorMoveProfile()
    profile.collision_check_config.longest_valid_segment_length = segment_length
    if abs(planning_time - DEFAULT_PLANNING_TIME) > 1e-12:
        if not _set_planning_time(profile, planning_time) and not _planning_time_warned:
            _planning_time_warned = True
            (log or print)(
                f"      ! cannot set the OMPL time limit with this build of the bindings; "
                f"runs will use {DEFAULT_PLANNING_TIME:g}s, not {planning_time:g}s")
    return profile


def _extract(results) -> list[np.ndarray]:
    """Read joint states out of a planner's result program.

    Walks the instructions rather than going through ``toJointTrajectory``: indexing the
    returned JointTrajectory trips a binding bug in the SWIG wrapper for JointState.
    """
    out: list[np.ndarray] = []

    def visit(composite) -> None:
        for i in range(len(composite)):
            instr = composite[i]
            if instr.isCompositeInstruction():
                visit(cl.InstructionPoly_as_CompositeInstruction(instr))
                continue
            if not instr.isMoveInstruction():
                continue
            move = cl.InstructionPoly_as_MoveInstructionPoly(instr)
            wp = move.getWaypoint()
            if wp.isStateWaypoint():
                pos = cl.WaypointPoly_as_StateWaypointPoly(wp).getPosition()
            elif wp.isJointWaypoint():
                pos = cl.WaypointPoly_as_JointWaypointPoly(wp).getPosition()
            else:
                continue
            out.append(np.asarray(pos, dtype=float).reshape(-1))

    visit(results)
    return out


def _capture(record: list | None, path: list[np.ndarray]) -> list[np.ndarray]:
    """Note the route a leg started from, before any of it is optimised away.

    The refinement passes rewrite a route heavily -- shortcutting bows it away from the
    parts and reduction throws most of its waypoints out -- so the sampling planner's own
    answer is gone by the time anything downstream sees it.  Keeping it is what lets the
    two be compared.
    """
    if record is not None:
        record.append([np.array(q, dtype=float) for q in path])
    return path


def _endpoint_block(cell: Cell, qa: np.ndarray, qb: np.ndarray) -> str | None:
    """Why an endpoint is unusable at the gun's current opening, or ``None`` if both are.

    OMPL discovers this for itself -- "Goal state is in collision", then a run spent
    failing to seed the goal tree -- but only after the whole time budget has gone, and
    every retry at the same opening reaches the same answer just as slowly.  Two contact
    queries settle it first.  The two-leg fallback already screens its intermediate pose
    this way; the endpoints were the omission.

    Both are worth checking, not just the goal.  A transit leaving a weld is planned at the
    opening the gun leaves with, and it is the *start* that the panel constrains there.
    """
    for label, q in (("start", qa), ("goal", qb)):
        if not cell.in_collision(q):
            continue
        touching = sorted(cell.contact_pairs(q).items(), key=lambda kv: kv[1])
        if touching:
            (first, second), distance = touching[0]
            return (f"the {label} pose has {first} {-distance / cell.man.scale:.1f} mm "
                    f"inside {second} with the gun at this opening")
        return f"the {label} pose is in collision with the gun at this opening"
    return None


def _plan_at_opening(cell: Cell, qa: np.ndarray, qb: np.ndarray, *, ompl: OmplBudget,
                     segment_length: float, check_step: float,
                     fallback_via: list[np.ndarray] | None,
                     shortcut_seconds: float, polish_seconds: float,
                     zone: LinearZone | None = None, log=print,
                     record: list | None = None) -> list[Run]:
    """Collision-free joint path from ``qa`` to ``qb`` at the gun's current opening.

    If the direct transit cannot be found, the move is retried in two legs through each
    of ``fallback_via`` -- normally the cell's start pose.  Routing a difficult transit
    through a known-clear pose is what a robot programmer would do by hand, and it turns
    a long detour around the panels into two easy problems.
    """
    blocked = _endpoint_block(cell, qa, qb)
    if blocked:
        # Nothing downstream can rescue this: every route at this opening ends here.  The
        # caller's next candidate opening, or the two-leg split, is the only way on.
        raise PlanningError(blocked)
    try:
        return _plan_direct(cell, qa, qb, ompl=ompl,
                            segment_length=segment_length, check_step=check_step,
                            shortcut_seconds=shortcut_seconds,
                            polish_seconds=polish_seconds,
                            zone=zone, log=log, record=record)
    except PlanningError:
        if not fallback_via:
            raise

    for i, mid in enumerate(fallback_via):
        if cell.in_collision(mid):
            continue
        log(f"      retrying via fallback pose {i + 1}")
        # The halves are recorded jointly below: this is still one leg of the output, and
        # the fallback pose is an implementation detail of how it was found.
        halves: list[list[np.ndarray]] = []
        try:
            # Planned without a linear zone: the two legs are joined below and the whole
            # route is split afterwards, so splitting each half here would put a phase
            # boundary at the fallback pose whether the geometry called for one or not.
            first = _plan_direct(cell, qa, mid, ompl=ompl,
                                 segment_length=segment_length, check_step=check_step,
                                 shortcut_seconds=0.0, polish_seconds=0.0,
                                 log=log, record=halves)
            second = _plan_direct(cell, mid, qb, ompl=ompl,
                                  segment_length=segment_length, check_step=check_step,
                                  shortcut_seconds=0.0, polish_seconds=0.0,
                                  log=log, record=halves)
        except PlanningError:
            continue
        if len(halves) == 2:
            _capture(record, halves[0] + halves[1][1:])
        # Refine the joined route rather than each leg: the detour through the fallback
        # pose is exactly the kind of corner these passes exist to cut.
        joined = first[0].states + second[0].states[1:]
        return _finish(cell, joined, zone=zone, shortcut_seconds=shortcut_seconds,
                       polish_seconds=polish_seconds, check_step=check_step, log=log)
    raise PlanningError("freespace transit failed, including via fallback poses")


def plan_freespace(cell: Cell, qa: np.ndarray, qb: np.ndarray, *,
                   ompl: OmplBudget | None = None, segment_length: float = 0.02,
                   check_step: float = 0.05,
                   fallback_via: list[np.ndarray] | None = None,
                   shortcut_seconds: float = 2.0, polish_seconds: float = 5.0,
                   planning_time: float = DEFAULT_PLANNING_TIME,
                   openings: list[float] | None = None,
                   zone: LinearZone | None = None,
                   record: list | None = None,
                   log=print) -> list[tuple[list[Run], float | None]]:
    """Plan a transit, choosing a gun opening for it when the natural one will not do.

    Some destinations simply cannot be reached at the opening the robot arrives with: the
    tip is 200 mm of swing, so an opening that clears a fixture on the way out fouls it on
    the way back.  ``openings`` lists the openings to consider, most preferred first --
    normally the opening carried over from the previous locator, then closed, then wide.

    Returns one ``(runs, opening)`` leg per gun state.  A single leg is always
    preferred, and a two-leg answer is only produced when no single opening works: the
    gun then changes at the intermediate pose, where the robot is stationary and the
    change costs no motion.

    ``record``, if given, is extended with each leg's route as the sampling planner
    returned it, one entry per returned leg and in the same order.  Failed attempts leave
    nothing behind: only the openings that were actually used contribute.
    """
    ompl = ompl or OmplBudget()

    def attempt(opening, a, b, budget, into=None):
        with cell.gun_opening(opening):
            return _plan_at_opening(cell, a, b, ompl=ompl,
                                    segment_length=segment_length,
                                    check_step=check_step, fallback_via=fallback_via,
                                    shortcut_seconds=budget,
                                    polish_seconds=polish_seconds if budget else 0.0,
                                    zone=zone, log=log, record=into)

    candidates = _opening_candidates(cell, openings)

    # One opening for the whole transit, in preference order: changing the gun is a real
    # operation on the machine, so it is a last resort rather than a free parameter.
    last = None
    for n, opening in enumerate(candidates):
        try:
            if n:
                log(f"      retrying with the gun at {opening:g} mm")
            raw: list = []
            leg = attempt(opening, qa, qb, shortcut_seconds, raw)
            if record is not None:
                record.extend(raw)
            return [(leg, opening)]
        except PlanningError as exc:
            # Said out loud because the endpoint screen rejects an opening in microseconds
            # and would otherwise pass in silence, where a failed OMPL run announces itself
            # at length.  Both reach the same place: this opening is not the one.
            log(f"      the gun at {opening:g} mm will not do: {exc}")
            last = exc

    if len(candidates) < 2 or not fallback_via:
        raise last or PlanningError("freespace transit failed")

    # No single opening reaches: split the move and change the gun partway, at a pose the
    # robot is already passing through and stationary at.
    for i, mid in enumerate(fallback_via):
        for first_open in candidates:
            with cell.gun_opening(first_open):
                if cell.in_collision(mid):
                    continue
            raw_first: list = []
            try:
                first = attempt(first_open, qa, mid, 0.0, raw_first)
            except PlanningError:
                continue
            for second_open in candidates:
                if second_open == first_open:
                    continue                    # already ruled out as a single opening
                raw_second: list = []
                try:
                    second = attempt(second_open, mid, qb, 0.0, raw_second)
                except PlanningError:
                    continue
                if record is not None:
                    record.extend(raw_first + raw_second)
                log(f"      no single gun opening reaches; changing from "
                    f"{first_open:g} mm to {second_open:g} mm at fallback pose {i + 1}")
                with cell.gun_opening(first_open):
                    first = _refine_runs(cell, first, zone=zone,
                                         shortcut_seconds=shortcut_seconds,
                                         polish_seconds=polish_seconds,
                                         check_step=check_step, log=log)
                with cell.gun_opening(second_open):
                    second = _refine_runs(cell, second, zone=zone,
                                          shortcut_seconds=shortcut_seconds,
                                          polish_seconds=polish_seconds,
                                          check_step=check_step, log=log)
                return [(first, first_open), (second, second_open)]
    raise last or PlanningError("freespace transit failed at every gun opening")


def _opening_candidates(cell: Cell, openings: list[float] | None) -> list[float | None]:
    """Openings to try for a transit, most preferred first and without duplicates."""
    if not cell.gun_joint_name:
        return [None]
    widest = cell.man.gun_opening_max
    wanted = list(openings or [])
    wanted += [0.0, widest, widest / 2.0]
    out: list[float] = []
    for value in wanted:
        value = float(min(max(value, 0.0), widest))
        if not any(abs(value - seen) < 1e-6 for seen in out):
            out.append(value)
    return out


def _ompl_run(cell: Cell, qa: np.ndarray, qb: np.ndarray, *, segment_length: float,
              planning_time: float, check_step: float, label: str, log
              ) -> tuple[float, float, list[np.ndarray]] | None:
    """One sampling-planner solve, scored and reported.  ``None`` when it did not solve.

    The planner's own message for a failure is left on ``_ompl_run.message`` rather than
    returned: only the last one is ever quoted, and threading it back through every caller
    to say the same thing is noise.
    """
    profiles = ProfileDictionary()
    profiles.addProfile(OMPL_NAMESPACE, "DEFAULT",
                        _ompl_profile(segment_length, planning_time, log))
    request = PlannerRequest()
    request.env = cell.env
    request.instructions = _make_program(cell, qa, qb)
    request.profiles = profiles
    t0 = time.time()
    response = OMPLMotionPlanner(OMPL_NAMESPACE).solve(request)
    dt = time.time() - t0
    if not response.successful:
        _ompl_run.message = str(response.message)
        log(f"      {label}: {_ompl_run.message} ({dt:.1f}s)")
        return None
    raw = _extract(response.results)
    cost, plain = _path_cost(cell, raw, check_step)
    log(f"      {label}: solved in {dt:.1f}s ({len(raw)} raw points, "
        f"cost {cost:.2f} s against {plain:.2f} s unpenalised)")
    return cost, plain, raw


_ompl_run.message = ""


def _plan_direct(cell: Cell, qa: np.ndarray, qb: np.ndarray, *, ompl: OmplBudget,
                 segment_length: float, check_step: float,
                 shortcut_seconds: float, polish_seconds: float,
                 zone: LinearZone | None = None, log=print,
                 record: list | None = None) -> list[Run]:
    if not cell.segment_collides(qa, qb, max_step=check_step):
        # A clear straight line is normally the best answer there is, and a sampling
        # planner asked to improve on it would only return it again.  But "clear" and
        # "sensible" part company when the line grazes a panel, so when the penalty says
        # this one does, the same pass that stands other routes off is given a chance to
        # bow it away -- there is nothing for OMPL to do here, but plenty for relocation.
        def straight() -> list[Run]:
            return _finish(cell, _capture(record, [qa, qb]), zone=zone,
                           shortcut_seconds=0.0, polish_seconds=0.0,
                           check_step=check_step, log=log)

        raw = cell.move_time(qa, qb)
        if not (cell.penalty is not None and cell.penalty.enabled) or shortcut_seconds <= 0:
            return straight()
        cost = cell.segment_cost(qa, qb, max_step=check_step, stops=True)
        if cost <= raw * 1.05:
            return straight()
        log(f"      direct move is clear but runs close to the parts "
            f"(cost {cost:.2f} s against {raw:.2f} s unpenalised); standing it off")
        # The unrefined route stays the bare straight line; _finish fills the interior in,
        # which a two-point path needs before relocation has anything to move.
        _capture(record, [qa, qb])
        return _finish(cell, [qa, qb], zone=zone, shortcut_seconds=shortcut_seconds,
                       polish_seconds=polish_seconds, check_step=check_step, log=log)

    # RRTConnect returns the first path it finds, and which homotopy class that lands in
    # is luck -- one run goes over the fixture, the next threads behind it.  The
    # optimisation pass afterwards can shorten a route and stand it off, but it cannot move
    # it to the other side of an obstacle, so whichever class arrives here is the one that
    # ships.  Sampling several solutions and keeping the best-scoring one is therefore the
    # only stage that can make that choice at all.
    candidates: list[tuple[float, float, list[np.ndarray]]] = []
    last = ""

    def run(planning_time: float, label: str) -> bool:
        nonlocal last
        found = _ompl_run(cell, qa, qb, segment_length=segment_length,
                          planning_time=planning_time, check_step=check_step,
                          label=label, log=log)
        if found is None:
            last = _ompl_run.message
            return False
        candidates.append(found)
        return True

    if ompl.phase_one_runs > 0:
        log(f"      phase 1: {ompl.phase_one_runs} runs of "
            f"{ompl.phase_one_seconds:g}s each, keeping the cheapest that solves")
        for attempt in range(1, ompl.phase_one_runs + 1):
            run(ompl.phase_one_seconds, f"phase 1 run {attempt}")

    if not candidates and ompl.phase_two_max_runs > 0:
        # Nothing to choose between at this point, so the goal changes from a good route to
        # any route, and the first one that arrives ends the phase.
        log(f"      phase 2: up to {ompl.phase_two_max_runs} runs of "
            f"{ompl.phase_two_seconds:g}s each, stopping at the first solution")
        for attempt in range(1, ompl.phase_two_max_runs + 1):
            if run(ompl.phase_two_seconds, f"phase 2 run {attempt}"):
                break

    if not candidates:
        raise PlanningError(
            f"freespace transit failed after {ompl.worst_case_runs} attempts: {last}")

    cost, plain, raw = min(candidates, key=lambda c: c[0])
    _capture(record, raw)
    if len(candidates) > 1:
        worst = max(c[0] for c in candidates)
        log(f"      keeping the best of {len(candidates)} solutions: cost {cost:.2f} s "
            f"against {worst:.2f} s for the worst")
    # Shortcut before reducing: the dense path gives the cuts somewhere to land.
    out = _finish(cell, raw, zone=zone, shortcut_seconds=shortcut_seconds,
                  polish_seconds=polish_seconds, check_step=check_step, log=log)
    log(f"      reduced to {sum(len(r.states) for r in out)} points in "
        f"{len(out)} run{'' if len(out) == 1 else 's'}")
    return out


# ---------------------------------------------------------------------------
# linear motion near the parts
# ---------------------------------------------------------------------------
PTP = "PTP"
LIN = "LIN"


@dataclass
class Run:
    """A stretch of one path that the robot executes under a single motion type."""
    motion: str                     # PTP | LIN
    states: list[np.ndarray]


@dataclass
class LinearZone:
    """Settings for turning the near-panel parts of a transit into linear motion."""
    near_mm: float = 0.0            # clearance at or under which the route counts as near
    min_run_mm: float = 0.0         # shortest stretch worth converting, in tool travel
    min_run_pct: float = 0.0        # ...or this much of the leg, whichever it meets first
    step_mm: float = 50.0           # sampling along a straight move when it is checked
    linear_speed_mm_s: float = 0.0  # tool speed cap; 0 leaves linear moves costed on joints
    crossing_speed_mm_s: float = 0.0  # costing-only speed for a move that reaches into
                                      # the band from outside it; 0 charges no surcharge

    @property
    def enabled(self) -> bool:
        return self.near_mm > 0.0


def _flatten(runs: list[Run]) -> list[np.ndarray]:
    """The one point list a set of runs describes; consecutive runs share their boundary."""
    if not runs:
        return []
    out = list(runs[0].states)
    for run in runs[1:]:
        out.extend(run.states[1:])
    return out


def _refine_runs(cell: Cell, runs: list[Run], *, zone: "LinearZone | None",
                 shortcut_seconds: float, polish_seconds: float, check_step: float,
                 log) -> list[Run]:
    """Optimise an already-split route and split it again.

    Used where a leg was planned without a refinement budget and is being refined
    afterwards -- the two halves of a transit that changes the gun partway, each of which
    has to be refined with its own opening in force.  The split is redone rather than
    preserved, since the passes move the waypoints and a move's profile follows its ends.
    """
    pts = _flatten(runs)
    if len(pts) < 2:
        return runs
    allowed = zone is not None and _linear_allowed(cell, pts, zone, log=log)
    model = MotionModel(cell, max_step=check_step, zone=zone if allowed else None)
    refined = _refine(model, pts, shortcut_seconds=shortcut_seconds,
                      polish_seconds=polish_seconds, log=log)
    return _split_runs(model, refined)


def _report_work(cell: Cell, before: dict[str, int], elapsed: float, log) -> None:
    """What the passes actually asked of the cell, and how much of it was avoided.

    The budgets are wall clock, so a pass that gets through five candidates in fifty
    seconds has spent them somewhere.  These are the primitives it can have spent them on,
    printed as a difference so each transit reports its own share.
    """
    counters = getattr(cell, "counters", None)
    if not log or not counters or before is None:
        return
    d = {k: v - before.get(k, 0) for k, v in counters.items()}
    asked = d["clearance"] + d["clearance_hits"]
    served = 100.0 * d["clearance_hits"] / asked if asked else 0.0
    log(f"      work: {d['clearance']} clearance queries ({served:.0f}% of {asked} served "
        f"from cache), {d['collision_tests']} collision tests, {d['state_loads']} state "
        f"loads, {d['fk']} forward kinematics, in {elapsed:.1f}s")


def _finish(cell: Cell, path: list[np.ndarray], *, zone: "LinearZone | None",
            shortcut_seconds: float, polish_seconds: float, check_step: float,
            log) -> list[Run]:
    """Densify a freshly planned route, optimise it, then split it by motion type.

    The route arrives as the handful of states the sampling planner happened to stop at,
    so it is filled in first: every pass below wants the route, not the tree's nodes, and
    a shortcut pass given four points has almost nothing to work with.

    Optimisation runs before the split, not after.  The passes are profile aware -- a move
    that will be executed linearly is costed under the tool speed cap and collision
    checked along the straight line the tool will really take -- so they can reshape a
    near-panel stretch without misjudging it.  The old order linearised first precisely
    because the passes could not do that, and then had to leave the result untouched.

    Which profile governs a move is decided per move, from where its two ends sit, so the
    split at the end is a reading of the finished path rather than a decision imposed on
    it.
    """
    before, t0 = getattr(cell, "counters", None), time.time()
    before = dict(before) if before is not None else None
    dense = _densify(cell, path, check_step)
    allowed = zone is not None and _linear_allowed(cell, dense, zone, log=log)
    model = MotionModel(cell, max_step=check_step, zone=zone if allowed else None)
    refined = _refine(model, dense, shortcut_seconds=shortcut_seconds,
                      polish_seconds=polish_seconds, log=log)
    runs = _split_runs(model, refined)
    _report_work(cell, before, time.time() - t0, log)
    if log:
        lin = sum(1 for r in runs if r.motion == LIN)
        if lin:
            moves = sum(len(r.states) - 1 for r in runs if r.motion == LIN)
            log(f"      {lin} linear runs over {moves} moves, "
                f"{len(runs) - lin} joint runs, {len(refined)} waypoints")
    return runs


def _tcp_travel(cell: Cell, states: list[np.ndarray]) -> float:
    """Distance the tool centre point covers along a joint path, in manifest units.

    ``fk`` answers in environment units, which are metres, while every threshold this is
    measured against is quoted in the manifest's own units.  Converting here rather than at
    the caller keeps the two from being compared raw, which read a 1.8 m stretch as 1.8
    against a 250 mm threshold and so failed every length test there was.
    """
    points = [cell.fk(q)[:3, 3] for q in states]
    raw = float(sum(np.linalg.norm(b - a) for a, b in zip(points, points[1:])))
    return raw / cell.man.scale


def _chain_collides(cell: Cell, chain: list[np.ndarray], check_step: float) -> bool:
    """``plan_linear`` clears the points it places; this clears the gaps between them."""
    return any(cell.segment_collides(a, b, max_step=check_step)
               for a, b in zip(chain, chain[1:]))


def _linear_allowed(cell: Cell, dense: list[np.ndarray], zone: "LinearZone",
                    log=None) -> bool:
    """Whether this leg earns linear motion at all, and the report explaining the verdict.

    Being near the panel is measured over the leg rather than over any one stretch of it.
    The points are a collision-check step apart, so the share of them in range is how much
    of the route runs near the panel, whether or not it does so in one go -- which matters
    because the apex of a retract between two welds leaves the band for an instant and
    would otherwise split an 85% leg into runs of 45% and 40% that a 50% threshold rejects
    twice over.

    Either test admits the leg: one stretch long enough in tool travel, or enough of the
    leg in range whatever the stretches look like individually.  ``min_run_mm`` is what a
    transit needs, since a sweeping move that clips the band for a moment is not working
    near the panel; ``min_run_pct`` is what a hop from one weld to the next needs, being
    entirely near the panel and far too short to pass any absolute threshold.

    A leg that meets neither is planned as joint motion throughout.
    """
    if not zone.enabled or len(dense) < 2:
        return False

    if log and len(dense) > 200:
        log(f"      measuring clearance at {len(dense)} points along the route to find "
            f"its near-panel stretches")
    near = [cell.clearance_mm(q) <= zone.near_mm for q in dense]
    in_range = 100.0 * sum(near) / len(near) if near else 0.0
    by_share = zone.min_run_pct > 0.0 and in_range >= zone.min_run_pct

    stretches: list[float] = []
    i = 0
    while i < len(dense):
        if not near[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(dense) and near[j + 1]:
            j += 1
        if j > i:
            stretches.append(_tcp_travel(cell, dense[i:j + 1]))
        i = j + 1

    by_length = any(t >= zone.min_run_mm for t in stretches)
    allowed = bool(by_length or by_share)

    if log and (stretches or in_range > 0.0):
        note = [f"{in_range:.0f}% of the leg in range"]
        if stretches:
            note.append(f"longest stretch {max(stretches):.0f} mm of "
                        f"{sum(stretches):.0f} mm near the panel, against a "
                        f"{zone.min_run_mm:g} mm threshold")
        if allowed:
            why = "on length" if by_length else f"on share against {zone.min_run_pct:g}%"
            note.append(f"near-panel moves run under the linear profile ({why})")
        else:
            note.append("neither threshold met, so the leg stays joint motion throughout")
        log("      " + ", ".join(note))
    return allowed


def _split_runs(model: "MotionModel", pts: list[np.ndarray]) -> list[Run]:
    """Group a finished path into runs of one motion type.

    The type of a move is asked of the model, so it is decided by where the waypoints
    ended up rather than by where they started -- the passes move them.  Consecutive runs
    share their boundary point: it is the last waypoint of one and the first of the next,
    which is what makes a waypoint's own marking the move that arrives at it.
    """
    if len(pts) < 2:
        return [Run(PTP, list(pts))]
    runs: list[Run] = []
    start = 0
    current = model.motion(pts[0], pts[1])
    for k in range(1, len(pts) - 1):
        nxt = model.motion(pts[k], pts[k + 1])
        if nxt != current:
            runs.append(Run(current, pts[start:k + 1]))
            start, current = k, nxt
    runs.append(Run(current, pts[start:]))
    return runs



def validate(cell: Cell, path: list[np.ndarray], max_step: float = 0.05, *,
             motion: str = PTP, zone: "LinearZone | None" = None) -> str | None:
    """Re-check a finished path; returns a description of the first bad move, or None.

    ``motion`` is the profile the phase will be executed under, and it decides which path
    each move is checked along: the joint chord for ``PTP``, the straight line the tool
    really takes for ``LIN``.  Checking a linear move in joint space would be checking a
    curve the robot does not follow, which is the one thing a final validation must not do.
    """
    linear = motion == LIN and zone is not None and zone.enabled
    model = MotionModel(cell, max_step=max_step, zone=zone) if linear else None
    fault = "reachable in a straight line" if linear else "collision free"
    for k, (a, b) in enumerate(zip(path, path[1:])):
        bad = ((not model.linear_ok(a, b)) if linear
               else cell.segment_collides(a, b, max_step=max_step))
        if bad:
            return f"move {k} -> {k + 1} is not {fault}"
    return None


# ---------------------------------------------------------------------------
# linear (Cartesian) motion
# ---------------------------------------------------------------------------
def interpolate_pose(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear in position, shortest-arc slerp in orientation."""
    out = np.eye(4)
    out[:3, 3] = a[:3, 3] + t * (b[:3, 3] - a[:3, 3])
    Ra, Rb = a[:3, :3], b[:3, :3]
    R = Ra.T @ Rb
    angle = float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    if angle < 1e-9:
        out[:3, :3] = Ra
        return out
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis = axis / (2.0 * np.sin(angle))
    th = angle * t
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    out[:3, :3] = Ra @ (np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K))
    return out


def plan_linear(cell: Cell, pose_a: np.ndarray, pose_b: np.ndarray, seed: np.ndarray, *,
                step_mm: float = 50.0, allow_collision: bool = False
                ) -> list[np.ndarray]:
    """Joint states following the straight Cartesian line from ``pose_a`` to ``pose_b``.

    Poses are world 4x4 in manifest units.  Points are spaced by ``step_mm`` -- a FANUC
    L move only needs enough points to pin the line, not a dense sampling.
    """
    dist = float(np.linalg.norm(pose_b[:3, 3] - pose_a[:3, 3]))
    n = max(1, int(np.ceil(dist / max(step_mm, 1e-6))))
    out: list[np.ndarray] = []
    current = np.asarray(seed, dtype=float)
    for k in range(n + 1):
        pose = interpolate_pose(pose_a, pose_b, k / n)
        q = cell.solve_pose(pose, [current], require_collision_free=not allow_collision)
        if q is None:
            raise PlanningError(
                f"no collision-free IK at {100.0 * k / n:.0f}% along a {dist:.0f} mm "
                f"linear move")
        out.append(q)
        current = q
    return out
