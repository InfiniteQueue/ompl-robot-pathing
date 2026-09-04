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
from dataclasses import dataclass, replace

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

    @property
    def tool_step(self) -> float:
        """How far the tool may move between two samples, in manifest units.

        The same figure for both profiles, and deliberately so.  A joint move is checked by
        ``segment_collides``, which subdivides wherever the tool travels further than
        ``Cell.tcp_check_mm`` between samples.  A linear move is checked in two stages --
        stations placed along the Cartesian line, then those same joint gaps between them
        -- and until this was shared the first stage ran on its own, coarser number.  That
        let the finer one be gated by the coarser: an 87 mm move took stations at 0%, 50%
        and 100%, the gaps between them were checked densely on the joint chord, and a
        fixture that the *line* entered from 16% to 34% was never sampled at all.

        Zero means the tool-space criterion is switched off, which this reads literally:
        the stations collapse onto the two ends and a linear move is then checked exactly
        as a joint move is, by ``--check-step-deg`` alone.  That is the same reduction
        turning it off already causes for joint moves, rather than a hidden default that
        would keep a promise the setting says is no longer being made.
        """
        step = float(getattr(self.cell, "tcp_check_mm", 0.0))
        return step if step > 0.0 else float("inf")

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

    def linear_chain(self, a: np.ndarray, b: np.ndarray) -> list[np.ndarray] | None:
        """The states the tool passes through running straight from ``a`` to ``b``.

        ``None`` when it cannot get there: no inverse kinematics somewhere along the line,
        or a collision in the joint gaps between the samples.

        Under a linear profile the controller drives the tool along the straight line and
        solves inverse kinematics as it goes, so that line -- not the joint chord between
        the same two states -- is what has to be reachable and clear.  A joint-space check
        here would be checking a path the robot does not take, and for the long chords the
        shortcut pass proposes the two are nowhere near each other.

        The chain is returned rather than reduced to a verdict because a caller that
        installs the interior of a move needs the points on *this* curve.  Interpolating
        the joint chord instead would verify one path and commit another.
        """
        try:
            chain = plan_linear(self.cell, self._pose_mm(a), self._pose_mm(b), a,
                                step_mm=self.tool_step)
        except PlanningError:
            return None                     # no inverse kinematics somewhere along it
        # Inverse kinematics returns the solution nearest its seed rather than the state
        # asked for, so pin the ends back before checking the gaps between the samples.
        chain[0] = np.asarray(a, dtype=float)
        chain[-1] = np.asarray(b, dtype=float)
        return None if _chain_collides(self.cell, chain, self.max_step) else chain

    def linear_ok(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Can the tool run straight from ``a`` to ``b``?"""
        return self.linear_chain(a, b) is not None

    def clear_path(self, a: np.ndarray, b: np.ndarray) -> list[np.ndarray] | None:
        """The interior points of this move, or ``None`` if the move is unusable.

        :meth:`blocked` with the route kept instead of thrown away, for the one caller that
        installs that route rather than only asking whether the move is allowed.  The
        distinction is the whole point of it: a joint move follows the joint chord, which
        ``_resample`` describes exactly, but a linear move follows the Cartesian line, and
        filling that stretch with joint-space interpolation puts points on a curve the tool
        never passes through.

        Not used by :meth:`blocked` itself, which is asked tens of thousands of times a
        pass and wants a verdict without building a point list to reach it.
        """
        if self.motion(a, b) == PTP:
            if self.cell.segment_collides(a, b, max_step=self.max_step):
                return None
            return _resample(self.cell, a, b, self.max_step)
        chain = self.linear_chain(a, b)
        if chain is None:
            return None
        return [np.asarray(q, dtype=float) for q in chain[1:-1]]

    # -- what it costs ------------------------------------------------------
    def _crosses(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Whether this move reaches into the band from outside it, or back out."""
        return self.zone is not None and self.near(a) != self.near(b)

    def crossing_penalty(self, a: np.ndarray, b: np.ndarray) -> float:
        """A flat surcharge on a move that reaches into the band from outside it.

        Added, not scaled, and the same for every crossing however long the move is.  What
        that buys is a charge that prices the *number* of times a route enters the band
        and nothing else -- so a route that dips in and out repeatedly pays for each dip,
        while a decision that leaves the crossing count alone is left alone.

        That is the whole of it, and it is worth being clear about what it therefore does
        not do, because the setting it replaces tried to do it.  Deleting a waypoint from
        inside the band leaves the crossing count unchanged -- one crossing before, one
        after -- so this cancels exactly and the decision falls back to ordinary time,
        where one fewer stop is one fewer pair of ramps.  That is the intent: such a
        deletion does not move where linear motion begins, which stays at the last
        waypoint outside the band either way, and the pair of near-coincident vias it used
        to leave straddling the edge cost a stop apiece for nothing.

        What still holds the sweep back is the tool speed cap, and it does so for a
        structural reason rather than by being tuned to.  A deletion that grows the sweep
        outward converts a *joint* hop into a linear one, so the cap lands on one side of
        the comparison and not the other and does not cancel; a deletion inside the band
        moves no such boundary, and a straight line is never longer than the polyline it
        replaces, so the cap correctly stays out of it.  The old charge could not tell
        those two apart, because it read the crossing move's total length -- which grows
        under both -- rather than what the length was doing.

        The figure is used for costing and nothing else: the robot is never scheduled by
        it, and it never reaches the exported program.

        Deliberately not scaled by the clearance penalty.  That penalty is a multiplier on
        *time spent* near the parts; a fixed preference is not time, and compounding the
        two would make the charge depend on where the crossing happens to sit rather than
        on the fact that it happened.  Every caller therefore adds this outside its own
        factor arithmetic.
        """
        if self.zone is None or self.zone.crossing_penalty_s <= 0.0:
            return 0.0
        return self.zone.crossing_penalty_s if self._crosses(a, b) else 0.0

    def _time_floor(self, a: np.ndarray, b: np.ndarray) -> float:
        """Seconds this linear move is held to, over and above the joint limits.

        The tool speed cap and nothing else.  It is a prediction -- the same figure
        schedules the exported program -- so it belongs in the move's time, where a
        preference does not: :meth:`crossing_penalty` is added to the cost by the callers
        instead of raising the time here.

        This is also what actually resists a linear sweep growing outward.  Such a growth
        turns a joint hop into a linear one, so the cap applies to the replacement and not
        to what it replaced; a rearrangement wholly inside the band converts nothing, and
        a straight line is never longer than the polyline it replaces, so the cap has
        nothing to say about it.  The discrimination falls out of the profile rule rather
        than being tuned in.
        """
        if self.zone is None or self.motion(a, b) == PTP:
            return 0.0
        if self.zone.linear_speed_mm_s <= 0.0:
            return 0.0
        return _tcp_travel(self.cell, [a, b]) / self.zone.linear_speed_mm_s

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

        The crossing penalty is always left out.  ``_crosses`` compares both ends and so
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

        The crossing penalty is added afterwards rather than folded into the floor, so it
        is not multiplied by the clearance factor: it is a fixed preference about how many
        times the route enters the band, not time spent anywhere.
        """
        penalised = self.cell.segment_cost(a, b, max_step=self.max_step,
                                           fa=fa, fb=fb, stops=stops)
        floor = self._time_floor(a, b)
        if floor > 0.0:
            raw = self.cell.move_time(a, b) if stops else self.cell.cruise_time(a, b)
            penalised = (max(penalised, floor) if raw <= 0.0
                         else penalised * max(1.0, floor / raw))
        return penalised + self.crossing_penalty(a, b)

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
        # The crossing penalty is added outside the factor, exactly as ``cost`` adds it to
        # the chord: charged on the same footing on both sides or it would not cancel when
        # the two arrangements cross the band the same number of times, which is the one
        # thing it is supposed to do.
        return sum(model.move_time(path[k], path[k + 1]) * max(factor(k), factor(k + 1))
                   + model.crossing_penalty(path[k], path[k + 1])
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


@dataclass
class Relocation:
    """How ``polish`` draws a relocation: how far, shaped how, and how many at least.

    Carried on this object rather than threaded as its own argument because it reaches
    every caller of ``polish`` already, and because it describes the same sampling loop
    the distances do.

    Distances are approximate tool travel in the manifest's units, not joint angles: the
    step is normalised through ``Cell.weights``, which is TCP travel per radian measured
    at the start pose, so a draw of 30 means "move the tool about 30 mm" whichever joints
    happen to carry it.

    ``exponent`` shapes the draw between the two, as ``min + (max - min) * x ** e`` for x
    uniform on [0, 1).  At 1 that is the flat draw this pass has always used, where a 5 mm
    nudge and a 150 mm shove are equally likely.  Above 1 it crowds towards ``min``, which
    is what a route already near its answer wants: most attempts then probe around the
    point instead of throwing it across the cell.  Below 1 it crowds towards ``max``.
    """
    min_mm: float = 5.0
    max_mm: float = 150.0
    exponent: float = 1.0
    min_attempts: int = 20          # polish only; shortcut's loop is timed alone

    def draw(self, scale: float) -> tuple[float, float, float]:
        """``(low, span, exponent)`` in scene units, ready for the sampling loop."""
        lo = max(self.min_mm, 0.0) * scale
        hi = max(self.max_mm, self.min_mm) * scale
        return lo, hi - lo, max(self.exponent, 1e-6)


def polish(model: "MotionModel", path: list[np.ndarray], *, time_budget: float = 5.0,
           rng: np.random.Generator | None = None,
           relocate: Relocation | None = None,
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
      removal sweep follows every accepted relocation.  How far it displaces, and how
      that distance is drawn between the bounds, is ``relocate``.

    ``simplify`` already applies the same removal test greedily, so this is a second,
    non-greedy opinion on it rather than the first -- what is new here is the relocation,
    and the sweep that follows it.  Every replacement is collision checked before it is
    kept, so the result stays traversable.

    ``Relocation.min_attempts`` is a floor on the sampling loop that outlasts the clock:
    the pass keeps drawing until it has made that many attempts however long they take.
    A relocation is a random draw, so a budget that expires early does not return a route
    judged and kept -- it returns one barely sampled, and on a slow leg, where each
    attempt costs a collision check along a Cartesian line, that is exactly where the
    clock runs out first.  The floor is what stops the pass being quietly skipped on the
    legs that most need it.

    What it does not override is a caller declining the pass.  ``time_budget <= 0`` is not
    a clock that ran out; it is ``--no-shortcut``, or a leg inside the two-leg fallback
    search that is deferring its refinement to ``_refine_runs`` and may yet be discarded.
    Those return untouched, and the leg that is kept meets the floor when it is really
    refined.
    """
    if len(path) < 3 or time_budget <= 0:
        return [np.asarray(p, dtype=float).copy() for p in path]

    cell = model.cell
    rng = rng or np.random.default_rng(1)
    # Hoisted out of the loop: this draws tens of thousands of times against a budget
    # measured in seconds, and a flat draw should not pay for a power that does nothing.
    relocate = relocate or Relocation()
    low, span, exponent = relocate.draw(cell.man.scale)
    floor = max(0, int(relocate.min_attempts))
    flat = abs(exponent - 1.0) < 1e-9
    pts = [np.asarray(p, dtype=float).copy() for p in path]
    fac = [model.penalty_factor(p) for p in pts]
    costs = [_hop(model, pts[i], pts[i + 1], fac[i], fac[i + 1])
             for i in range(len(pts) - 1)]
    before = sum(costs)
    deadline = time.time() + time_budget
    dropped = moved = tried = 0

    def unfinished() -> bool:
        """Whether the pass may keep working: the floor outranks the clock."""
        return tried < floor or time.time() < deadline

    def drop_sweep() -> None:
        nonlocal dropped
        k = 1
        # Under the same rule as the sampling loop, so that a relocation accepted past
        # the deadline still gets the removal sweep the pass promises follows every one.
        while k < len(pts) - 1 and unfinished():
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
    while unfinished() and len(pts) > 2:
        tried += 1
        k = int(rng.integers(1, len(pts) - 1))
        # Drawn in joint space but scaled through the cell's joint weights, so an attempt
        # moves the tool about as far whichever joints it happens to use.
        direction = rng.normal(size=len(pts[k]))
        reach = float(np.linalg.norm(direction * cell.weights))
        if reach <= 0.0:
            continue
        x = float(rng.random())
        step = low + span * (x if flat else x ** exponent)
        candidate = np.clip(pts[k] + direction * (step / reach),
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
        over = time.time() - deadline
        held = (f", held past its {time_budget:g}s budget by {over:.1f}s to reach the "
                f"{floor}-attempt floor" if over > 0.05 and floor else "")
        log(f"      polish: dropped {dropped} and relocated {moved} waypoints from "
            f"{tried} attempts, penalised time {before:.2f} -> {after:.2f} s "
            f"({gain:.0f}% better), {len(pts)} points{held}")
    return pts


def _refine(model: "MotionModel", path: list[np.ndarray], *, shortcut_seconds: float,
            polish_seconds: float, relocate: "Relocation | None" = None,
            log=print) -> list[np.ndarray]:
    """The whole post-processing chain, in the order the three passes need to run.

    Shortcutting reshapes the route while it is still dense, reduction picks which of those
    points are actually worth stopping at, and polishing then judges those stops under the
    time they really cost.

    Both budgets are zero on a leg planned inside the two-leg fallback search, which is
    trying gun openings in pairs and would otherwise refine legs it is about to discard.
    Such a leg is refined once a pair is settled on, by ``_refine_runs``, so the pass is
    deferred rather than skipped -- and it says so, since "0s, both spent in full" reads
    as a budget that was consumed.
    """
    if shortcut_seconds <= 0 and polish_seconds <= 0:
        log(f"      leaving {len(path)} points unrefined for now: no budget at this "
            f"stage, and the route may yet be discarded")
    else:
        log(f"      refining {len(path)} points: up to {shortcut_seconds:g}s shortcutting "
            f"then {polish_seconds:g}s polishing, both spent in full")
    improved = shortcut(model, path, time_budget=shortcut_seconds, relocate=relocate,
                        log=log)
    reduced = simplify(model, improved)
    return polish(model, reduced, time_budget=polish_seconds, relocate=relocate,
                  log=log)


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
             relocate: Relocation | None = None,
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
        # Outside the factor, on the same footing as everywhere else it is charged.
        return model.cruise_time(x, y) * max(fx, fy) + model.crossing_penalty(x, y)

    def span_cost(i: int, j: int) -> float:
        return sum(step_cost(dense[k], dense[k + 1], fac[k], fac[k + 1])
                   for k in range(i, j))

    rng = rng or np.random.default_rng(0)
    draw = (relocate or Relocation()).draw(model.cell.man.scale)
    before = span_cost(0, len(dense) - 1)
    deadline = time.time() + time_budget
    tried = cuts = moves = 0

    while time.time() < deadline and len(dense) > 2:
        tried += 1
        if rng.random() < 0.5:
            cuts += _try_cut(model, dense, fac, rng, penalised, span_cost)
        else:
            moves += _try_relocate(model, dense, fac, rng, penalised, step_cost, draw)

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
    # The interior is asked of the model rather than interpolated here.  Under a linear
    # profile the check above runs along the Cartesian line, and joint-space points would
    # not lie on it: the pass would verify one curve and install another, leaving moves
    # that claim to be straight lines nobody ever checked.
    points = model.clear_path(dense[i], dense[j])
    if points is None:
        return 0
    if penalised:
        factors = [model.penalty_factor(p) for p in points]
        chain = [dense[i]] + points + [dense[j]]
        chain_f = [fac[i]] + factors + [fac[j]]
        direct = sum(model.cruise_time(x, y) * max(fx, fy)
                     + model.crossing_penalty(x, y)
                     for x, y, fx, fy in zip(chain, chain[1:], chain_f, chain_f[1:]))
        if direct >= span - 1e-9:
            return 0
    else:
        factors = [1.0] * len(points)
    dense[i + 1:j] = points
    fac[i + 1:j] = factors
    return 1


def _try_relocate(model: "MotionModel", dense, fac, rng, penalised, step_cost,
                  draw: tuple[float, float, float] | None = None) -> int:
    """Displace one interior waypoint and keep the move if it lowers the local cost.

    The displacement is drawn in joint space but scaled by the cell's joint weights, so a
    given attempt moves the tool about as far whichever joints it uses -- otherwise almost
    every sample would be a wrist twiddle that changes nothing.

    ``draw`` is ``Relocation.draw``'s ``(low, span, exponent)``, already in scene units;
    the caller hoists it out of its own loop.
    """
    if len(dense) < 3:
        return 0
    cell = model.cell
    k = int(rng.integers(1, len(dense) - 1))        # endpoints are fixed by the caller
    low, span, exponent = draw or Relocation().draw(cell.man.scale)
    x = float(rng.random())
    target = low + span * (x if exponent == 1.0 else x ** exponent)
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

    def capped(self, runs: int) -> "OmplBudget":
        """This budget with each phase held to at most ``runs`` attempts.

        For the work a transit only reaches once its preferred answer has failed.  Phase
        one's expense buys a choice between homotopy classes, and that is worth paying on
        the route the transit is most likely to ship; by the third gun opening, or by a
        leg of a two-leg split that exists only because nothing else reached at all, the
        question has already collapsed to whether there is a route.  Spending the full
        budget there costs the great majority of the worst case and decides very little.

        ``runs <= 0`` leaves the budget alone, which is the old exhaustive behaviour.
        """
        if runs <= 0:
            return self
        return replace(self, phase_one_runs=min(self.phase_one_runs, runs),
                       phase_two_max_runs=min(self.phase_two_max_runs, runs))


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
                     via: np.ndarray | None,
                     shortcut_seconds: float, polish_seconds: float,
                     zone: LinearZone | None = None,
                     cartesian: "CartesianBudget | None" = None,
                     relocate: "Relocation | None" = None, log=print,
                     record: list | None = None) -> list[Run]:
    """Collision-free joint path from ``qa`` to ``qb`` at the gun's current opening.

    ``via`` is the fallback pose to route through, or ``None`` for the direct transit.
    Which of the two is wanted is the caller's decision rather than this function's:
    ``plan_freespace`` works through every gun opening directly before it works through
    any of them via a fallback pose, so that the cheap answer is exhausted everywhere
    before the expensive one is started anywhere.

    Routing through a via is what a robot programmer would do by hand -- it turns a long
    detour around the panels into two easy problems -- but it costs two full solves rather
    than one, which is why it is not tried until directness has been given up on.
    """
    blocked = _endpoint_block(cell, qa, qb)
    if blocked:
        # Nothing downstream can rescue this: every route at this opening ends here,
        # whether or not it goes by way of a via.  The caller's next candidate opening, or
        # the two-leg split, is the only way on.
        raise PlanningError(blocked)

    if via is None:
        return _plan_direct(cell, qa, qb, ompl=ompl,
                            segment_length=segment_length, check_step=check_step,
                            shortcut_seconds=shortcut_seconds,
                            polish_seconds=polish_seconds,
                            zone=zone, cartesian=cartesian,
                            relocate=relocate, log=log, record=record)

    if cell.in_collision(via):
        raise PlanningError("the fallback pose is in collision with the gun at this "
                            "opening")
    # The halves are recorded jointly below: this is still one leg of the output, and the
    # fallback pose is an implementation detail of how it was found.
    halves: list[list[np.ndarray]] = []
    # Planned without a linear zone: the two legs are joined below and the whole route is
    # split afterwards, so splitting each half here would put a phase boundary at the
    # fallback pose whether the geometry called for one or not.
    first = _plan_direct(cell, qa, via, ompl=ompl,
                         segment_length=segment_length, check_step=check_step,
                         shortcut_seconds=0.0, polish_seconds=0.0,
                         log=log, record=halves)
    second = _plan_direct(cell, via, qb, ompl=ompl,
                          segment_length=segment_length, check_step=check_step,
                          shortcut_seconds=0.0, polish_seconds=0.0,
                          log=log, record=halves)
    if len(halves) == 2:
        _capture(record, halves[0] + halves[1][1:])
    # Refine the joined route rather than each leg: the detour through the fallback
    # pose is exactly the kind of corner these passes exist to cut.
    joined = first[0].states + second[0].states[1:]
    return _finish(cell, joined, zone=zone, relocate=relocate,
                   shortcut_seconds=shortcut_seconds,
                   polish_seconds=polish_seconds, check_step=check_step, log=log)


def plan_freespace(cell: Cell, qa: np.ndarray, qb: np.ndarray, *,
                   ompl: OmplBudget | None = None, segment_length: float = 0.02,
                   check_step: float = 0.05,
                   fallback_via=None,
                   fallback_runs: int = 0,
                   shortcut_seconds: float = 2.0, polish_seconds: float = 5.0,
                   planning_time: float = DEFAULT_PLANNING_TIME,
                   openings: list[float] | None = None,
                   extra_openings: int = 0, opening_round_mm: float = 0.0,
                   zone: LinearZone | None = None,
                   cartesian: "CartesianBudget | None" = None,
                   relocate: "Relocation | None" = None,
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

    ``fallback_runs`` caps the sampling-planner budget for everything past the first
    attempt at the preferred opening -- see ``OmplBudget.capped``.  That tail is where
    nearly all of the worst case sits, and it is also where the full budget buys least.

    ``fallback_via`` is either the poses to detour through or a zero-argument callable
    returning them.  The callable form exists because finding them is now a search of its
    own -- see :mod:`weldpath.fallback` -- and most transits solve at the first opening and
    never need one.  It is called at most once, and only after every opening has been tried
    directly, so a transit that solves pays nothing for the search it did not use.
    """
    ompl = ompl or OmplBudget()
    reduced = ompl.capped(fallback_runs)

    def attempt(opening, a, b, budget, into=None, effort=None, via=None):
        with cell.gun_opening(opening):
            return _plan_at_opening(cell, a, b, ompl=effort or ompl,
                                    segment_length=segment_length,
                                    check_step=check_step, via=via,
                                    shortcut_seconds=budget,
                                    polish_seconds=polish_seconds if budget else 0.0,
                                    zone=zone, cartesian=cartesian,
                                    relocate=relocate, log=log, record=into)

    candidates = _opening_candidates(cell, openings, extra_openings,
                                     opening_round_mm)

    # One opening for the whole transit, in preference order: changing the gun is a real
    # operation on the machine, so it is a last resort rather than a free parameter.
    #
    # Directness is the outer question and the gun opening the inner one.  Every opening is
    # tried as a single move before any of them is tried as a detour, because a detour
    # costs two full solves against one and the cheapest thing that can still work should
    # be exhausted first.  The old order asked both questions at once -- each opening
    # direct, then immediately that same opening through every via -- which spent the
    # whole two-leg budget at the preferred opening before so much as looking at the next.
    resolved: list | None = None

    def vias() -> list:
        """The fallback poses, found on first use and remembered."""
        nonlocal resolved
        if resolved is None:
            got = fallback_via() if callable(fallback_via) else fallback_via
            resolved = list(got or [])
        return resolved

    def routes():
        """``None`` for the direct route, then one entry per fallback pose.

        A generator rather than a list so that ``vias()`` is not reached until the direct
        route has been tried at every opening and failed.
        """
        yield None
        yield from vias()

    last = None
    for v, via in enumerate(routes()):
        if v:
            log(f"      retrying via fallback pose {v}/{len(vias())}")
        for n, opening in enumerate(candidates):
            preferred = v == 0 and n == 0
            try:
                if not preferred:
                    log(f"      retrying{_gun_note(opening)}"
                        f"{_effort_note(reduced, ompl)}")
                raw: list = []
                leg = attempt(opening, qa, qb, shortcut_seconds, raw,
                              effort=ompl if preferred else reduced, via=via)
                if record is not None:
                    record.extend(raw)
                return [(leg, opening)]
            except PlanningError as exc:
                # Said out loud because the endpoint screen rejects an opening in
                # microseconds and would otherwise pass in silence, where a failed OMPL run
                # announces itself at length.  Both reach the same place: this opening is
                # not the one.
                log(f"      no route{_gun_note(opening)}: {exc}")
                last = exc

    if len(candidates) < 2 or not vias():
        raise last or PlanningError("freespace transit failed")

    # No single opening reaches: split the move and change the gun partway, at a pose the
    # robot is already passing through and stationary at.
    for i, mid in enumerate(vias()):
        for first_open in candidates:
            with cell.gun_opening(first_open):
                if cell.in_collision(mid):
                    continue
            raw_first: list = []
            try:
                first = attempt(first_open, qa, mid, 0.0, raw_first, effort=reduced,
                                via=None)
            except PlanningError:
                continue
            for second_open in candidates:
                if abs(second_open - first_open) < OPENING_TOL_MM:
                    continue                    # already ruled out as a single opening
                raw_second: list = []
                try:
                    second = attempt(second_open, mid, qb, 0.0, raw_second,
                                     effort=reduced, via=None)
                except PlanningError:
                    continue
                if record is not None:
                    record.extend(raw_first + raw_second)
                log(f"      no single gun opening reaches; changing from "
                    f"{first_open:g} mm to {second_open:g} mm at fallback pose "
                    f"{i + 1}/{len(vias())}")
                with cell.gun_opening(first_open):
                    first = _refine_runs(cell, first, zone=zone, relocate=relocate,
                                         shortcut_seconds=shortcut_seconds,
                                         polish_seconds=polish_seconds,
                                         check_step=check_step, log=log)
                with cell.gun_opening(second_open):
                    second = _refine_runs(cell, second, zone=zone, relocate=relocate,
                                          shortcut_seconds=shortcut_seconds,
                                          polish_seconds=polish_seconds,
                                          check_step=check_step, log=log)
                return [(first, first_open), (second, second_open)]
    raise last or PlanningError("freespace transit failed at every gun opening")


def _gun_note(opening: float | None) -> str:
    """" with the gun at 40 mm", or nothing at all where the cell has no gun joint.

    ``_opening_candidates`` returns ``[None]`` for a cell without one, and there is no
    opening to name in that case -- naming one anyway raised a TypeError out of the format
    string, turning a transit that merely failed into a crash.
    """
    return "" if opening is None else f" with the gun at {opening:g} mm"


def _effort_note(reduced: OmplBudget, full: OmplBudget) -> str:
    """How the reduced budget differs, for the log line that announces a retry."""
    if reduced is full:
        return ""
    return (f", at up to {reduced.phase_one_runs} run"
            f"{'' if reduced.phase_one_runs == 1 else 's'} per phase")


# Two openings closer together than this are the same command as far as the machine is
# concerned: the gun is a mechanism with backlash, not a number.  Anything nearer is a
# rounding artefact -- widest/2 landing on a declared opening, or a bisection rounding onto
# its own neighbour -- and trying it twice buys a second identical failure at full price.
OPENING_TOL_MM = 1e-6


def unique_openings(values: list[float], limit: float | None = None) -> list[float]:
    """``values`` in order, without repeats, optionally clamped to ``0 .. limit``.

    Every list of openings that gets walked goes through here, so that "have we tried this
    one already" is answered the same way everywhere.  Order is preserved because these
    lists are preference orders: the first survivor of a duplicate pair is the one whose
    reason for being tried came first.
    """
    out: list[float] = []
    for value in values:
        value = float(value)
        if limit is not None:
            value = min(max(value, 0.0), limit)
        if not any(abs(value - seen) < OPENING_TOL_MM for seen in out):
            out.append(value)
    return out


def _bisect_openings(out: list[float], extra: int, round_mm: float,
                     widest: float) -> None:
    """Append ``extra`` further openings, each halving the widest untried gap so far.

    The named openings answer where the gun has to be; these answer where else it might
    usefully be, and there is no geometry here to reason from -- the tip is 200 mm of
    swing and which openings clear a fixture is not a function of the number.  So the
    openings are chosen to cover the range rather than to be individually plausible: take
    the widest stretch nothing has been tried in and try its middle, which is the choice
    that leaves the largest remaining hole as small as possible.

    Rounding to ``round_mm`` is what keeps the values sayable on the shop floor -- 45 mm
    rather than 43.7 mm -- at the cost of the split being slightly off centre.  A rounded
    value that lands on an opening already in the list is dropped and the next widest gap
    taken instead, so the count is a ceiling: a coarse rounding over a narrow range runs
    out of distinct openings before it runs out of attempts.
    """
    for _ in range(max(extra, 0)):
        known = sorted(out)
        gaps = sorted(zip(known, known[1:]), key=lambda g: g[1] - g[0], reverse=True)
        for lo, hi in gaps:
            mid = (lo + hi) / 2.0
            if round_mm > 0.0:
                mid = round(mid / round_mm) * round_mm
            mid = float(min(max(mid, 0.0), widest))
            if any(abs(mid - seen) < OPENING_TOL_MM for seen in out):
                continue                       # rounded onto a neighbour: try a wider gap
            out.append(mid)
            break
        else:
            return                             # every gap is narrower than the rounding


def _opening_candidates(cell: Cell, openings: list[float] | None, extra: int = 0,
                        round_mm: float = 0.0) -> list[float | None]:
    """Openings to try for a transit, most preferred first and without duplicates.

    ``openings`` are the two the transit's own ends ask for, departure first.  After them
    come closed, widest and half open -- the three that between them cover the useful
    shapes of the gun -- and then ``extra`` further openings bisecting whatever range is
    left, as described in ``_bisect_openings``.
    """
    if not cell.gun_joint_name:
        return [None]
    widest = cell.man.gun_opening_max
    wanted = list(openings or [])
    wanted += [0.0, widest, widest / 2.0]
    out = unique_openings(wanted, widest)
    _bisect_openings(out, extra, round_mm, widest)
    return out


def _route_fault(cell: Cell, path: list[np.ndarray], check_step: float) -> str | None:
    """The first move of a raw planner route that is not clear here, or ``None``.

    A sampling planner validates its own route, but not under this criterion.  OMPL walks
    a uniform joint-space grid at ``--segment-length-rad``; ``segment_collides`` walks the
    same kind of grid and then bisects wherever the tool moves further than
    ``--check-step-mm``, which where the arm is long is several times finer.  So a route
    can be valid to the planner and blocked here.

    Nothing downstream re-establishes the difference.  ``_path_cost`` scores the route with
    ``segment_cost``, which says outright that it assumes the segment already known clear,
    and the refinement passes only ever check the moves they propose -- so a blocked
    stretch that no cut or relocation happened to land on is refined, split and shipped,
    and is caught, if at all, only by ``_verify_runs`` at the very end.

    Rejecting it here costs one sweep and hands the failure to the retries the caller
    already has: another run of phase one, phase two, the Cartesian tree, or the next gun
    opening.  A run that solves into a blocked route has not solved.
    """
    for k, (a, b) in enumerate(zip(path, path[1:])):
        if cell.segment_collides(a, b, max_step=check_step):
            return f"move {k} -> {k + 1} of {len(path)} points is blocked"
    return None


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
    fault = _route_fault(cell, raw, check_step)
    if fault is not None:
        _ompl_run.message = (f"returned a route that is not clear under this cell's own "
                             f"check ({fault})")
        log(f"      {label}: {_ompl_run.message} ({dt:.1f}s)")
        return None
    cost, plain = _path_cost(cell, raw, check_step)
    log(f"      {label}: solved in {dt:.1f}s ({len(raw)} raw points, "
        f"cost {cost:.2f} s against {plain:.2f} s unpenalised)")
    return cost, plain, raw


_ompl_run.message = ""


def _plan_direct(cell: Cell, qa: np.ndarray, qb: np.ndarray, *, ompl: OmplBudget,
                 segment_length: float, check_step: float,
                 shortcut_seconds: float, polish_seconds: float,
                 zone: LinearZone | None = None,
                 cartesian: "CartesianBudget | None" = None,
                 relocate: "Relocation | None" = None, log=print,
                 record: list | None = None) -> list[Run]:
    if not cell.segment_collides(qa, qb, max_step=check_step):
        # A clear straight line is normally the best answer there is, and a sampling
        # planner asked to improve on it would only return it again.  But "clear" and
        # "sensible" part company when the line grazes a panel, so when the penalty says
        # this one does, the same pass that stands other routes off is given a chance to
        # bow it away -- there is nothing for OMPL to do here, but plenty for relocation.
        def straight() -> list[Run]:
            return _finish(cell, _capture(record, [qa, qb]), zone=zone, relocate=relocate,
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
        return _finish(cell, [qa, qb], zone=zone, relocate=relocate,
                       shortcut_seconds=shortcut_seconds,
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

    if not candidates and cartesian is not None and cartesian.enabled \
            and zone is not None and zone.enabled:
        # Between the two phases rather than in place of either.  Every route this finds
        # is also a joint-space route -- a path of straight tool moves is still a path
        # through joint space -- so as a fallback it searches a strict *subset* of what
        # phase two searches and cannot stand in for it.  What it has instead is guidance:
        # steering along the tool's own line, with orientations drawn near the endpoints',
        # concentrates the search into the corridor beside the panel, which is exactly
        # where uniform joint sampling spends its whole budget and finds nothing.
        #
        # It goes before phase two because phase two is the expensive half of the worst
        # case, so a transit this solves is one whose failure never has to be paid for.
        # It is gated on the zone because a route made of straight moves only earns its
        # cost where linear motion was wanted in the first place; with no band in force
        # there is nothing here that phase two would not do better.
        from .cartesian import plan_cartesian    # deferred: cartesian.py reads this module
        log(f"      cartesian tree: up to {cartesian.seconds:g}s searching linear space "
            f"for a solution")
        t0 = time.time()
        try:
            route = plan_cartesian(cell, qa, qb, max_step=check_step,
                                   budget=cartesian, log=log)
        except PlanningError as exc:
            log(f"      {exc} ({time.time() - t0:.1f}s)")
        else:
            cost, plain = _path_cost(cell, route, check_step)
            log(f"      cartesian tree: solved in {time.time() - t0:.1f}s "
                f"({len(route)} points, cost {cost:.2f} s against {plain:.2f} s "
                f"unpenalised)")
            candidates.append((cost, plain, route))

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
    out = _finish(cell, raw, zone=zone, relocate=relocate,
                  shortcut_seconds=shortcut_seconds,
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
    linear_speed_mm_s: float = 0.0  # tool speed cap; 0 leaves linear moves costed on joints
    crossing_penalty_s: float = 0.0  # flat costing-only surcharge on each move that
                                     # reaches into the band from outside it; 0 charges none

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
                 relocate: "Relocation | None" = None, log=print) -> list[Run]:
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
                      polish_seconds=polish_seconds, relocate=relocate, log=log)
    runs = _split_runs(model, refined)
    _verify_runs(model, runs, "refined leg", log=log)
    return runs


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
            relocate: "Relocation | None" = None, log=print) -> list[Run]:
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
                      polish_seconds=polish_seconds, relocate=relocate, log=log)
    runs = _split_runs(model, refined)
    _verify_runs(model, runs, "planned route", log=log)
    _report_work(cell, before, time.time() - t0, log)
    if log:
        lin = sum(1 for r in runs if r.motion == LIN)
        if lin:
            moves = sum(len(r.states) - 1 for r in runs if r.motion == LIN)
            log(f"      {lin} linear runs over {moves} moves, "
                f"{len(runs) - lin} joint runs, {len(refined)} waypoints")
    return runs


def _verify_runs(model: "MotionModel", runs: list[Run], where: str, log=None) -> None:
    """Check a finished route along the path each of its moves will really take.

    The optimisation passes check the moves they *propose*, and only those.  ``shortcut``
    never offers an adjacent pair -- ``_try_cut`` returns early on ``j - i < 2`` -- and
    ``simplify``'s reach loop runs ``while j > i + 1``, so the pair it finally settles on
    is appended without a check.  A move that came out of ``_densify`` and that nothing
    happened to replace therefore leaves here carrying only the guarantee the route
    arrived with.

    That guarantee is weaker than this module's.  A sampling planner validates a uniform
    joint-space grid at ``--segment-length-rad``; ``segment_collides`` lays down the same
    kind of grid and then bisects it again wherever the tool travels further than
    ``--check-step-mm``, which where the arm is long is several times finer.  So a route
    can be valid to the planner and blocked here.

    Both profiles are swept, each along the path it will really take: the Cartesian line
    for ``LIN``, the joint chord for ``PTP``.  Sweeping only the linear runs, as this once
    did, did not make the joint runs sound -- it only meant nobody had looked at them, and
    a blocked move landing in one left no trace at all.

    This is the check ``toolpath.validate`` makes at the end of the run, brought forward to
    where it can still be acted on.  Raised from ``_finish`` it is a ``PlanningError`` like
    any other and the caller retries at the next gun opening or through a fallback pose;
    raised at the end it ends the run.
    """
    for run in runs:
        linear = run.motion == LIN
        kind = "linear" if linear else "joint"
        for k, (a, b) in enumerate(zip(run.states, run.states[1:])):
            if not model.blocked(a, b):
                continue
            fault = (_linear_fault(model, a, b) if linear
                     else _joint_fault(model, a, b))
            how = ("in a straight line" if linear else "as a joint move")
            if log:
                log(f"      ! the finished route is not traversable {how} "
                    f"at move {k} -> {k + 1} of a {kind} run; discarding it")
                log(f"        {fault}")
            raise PlanningError(
                f"{where}: a {kind} run is not traversable at move {k} -> {k + 1} "
                f"({fault})")


def _joint_fault(model: "MotionModel", a: np.ndarray, b: np.ndarray) -> str:
    """Why this joint move is blocked, and whether a coarser check could have seen it.

    ``segment_collides`` is two tests at once: a uniform grid at ``--check-step-deg``, and
    a bisection of that grid wherever the tool moves further than ``--check-step-mm``
    between samples.  A move can pass the first and fail the second, and that is exactly
    the move a sampling planner hands over believing it valid -- OMPL checks a uniform
    joint-space grid and nothing else.

    Saying which of the two tripped separates a route that was never clear from one this
    module is merely the first to look at closely enough, and those want different fixes.
    """
    cell = model.cell
    travel = _tcp_travel(cell, [a, b])
    span = float(np.max(np.abs(np.asarray(b, dtype=float) - np.asarray(a, dtype=float))))
    head = (f"{travel:.0f} mm of tool travel over {np.degrees(span):.1f} deg of joint "
            f"motion, against a {model.tool_step:g} mm tool step")
    if _grid_blocked(cell, a, b, model.max_step):
        return f"{head}; blocked on the joint grid alone, so the route was never clear"
    return (f"{head}; clear on the joint grid alone -- only the tool-space subdivision "
            f"sees it, and that is finer than the sampling planner ever checked")


def _grid_blocked(cell: Cell, a: np.ndarray, b: np.ndarray, max_step: float) -> bool:
    """The uniform joint-space half of ``segment_collides``, without the tool bisection.

    Mirrors that method's base grid rather than calling it with the tool step turned off,
    which would mean writing to a field the whole cell shares.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    delta = b - a
    n = max(2, int(np.ceil(float(np.max(np.abs(delta))) / max_step)) + 1)
    return any(cell.in_collision(a + t * delta) for t in np.linspace(0.0, 1.0, n))


def _linear_fault(model: "MotionModel", a: np.ndarray, b: np.ndarray) -> str:
    """Why this move is not traversable in a straight line, said in enough detail to act on.

    "not traversable" covers three quite different faults, and they want different fixes.
    The move may be too coarse to have been checked as a line in the first place: below
    ``--check-step-mm`` of tool travel ``plan_linear`` places no station between the ends,
    so the linear test reduces to the joint one and the two cannot disagree -- a refusal
    therefore means the step carried the tool further than that, which is a property of
    how the route was spaced rather than of the geometry.  The line may have no inverse
    kinematics somewhere along it, which is a reachability problem.  Or it may be clear at
    every station and blocked in a gap between two of them, which is geometry.

    Whether the joint chord is clear separates the last case again: a chord that is also
    blocked means the route was never sound, while a clear one means the two curves have
    parted company over this step, and that is what densifying more finely would fix.

    Only ever called on a move that has already failed, so it re-does the work the verdict
    came from.  That is one repeat per discarded route, against the difference between
    knowing a route failed and knowing why.
    """
    cell = model.cell
    pa, pb = model._pose_mm(a), model._pose_mm(b)
    travel = float(np.linalg.norm(pb[:3, 3] - pa[:3, 3]))
    step = model.tool_step
    chord = ("clear" if not cell.segment_collides(a, b, max_step=model.max_step)
             else "also blocked")
    head = (f"{travel:.0f} mm of tool travel against a {step:g} mm linear step, "
            f"joint chord {chord}")
    try:
        chain = plan_linear(cell, pa, pb, a, step_mm=step)
    except PlanningError as exc:
        return f"{head}; {exc}"
    chain[0] = np.asarray(a, dtype=float)
    chain[-1] = np.asarray(b, dtype=float)
    gap = next((j for j, (x, y) in enumerate(zip(chain, chain[1:]))
                if cell.segment_collides(x, y, max_step=model.max_step)), None)
    if gap is None:                     # the verdict has changed under us; say so plainly
        return f"{head}; no fault found on a second look"
    return (f"{head}; clear at all {len(chain)} stations but blocked between "
            f"{gap} and {gap + 1}")


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
