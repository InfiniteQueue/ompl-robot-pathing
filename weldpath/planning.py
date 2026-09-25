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
from dataclasses import dataclass, field, replace

import numpy as np

from tesseract_robotics import tesseract_command_language as cl
from tesseract_robotics.tesseract_common import ProfileDictionary
from tesseract_robotics.tesseract_motion_planners import PlannerRequest
from tesseract_robotics.tesseract_motion_planners_ompl import (
    OMPLMotionPlanner, OMPLRealVectorMoveProfile)

from . import stagetrace
from .cell import STOP_BAND_JOINT, Cell

try:                                        # not in every build of the bindings
    from tesseract_robotics.tesseract_collision import (
        CollisionEvaluatorType_LVS_CONTINUOUS as _LVS_CONTINUOUS)
except ImportError:                         # pragma: no cover - binding dependent
    _LVS_CONTINUOUS = None

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
        # The same, read against the wider introduce limit rather than the band.
        self._reach: dict[bytes, bool] = {}
        # Replacements turned away for reaching in from outside the limit, so a run can
        # say whether the rule bit at all rather than leaving it to be inferred.
        self.overreach_refusals = 0

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
            hit = _reads_near(self.cell, q, self.zone.near_mm)
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

    def within_reach(self, q: np.ndarray) -> bool:
        """Whether a linear move newly introduced out at ``q`` is allowed to start there.

        The same measurement ``near`` makes, read against ``--linear-introduce-mm``
        instead of the band, and cached the same way.  A saturated reading is "nothing
        within the probe", which is further out than the limit and so not within reach --
        the probe is sized past the limit by ``ToolpathPlanner`` for exactly that reason.

        This is deliberately a yes-or-no and not a distance.  Beyond the probe every state
        reads the same, so a rule that ranked one overreach against another would be
        comparing two readings that are both just "out of range".
        """
        if self.zone is None or self.zone.reach_mm <= 0.0:
            return True
        key = np.asarray(q, dtype=float).tobytes()
        hit = self._reach.get(key)
        if hit is None:
            hit = _reads_near(self.cell, q, self.zone.reach_mm)
            self._reach[key] = hit
        return hit

    def _overreaches(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Whether this one move is linear motion reaching in from outside the limit."""
        return (self.motion(a, b) == LIN
                and not (self.within_reach(a) and self.within_reach(b)))

    def overreaches(self, a: np.ndarray, b: np.ndarray, replaced) -> bool:
        """Would replacing ``replaced`` with a->b start linear motion too far out?

        The counterpart to :meth:`demotes`, and needed for the same reason read the other
        way round.  The endpoint rule makes a move linear when *either* end is in the
        band, which says nothing about where the other end is: a pass that cuts from a
        state 900 mm clear straight to one against the panel produces a single linear move
        whose line has to have collision-free inverse kinematics for its whole length, and
        which the robot flies under the tool speed cap all the way in.  Linear motion is
        wanted where the tool is working, not for the approach to it.

        So a replacement is refused when it is linear and an end of it sits beyond
        ``--linear-introduce-mm`` -- unless the stretch it replaces already had a move
        doing the same thing.  That exemption is what keeps this a limit on *introducing*
        linear motion.  A route that already reaches in from far out, because the band
        gate labelled it that way or a Cartesian route was recut there, may still be
        shortened, thinned and relocated; the passes simply cannot create the reach where
        it was not there before.

        Off by default in the sense that matters: with no limit set, or with none of the
        three states measured beyond it, this is the constant False it was before.
        """
        if self.zone is None or self.zone.reach_mm <= 0.0:
            return False
        if not self._overreaches(a, b):
            return False
        chain = [a, *replaced, b]
        if any(self._overreaches(x, y) for x, y in zip(chain, chain[1:])):
            return False
        self.overreach_refusals += 1
        return True

    def refuses(self, a: np.ndarray, b: np.ndarray, replaced) -> bool:
        """Whether the profile rules forbid replacing ``replaced`` with the move a->b.

        The one question the optimisation passes ask.  Both halves guard the same thing
        from opposite sides -- :meth:`demotes` stops a near-panel stretch being dissolved
        into a joint chord through it, :meth:`overreaches` stops a linear one being grown
        outward into the approach -- and neither is about cost, so they are settled before
        the move is checked or priced.
        """
        return self.demotes(a, b, replaced) or self.overreaches(a, b, replaced)

    # -- what it forbids ----------------------------------------------------
    def blocked(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Whether this move is unusable, checked along the path it will really take."""
        if self.motion(a, b) == PTP:
            return self.cell.segment_collides(a, b, max_step=self.max_step)
        return not self.linear_ok(a, b)

    def check_and_cost(self, a: np.ndarray, b: np.ndarray, *, fa: float | None = None,
                       fb: float | None = None, stops: bool = False) -> float | None:
        """:meth:`blocked` and :meth:`cost` off one walk.  ``None`` when it is blocked.

        Every caller that costs a move has already had to ask whether it is usable, and
        asked separately the two questions load the same joint states twice over -- once
        to test contact, once to measure clearance.  Here the walk that answers the first
        keeps what the second needs.

        Both profiles share it.  A joint move hands over the factors on its joint grid; a
        linear move hands over the factors at the stations along the tool's line, which is
        the walk that proved it clear and the only curve it may be priced on.
        """
        if self.motion(a, b) == PTP:
            hit, facs = self.cell.segment_scan(a, b, max_step=self.max_step,
                                               factors=self.penalised)
            if hit:
                return None
        else:
            chain, facs = self.linear_chain(a, b, factors=self.penalised)
            if chain is None:
                return None
        return self.cost(a, b, fa=fa, fb=fb, stops=stops, facs=facs)

    def linear_chain(self, a: np.ndarray, b: np.ndarray, factors: bool = False):
        """The states the tool passes through running straight from ``a`` to ``b``.

        Returns ``(chain, factors)``, or ``(None, None)`` when it cannot get there: no
        inverse kinematics somewhere along the line, or a collision in the joint gaps
        between the samples.  ``factors`` asks for the penalty factor at each state of the
        chain, taken off the walk that proves the gaps clear rather than measured
        afterwards; unset, they come back as 1.0 and nothing is queried.

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
            chain = plan_linear(self.cell, self.cell.pose_mm(a), self.cell.pose_mm(b), a,
                                step_mm=self.tool_step, step_rad=self.max_step)
        except PlanningError:
            return None, None               # no inverse kinematics somewhere along it
        if _line_end_fault(chain, a, b) is not None:
            return None, None               # the line does not arrive at ``b``
        # Inverse kinematics returns the solution nearest its seed rather than the state
        # asked for, so pin the ends back before checking the gaps between the samples.
        chain[0] = np.asarray(a, dtype=float)
        chain[-1] = np.asarray(b, dtype=float)
        hit, facs = _chain_scan(self.cell, chain, self.max_step,
                                factors=factors and self.penalised)
        if hit:
            return None, None
        return chain, (facs if facs is not None else [1.0] * len(chain))

    def linear_ok(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Can the tool run straight from ``a`` to ``b``?"""
        return self.linear_chain(a, b)[0] is not None

    def clear_path(self, a: np.ndarray, b: np.ndarray, factors: bool = False):
        """The interior points of this move, or ``None`` if the move is unusable.

        :meth:`blocked` with the route kept instead of thrown away, for the one caller that
        installs that route rather than only asking whether the move is allowed.  The
        distinction is the whole point of it: a joint move follows the joint chord, which
        ``_resample`` describes exactly, but a linear move follows the Cartesian line, and
        filling that stretch with joint-space interpolation puts points on a curve the tool
        never passes through.

        Not used by :meth:`blocked` itself, which is asked tens of thousands of times a
        pass and wants a verdict without building a point list to reach it.

        Returns ``(points, factors)``, the factors being those of the interior points and
        no others; ``(None, None)`` when the move is unusable.  ``factors`` asks for them
        to be measured -- unset, they come back as 1.0 and nothing is queried, which is
        what a caller with the penalty switched off wants.

        Either way they come off the same walk that proved the move clear, though by
        different routes to it.  For a joint move ``_resample`` returns exactly the
        interior of the grid ``segment_scan`` steps along, so the states line up one for
        one.  A linear move's states come from inverse kinematics along the tool's line
        rather than from that grid, but the gaps between them are walked to prove them
        clear, and each such walk starts on a chain state -- so the factors are read there
        instead.  Neither pays a state load for them.
        """
        if self.motion(a, b) == PTP:
            blocked, facs = self.cell.segment_scan(a, b, max_step=self.max_step,
                                                   factors=factors)
            if blocked:
                return None, None
            points = _resample(self.cell, a, b, self.max_step)
            return points, (list(facs[1:-1]) if facs is not None
                            else [1.0] * len(points))
        chain, facs = self.linear_chain(a, b, factors=factors)
        if chain is None:
            return None, None
        return [np.asarray(q, dtype=float) for q in chain[1:-1]], list(facs[1:-1])

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
             fb: float | None = None, stops: bool = False,
             facs: list[float] | None = None) -> float:
        """Penalised time, with the tool speed cap folded in.

        The cap raises the base time; the clearance penalty on top of it is unchanged, so
        the two are combined by scaling rather than by replacement.  Optimising without
        this would let the passes trade a joint move for a linear one that is quicker on
        the joint limits and slower once the tool speed governs it.

        The crossing penalty is added afterwards rather than folded into the floor, so it
        is not multiplied by the clearance factor: it is a fixed preference about how many
        times the route enters the band, not time spent anywhere.

        The clearance penalty is read along the path the move really takes.  For a joint
        move that is the joint grid, and ``facs`` are its factors.  For a linear move it is
        the tool's straight line, and ``facs`` are the factors at the stations along it,
        one per station, as :meth:`linear_chain` returns them; left unset they are measured
        here.  This used to sample the joint chord for both, so a long linear move was
        charged for a curve it never flies -- one that nothing collision checks and that
        can pass through the parts.  Measured on a 707 mm move whose line stood 3.6 mm
        clear at worst, the chord reached 64.5 mm inside the panel and priced the move at
        54.4 s where its line gave 3.6 s, and simplify and polish kept a 918 mm detour
        rather than take it.  A linear move with no reachable line has no price and
        costs infinity; every caller that installs a move has proved it clear first.
        """
        if self.motion(a, b) == LIN:
            raw = self.cell.move_time(a, b) if stops else self.cell.cruise_time(a, b)
            if not self.penalised or self.cell._pm is None:
                penalised = raw
            else:
                if facs is None:
                    chain, facs = self.linear_chain(a, b, factors=True)
                    if chain is None:
                        return float("inf")
                penalised = self.cell.price(raw, facs)
        else:
            penalised = self.cell.segment_cost(a, b, max_step=self.max_step,
                                               fa=fa, fb=fb, stops=stops, facs=facs)
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
    last = len(path) - 1
    while i < len(path) - 1:
        j = last
        while j > i + 1:
            # Not a stop the robot may make.  Only reached past, not forced: the adjacent
            # point is still taken unchecked when nothing further works, and
            # ``clear_stop_band`` deals with it afterwards.
            if j < last and model.cell.in_stop_band(path[j]):
                j -= 1
                continue
            if model.refuses(path[i], path[j], path[i + 1:j]):
                j -= 1
                continue
            chord = model.check_and_cost(path[i], path[j],
                                         fa=factor(i), fb=factor(j), stops=True)
            if chord is None:
                j -= 1
                continue
            if chord > polyline_cost(i, j) + 1e-9:
                j -= 1
                continue
            break
        out.append(path[j])
        i = j
    return out


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
    min_attempts: int = 20          # polish is the only pass that relocates

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
    # Stop-to-stop, ramps included: every point here is one the robot really stops at.
    costs = [model.cost(pts[i], pts[i + 1], fa=fac[i], fb=fac[i + 1], stops=True)
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
            if model.refuses(pts[k - 1], pts[k + 1], [pts[k]]):
                k += 1
                continue
            direct = model.check_and_cost(pts[k - 1], pts[k + 1], fa=fac[k - 1],
                                          fb=fac[k + 1], stops=True)
            if direct is None:
                k += 1
                continue
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
        if not cell.within_limits(candidate) or cell.in_stop_band(candidate):
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
        if (model.refuses(pts[k - 1], candidate, [pts[k]])
                or model.refuses(candidate, pts[k + 1], [pts[k]])):
            continue
        f = model.penalty_factor(candidate)
        first = model.check_and_cost(pts[k - 1], candidate, fa=fac[k - 1], fb=f,
                                     stops=True)
        if first is None:
            continue
        second = model.check_and_cost(candidate, pts[k + 1], fa=f, fb=fac[k + 1],
                                      stops=True)
        if second is None:
            continue
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


# How far past the edge of the stop band ``clear_stop_band`` tries putting joint 5, in
# degrees, nearest first.  The first clears the edge by enough not to read as inside it
# after rounding; the others are there for when the move off the edge is blocked.
STOP_BAND_NUDGES_DEG = (0.5, 5.0, 15.0)


def clear_stop_band(model: "MotionModel", path: list[np.ndarray],
                    log=None) -> list[np.ndarray]:
    """Take every interior waypoint out of the joint 5 stop band, or refuse the route.

    The band constrains where the robot may **stop**, not where it may travel, so the
    searches and the dense fill never hear of it and this runs on the reduced path, where
    every point is a stop.  ``simplify`` and ``polish`` already avoid choosing one inside
    the band; this is for the ones they could not avoid.  The ends belong to the locators
    either side and are left alone.

    Each offender gets two repairs, and the cheaper of those that work is kept:

    * **remove** -- go straight past it, if the move is clear and ``refuses`` allows it.
      Unlike ``polish``'s removal this is kept even when it costs time.
    * **nudge** -- set joint 5 just outside the band, on the side it is already on first,
      and keep its other joints.  Both moves through it are checked and costed along the
      path their profile gives them, and a nudge ``refuses`` rejects is not offered.

    Neither working is a ``PlanningError``, which the caller treats like any other failed
    route: the next gun opening, the next fallback pose.
    """
    cell = model.cell
    if cell.stop_band_rad <= 0.0 or len(path) < 3:
        return path
    pts = [np.asarray(p, dtype=float).copy() for p in path]
    removed = nudged = 0
    k = 1
    while k < len(pts) - 1:
        q = pts[k]
        if not cell.in_stop_band(q):
            k += 1
            continue
        a, b = pts[k - 1], pts[k + 1]
        fa, fb = model.penalty_factor(a), model.penalty_factor(b)
        options = []                            # (cost, replacement or None to remove)
        if not model.refuses(a, b, [q]):
            direct = model.check_and_cost(a, b, fa=fa, fb=fb, stops=True)
            if direct is not None:
                options.append((direct, None))
        side = 1.0 if float(q[STOP_BAND_JOINT]) >= 0.0 else -1.0
        for sign in (side, -side):
            found = None
            for extra in STOP_BAND_NUDGES_DEG:
                candidate = q.copy()
                candidate[STOP_BAND_JOINT] = sign * (cell.stop_band_rad
                                                     + np.deg2rad(extra))
                if not cell.within_limits(candidate):
                    break                       # further out on this side is no better
                if (model.refuses(a, candidate, [q])
                        or model.refuses(candidate, b, [q])):
                    continue
                f = model.penalty_factor(candidate)
                first = model.check_and_cost(a, candidate, fa=fa, fb=f, stops=True)
                if first is None:
                    continue
                second = model.check_and_cost(candidate, b, fa=f, fb=fb, stops=True)
                if second is None:
                    continue
                found = (first + second, candidate)
                break
            if found is not None:
                options.append(found)
                break                           # the nearer side worked
        if not options:
            raise PlanningError(
                f"waypoint {k} of {len(pts)} stops with joint 5 at "
                f"{np.rad2deg(float(q[STOP_BAND_JOINT])):.1f} deg, inside the "
                f"+/-{np.rad2deg(cell.stop_band_rad):g} deg stop band, and can neither "
                f"be removed nor moved out of it")
        _, replacement = min(options, key=lambda o: o[0])
        if replacement is None:
            del pts[k]
            removed += 1                        # the point now at k is tested next
        else:
            pts[k] = replacement
            nudged += 1
            k += 1
    if log and (removed or nudged):
        log(f"      stop band: removed {removed} and moved {nudged} waypoints holding "
            f"joint 5 within {np.rad2deg(cell.stop_band_rad):g} deg of zero, "
            f"{len(pts)} points")
    return pts


def _refine(model: "MotionModel", path: list[np.ndarray], *, shortcut_seconds: float,
            polish_seconds: float, relocate: "Relocation | None" = None,
            anchors: list[int] | None = None, log=print) -> list[np.ndarray]:
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
    improved = shortcut(model, path, time_budget=shortcut_seconds, anchors=anchors,
                        log=log)
    stagetrace.stage("shortcut", model, improved)
    reduced = simplify(model, improved)
    stagetrace.stage("simplify", model, reduced)
    polished = polish(model, reduced, time_budget=polish_seconds, relocate=relocate,
                      log=log)
    stagetrace.stage("polish", model, polished)
    # Outside the budgets on purpose: a route refined with none still ships its stops.
    cleared = clear_stop_band(model, polished, log=log)
    stagetrace.stage("stop band", model, cleared)
    if log and model.overreach_refusals:
        log(f"      {model.overreach_refusals} replacements refused for starting linear "
            f"motion beyond {model.zone.reach_mm:g} mm of clearance")
    return cleared


def _resample(cell: Cell, a: np.ndarray, b: np.ndarray, step: float) -> list[np.ndarray]:
    """Points strictly between a and b, spaced no further apart than ``step``."""
    n = max(1, int(np.ceil(float(np.max(np.abs(b - a))) / max(step, 1e-9))))
    return [a + (b - a) * (k / n) for k in range(1, n)]


def _thin(a: np.ndarray, points: list[np.ndarray], b: np.ndarray,
          step: float) -> list[np.ndarray]:
    """Enough of a chain to describe it at ``step``, and no more.

    ``plan_linear`` places a station every ``--check-step-mm`` of tool travel, which is
    far finer than the joint-space spacing the rest of the fill uses -- on a metre-long
    move, well over a hundred points against seven.  Handing all of them on would change
    how densely the route is sampled at the same time as changing which curve it is
    sampled along, and only the second is wanted here: what the passes were missing is
    candidate points that lie on the route, not more of them.

    So this applies ``_resample``'s own criterion -- largest joint delta since the last
    point kept -- walking the real curve instead of the chord.  A point within a step of
    the far end is dropped as well, so the fill cannot leave a step so short that it
    costs a stop for no distance.
    """
    out: list[np.ndarray] = []
    last = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    for q in points:
        q = np.asarray(q, dtype=float)
        if float(np.max(np.abs(q - last))) < step:
            continue
        if float(np.max(np.abs(b - q))) < step:
            continue
        out.append(q)
        last = q
    return out


def _fill(cell: Cell, model: "MotionModel | None", a: np.ndarray, b: np.ndarray,
          step: float) -> list[np.ndarray]:
    """Points strictly between ``a`` and ``b``, on the curve the robot will really fly.

    A joint move follows the joint chord and ``_resample`` describes it exactly.  A linear
    move does not: the tool runs a straight line in space and the joints come from inverse
    kinematics along it, so interpolating the chord puts the fill on a curve the route
    never passes through.

    That distinction was already respected everywhere a move is *checked* or *installed*,
    and nowhere the route is *sampled*, which left the optimisation passes drawing their
    candidate cut endpoints from states the robot does not visit.  Measured on a transit
    that withdrew 921 mm to cross between two welds 59 mm apart: cuts taken off the real
    line were clear and 46% cheaper than the route that shipped, and none of their
    endpoints existed in the chord-interpolated fill the pass was given.

    Falls back to the chord when the linear move will not validate.  Densifying is not the
    place to reject a route -- nothing here has ever done so, and ``_verify_runs`` sweeps
    the finished path along the path each move really takes -- so a stretch that cannot be
    flown linearly is filled as before and left to fail where failures are reported.

    The profile is asked of the model rather than taken from where the path came from,
    and the difference is the point.  Every solver's edges do have a known type -- OMPL
    only makes joint-space straight lines, and cannot make anything else -- but that is
    how the edge was *planned*, and ``_split_runs`` decides how it will be *flown* from
    where its ends sit.  An OMPL edge with both ends in the band ships linear.  On Path 1
    that is the whole route, and the edge in question has a blocked joint chord and a
    clear cartesian line: filling it on its provenance would sample a curve nothing can
    fly.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if float(np.max(np.abs(b - a))) <= step:
        # Nothing to fill in, so do not pay a chain to be told so.  This is the whole of a
        # cartesian-tree route, which arrives already dense -- its stations are one
        # ``--check-step-mm`` of tool travel apart, far inside one joint step -- and whose
        # states are the ones it validated.  Re-deriving them is what that planner keeps
        # its own edge states to avoid.
        return []
    if model is not None and model.motion(a, b) == LIN:
        points, _ = model.clear_path(a, b)
        if points is not None:
            return _thin(a, points, b, step)
    return _resample(cell, a, b, step)


def _densify_marked(cell: Cell, path: list[np.ndarray], step: float,
                    model: "MotionModel | None" = None
                    ) -> tuple[list[np.ndarray], list[int]]:
    """A solver's handful of states filled in, plus where the points it was given ended up.

    OMPL returns very few points -- four is typical -- and every pass downstream wants the
    route rather than the tree's nodes.  The spacing is the same ``--check-step-deg`` the
    checking uses, so consecutive points differ by less than anything here can resolve.

    The second return is the index in the dense list of each point of ``path``, in order.
    Those are the route's real waypoints; everything between them is fill.  Nothing
    downstream can tell the two apart from the geometry -- a filled point sits on the
    straight line between its neighbours, and so does a waypoint on a straight stretch --
    so a pass that needs to know has to be told here or not at all.

    ``model`` decides how each stretch is filled: given one, a move it judges linear is
    filled along the tool's line rather than the joint chord.  See :func:`_fill`.  Without
    one every stretch is a chord, which is what a caller with no profile in force wants.
    """
    out: list[np.ndarray] = [np.asarray(path[0], dtype=float)]
    marks = [0]
    for a, b in zip(path, path[1:]):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        out.extend(_fill(cell, model, a, b, step))
        out.append(b)
        marks.append(len(out) - 1)
    return out, marks


def shortcut(model: "MotionModel", path: list[np.ndarray], *, time_budget: float = 2.0,
             rng: np.random.Generator | None = None,
             anchors: list[int] | None = None,
             log=None) -> list[np.ndarray]:
    """Reshape a path to take less time, penalised for running close to the parts.

    ``simplify`` can only delete waypoints that a straight move already bypasses, so it
    never changes the route: a sampled path that swings the arm around the base to reach a
    point beside it keeps that swing.  This pass reshapes the route instead, by one move:

    * **cut** -- replace a stretch of path with the straight move between its ends, when
      that move is collision free and cheaper.  This is what removes gross detours.

    It used to offer a second move, **relocate**, which displaced one point of the dense
    path and kept it if the pair of steps through it got cheaper.  On a dense path that
    move cannot do the job it was there for, and the reason is the cost function rather
    than the geometry.  A step is charged at ``max`` of the factors at its two ends, so
    displacing a single point changes nothing unless that point is a *strict* local
    maximum of the penalty -- otherwise both maxima stay pinned by the neighbours it did
    not move, and the only effect is the extra travel of a detour out and straight back,
    which the triangle inequality guarantees loses.  The points are a collision-check step
    apart and clearance varies smoothly over that distance, so strict local maxima are
    rare and shallow.  Measured on a real leg it kept 1 of some 15 draws while cuts kept 8
    of 15, and each losing draw still paid for its collision checks.  ``polish`` keeps its
    own relocation, where the points are real stops far enough apart for one to move
    independently of its neighbours, and there it earns its place.

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

    ``anchors`` says which entries of ``path`` are real waypoints rather than fill, for
    the caller that densified before calling.  It matters only under the penalty's
    ``whole_move`` rule, where the unit being charged is a move between waypoints and not
    a sub-step, so the pass has to know where one move ends and the next begins.  Left
    unset, the points handed in are taken to be the waypoints, which is what they are when
    nobody has densified yet.

    Work is bounded by ``time_budget`` seconds; the result is always collision free, since
    every replacement is checked before it is kept.
    """
    if len(path) < 3 or time_budget <= 0:
        return [np.asarray(p, dtype=float).copy() for p in path]

    cell = model.cell
    max_step = model.max_step
    if anchors is None:
        dense, marks = _densify_marked(cell, path, max_step, model=model)
    else:
        # Already dense: densifying again would be a no-op that invalidated the indices.
        dense = [np.asarray(p, dtype=float) for p in path]
        marks = sorted({0, len(dense) - 1, *(int(m) for m in anchors)})

    # One clearance query per waypoint, cached: the geometry query dominates, so scoring a
    # candidate has to be arithmetic over remembered factors rather than fresh queries.
    penalised = model.penalised
    fac = [model.penalty_factor(p) if penalised else 1.0 for p in dense]

    # Which unit the penalty is charged over.  Per sub-step the anchors are irrelevant and
    # the cost is a plain sum; per move they decide where one charge ends and the next
    # begins, and the pass has to respect the same division the emitted route will have.
    whole = penalised and bool(getattr(cell.penalty, "whole_move", False))

    def step_cost(x: np.ndarray, y: np.ndarray, fx: float, fy: float) -> float:
        # Outside the factor, on the same footing as everywhere else it is charged.
        return model.cruise_time(x, y) * max(fx, fy) + model.crossing_penalty(x, y)

    def move_cost(i: int, j: int) -> float:
        """dense[i..j] as one move: its worst state prices the whole of it."""
        cross = sum(model.crossing_penalty(dense[k], dense[k + 1]) for k in range(i, j))
        secs = sum(model.cruise_time(dense[k], dense[k + 1]) for k in range(i, j))
        return secs * max(fac[i:j + 1]) + cross

    def span_cost(i: int, j: int) -> float:
        """Cost of dense[i..j], divided into moves the way the anchors divide it.

        ``i`` and ``j`` are boundaries whether or not they are anchors, which is what lets
        a caller ask for the cost of a region it is about to cut at.
        """
        if not whole:
            return sum(step_cost(dense[k], dense[k + 1], fac[k], fac[k + 1])
                       for k in range(i, j))
        bounds = [i] + [m for m in marks if i < m < j] + [j]
        return sum(move_cost(u, v) for u, v in zip(bounds, bounds[1:]))

    rng = rng or np.random.default_rng(0)
    before = span_cost(0, len(dense) - 1)
    deadline = time.time() + time_budget
    tried = cuts = 0

    while time.time() < deadline and len(dense) > 2:
        tried += 1
        cuts += _try_cut(model, dense, fac, marks, rng, penalised, whole, span_cost)

    if log:
        after = span_cost(0, len(dense) - 1)
        gain = 100.0 * (1.0 - after / before) if before > 0 else 0.0
        detail = "penalised cruise time" if penalised else "cruise time"
        log(f"      shortcut: {cuts} cuts kept from {tried} attempts, "
            f"{detail} {before:.2f} -> {after:.2f} s ({gain:.0f}% better)")
    return dense


def _try_cut(model: "MotionModel", dense, fac, marks, rng, penalised, whole,
             span_cost) -> int:
    """Replace dense[i..j] with the straight move between the ends, if that is cheaper.

    The comparison is made over the whole region between the anchors either side of the
    cut, not over ``i..j`` alone.  Under the per-sub-step rule those outer stretches are
    identical on both sides and cancel, leaving exactly the ``i..j`` comparison this pass
    has always made.  Under the whole-move rule they do not: a cut that swallows an anchor
    merges two moves into one, and the merged move is charged at the worst state of both,
    which is a cost the middle alone does not show.

    **A cut does not create anchors.**  It is tempting to make its two ends waypoints,
    since it installs a real straight move between them, but under the whole-move rule
    that turns the pass into a waypoint generator: adding a boundary can only ever lower
    the cost, because a maximum over part of a move is never above the maximum over all of
    it.  Measured on a straight stub route where no cut changes the geometry at all, the
    pass took 6 of them and reported the route 50% cheaper -- entirely by re-dividing it.
    Where the waypoints go is ``simplify``'s decision, made on stop-to-stop time where an
    extra stop costs a pair of ramps; this pass may only reshape the route between them.
    """
    i, j = sorted(rng.integers(0, len(dense), size=2))
    if j - i < 2:
        return 0
    # How wide the comparison has to be.  Under the whole-move rule a cut that swallows an
    # anchor merges two moves, and the merged move is charged at the worst state of both,
    # which cannot be seen from i..j alone -- so the region runs out to the anchors either
    # side.  Under the per-sub-step rule cost is a plain sum over steps, so
    #
    #     span_cost(lo, hi) = span_cost(lo, i) + span_cost(i, j) + span_cost(j, hi)
    #
    # the outer stretches appear identically on both sides of every test below and cancel.
    # Widening there would cost four passes over roughly twice the ground for an answer
    # already in hand, and each step of a pass prices a move -- which on a linear leg is
    # forward kinematics.
    if whole:
        lo = max([m for m in marks if m <= i], default=0)
        hi = min([m for m in marks if m >= j], default=len(dense) - 1)
    else:
        lo, hi = i, j
    before = span_cost(lo, hi)
    # A lower bound on what the region can cost once cut, used to reject cheaply.  Two
    # things make it a bound rather than the answer: every factor is at least 1, so the
    # unpenalised cruise time of the direct move is a floor under its middle; and cutting
    # the region at i and j can only lower it, since a maximum over part of a move never
    # exceeds the maximum over the whole.  The real figure is taken below, after the
    # replacement exists.
    outer = span_cost(lo, i) + span_cost(j, hi) if whole else 0.0
    if outer + model.cell.cruise_time(dense[i], dense[j]) >= before - 1e-9:
        return 0
    if model.refuses(dense[i], dense[j], dense[i + 1:j]):
        return 0
    # The interior is asked of the model rather than interpolated here.  Under a linear
    # profile the check above runs along the Cartesian line, and joint-space points would
    # not lie on it: the pass would verify one curve and install another, leaving moves
    # that claim to be straight lines nobody ever checked.
    points, factors = model.clear_path(dense[i], dense[j], factors=penalised)
    if points is None:
        return 0
    # Scored in full with or without the penalty.  The bound above is joint cruise time
    # alone, and without a penalty this used to be the whole test -- so a cut that turned
    # joint hops into a linear move held to the tool speed cap, or that added a crossing
    # of the band, was kept whenever the joints alone got there sooner.  The factors are
    # all 1.0 when the penalty is off, which reduces this to exactly that comparison.
    chain = [dense[i]] + points + [dense[j]]
    chain_f = [fac[i]] + factors + [fac[j]]
    secs = [model.cruise_time(x, y) for x, y in zip(chain, chain[1:])]
    cross = sum(model.crossing_penalty(x, y) for x, y in zip(chain, chain[1:]))
    if whole:
        middle = sum(secs) * max(chain_f) + cross
    else:
        middle = sum(s * max(fx, fy) for s, fx, fy
                     in zip(secs, chain_f, chain_f[1:])) + cross
    if outer + middle >= before - 1e-9:
        return 0

    old_points, old_factors = dense[i + 1:j], fac[i + 1:j]
    old_marks = list(marks)
    dense[i + 1:j] = points
    fac[i + 1:j] = factors
    shift = (i + 1 + len(points)) - j
    marks[:] = [m for m in marks if m <= i] + [m + shift for m in marks if m >= j]
    # Under the whole-move rule, score the region as it now really stands and put it back
    # if that was not an improvement.  Scoring in place rather than predicting: a cut that
    # swallowed an anchor leaves a move spanning further than i..j, and rebuilding that
    # arithmetic outside the list it applies to is how the two drift apart.  Per sub-step
    # there is nothing to re-score -- the region is i..j, and its cost after the cut is
    # `middle`, which the test above has already compared.
    if whole and span_cost(lo, hi + shift) >= before - 1e-9:
        dense[i + 1:i + 1 + len(points)] = old_points
        fac[i + 1:i + 1 + len(points)] = old_factors
        marks[:] = old_marks
        return 0
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
_continuous_warned = False


@dataclass
class Deadline:
    """How long one segment's search may go on for, whatever it is doing.

    Every budget below this is a budget for a part of the search -- runs, seconds per run,
    seconds of shortcutting -- and they multiply.  A transit that fails everywhere spends
    its phase-one runs at each opening, then the Cartesian searches, then phase two, then
    the whole of that again through each fallback pose, then a solve for each ordered pair
    of openings at each of those poses.  Every one of those numbers is defensible on its
    own and the product of them is hours, on a segment that may simply have no route.

    So this is the figure that is actually about the operator's afternoon: past it the
    search stops where it stands and the segment is reported as one that found no route,
    which is what the run loop already does with a segment that fails outright -- it logs
    the reason, records it on the segment and goes on to the next.  Nothing part-built is
    kept, and nothing already found is thrown away: solutions in hand when the clock runs
    out are still ranked, refined and shipped.

    ``seconds <= 0`` never expires, which is what everything did before this existed.
    Measured on the monotonic clock, so it is unaffected by the system clock being set.
    """
    seconds: float = 0.0
    started: float = field(default_factory=time.monotonic)

    @property
    def unlimited(self) -> bool:
        return self.seconds <= 0.0

    @property
    def spent(self) -> float:
        return time.monotonic() - self.started

    @property
    def left(self) -> float:
        return float("inf") if self.unlimited else self.seconds - self.spent

    @property
    def expired(self) -> bool:
        return not self.unlimited and self.left <= 0.0

    def clamp(self, seconds: float) -> float:
        """``seconds``, cut to what is left.

        Applied to every per-run limit, so the last run of a segment stops at the deadline
        rather than a run's length past it.  A search asked for the time remaining and
        given none is not started at all -- see ``ran_out``.
        """
        return seconds if self.unlimited else max(0.0, min(seconds, self.left))

    def ran_out(self, log, what: str) -> bool:
        """Whether the clock has gone, saying so once at the point work stopped.

        The caller asks this where it is about to start something, rather than being
        interrupted partway: a run stopped halfway leaves nothing usable behind, so there
        is no value in cutting one short beyond not beginning it.
        """
        if not self.expired:
            return False
        log(f"      {self.seconds / 60:g} minutes spent on this segment, which is the "
            f"limit; giving up on {what}")
        return True


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
                  continuous_check: bool = False,
                  log=None) -> OMPLRealVectorMoveProfile:
    """The profile one OMPL run is made under.

    ``continuous_check`` swaps the collision evaluator from ``DISCRETE`` to
    ``LVS_CONTINUOUS``.  The default samples states along each edge and tests each one,
    which is what lets a move step over a fixture: the resolution is
    ``longest_valid_segment_length``, a single distance in **joint** space, and how far
    that carries the tool depends entirely on which joint moved.  At 0.02 rad it is some
    71 mm at j1 against 16 mm at j6 on this robot, so one figure is either wasteful at the
    wrist or blind at the shoulder.  It is blind at the shoulder, and that is the
    ``move 0 -> 1`` rejection: our own check bisects wherever the tool travels more than
    ``--check-step-mm``, sees what the sampler stepped over, and refuses the route.

    OMPL cannot be asked to measure the tool instead.  It plans over an abstract state
    space and has no forward kinematics, so Cartesian displacement is not a quantity it
    can compute; the places one would inject it -- a custom ``MotionValidator``, or an
    override of ``StateSpace::validSegmentCount`` -- are not in these bindings, which
    export the planner, the configurators and this profile and nothing else of OMPL's.

    The continuous evaluator sidesteps the question rather than answering it: it sweeps
    each sub-step instead of sampling it, so there are no gaps to step over whatever the
    lever arm.  ``LVS_CONTINUOUS`` rather than ``CONTINUOUS`` because the sweep is a
    linear interpolation of link transforms while a joint move carries the link along an
    arc -- keeping the joint-space subdivision holds that approximation to a short step.

    Off by default.  It is dearer per test than sampling, and OMPL's budget is already
    what limits this cell, so it can buy fewer solutions found in exchange for fewer
    rejected -- which of those wins is a question about a cell and not one to guess at.
    """
    global _planning_time_warned, _continuous_warned
    profile = OMPLRealVectorMoveProfile()
    profile.collision_check_config.longest_valid_segment_length = segment_length
    if continuous_check:
        if _LVS_CONTINUOUS is None:
            if not _continuous_warned:
                _continuous_warned = True
                (log or print)(
                    "      ! this build of the bindings does not expose "
                    "CollisionEvaluatorType_LVS_CONTINUOUS; checks stay discrete")
        else:
            profile.collision_check_config.type = _LVS_CONTINUOUS
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


def _plan_via(cell: Cell, qa: np.ndarray, via: np.ndarray, qb: np.ndarray, *,
              ompl: OmplBudget, segment_length: float, check_step: float,
              continuous_check: bool = False,
              shortcut_seconds: float, polish_seconds: float,
              zone: LinearZone | None = None,
              openings: "GunOpenings | None" = None,
              deadline: "Deadline | None" = None,
              relocate: "Relocation | None" = None, log=print,
              record: list | None = None) -> list[Run]:
    """A route from ``qa`` to ``qb`` by way of ``via``, all at the one gun opening.

    Routing through a fallback pose is what a robot programmer would do by hand -- it turns
    a long detour around the panels into two easy problems -- but it costs two full solves
    rather than one, which is why it is not reached until directness has been given up on
    at every opening.

    One opening for the whole leg, and therefore no rotation.  The gun does not change
    partway through a move and these two halves are one leg of the output, so the caller
    walks the openings one at a time and ``openings`` holds that single value in both
    lists.  Each phase then spends its own runs on it, which is what every opening got
    before the rotation existed.
    """
    halves: list[list[np.ndarray]] = []
    # Planned without a linear zone: the two legs are joined below and the whole route is
    # split afterwards, so splitting each half here would put a phase boundary at the
    # fallback pose whether the geometry called for one or not.
    first, _ = _plan_direct(cell, qa, via, ompl=ompl, segment_length=segment_length,
                            check_step=check_step, continuous_check=continuous_check,
                            shortcut_seconds=0.0, polish_seconds=0.0,
                            openings=openings, deadline=deadline, log=log, record=halves)
    second, _ = _plan_direct(cell, via, qb, ompl=ompl, segment_length=segment_length,
                             check_step=check_step, continuous_check=continuous_check,
                             shortcut_seconds=0.0, polish_seconds=0.0,
                             openings=openings, deadline=deadline, log=log,
                             record=halves)
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
                   check_step: float = 0.05, continuous_check: bool = False,
                   fallback_via=None,
                   fallback_runs: int = 0,
                   shortcut_seconds: float = 2.0, polish_seconds: float = 5.0,
                   planning_time: float = DEFAULT_PLANNING_TIME,
                   openings: list[float] | None = None,
                   main_openings: int = 5, extra_openings: int = 0,
                   opening_round_mm: float = 0.0,
                   zone: LinearZone | None = None,
                   cartesian: "CartesianBudget | None" = None,
                   relocate: "Relocation | None" = None,
                   deadline: "Deadline | None" = None,
                   record: list | None = None,
                   log=print) -> list[tuple[list[Run], float | None]]:
    """Plan a transit, choosing a gun opening for it when the natural one will not do.

    Some destinations simply cannot be reached at the opening the robot arrives with: the
    tip is 200 mm of swing, so an opening that clears a fixture on the way out fouls it on
    the way back.  ``openings`` names the two the transit's own ends ask for, departure
    first, and ``main_openings`` and ``extra_openings`` say how many in all to find --
    ``_opening_lists`` finds them, screens them against this transit's own poses, and
    splits them into the rotation phase one deals round and the reserve phase two walks.

    Returns one ``(runs, opening)`` leg per gun state.  A single leg is always preferred,
    and a two-leg answer is only produced when no single opening works: the gun then
    changes at the intermediate pose, where the robot is stationary and the change costs
    no motion.

    **The direct transit picks its own opening**, which is why it is one call here rather
    than one per opening.  Phase one deals its runs round the main list and keeps the
    cheapest solution of the whole set, so the choice between openings is made on what
    each one produced rather than on the order they were offered in.  Everything after it
    fixes one opening per attempt instead, because a route through a fallback pose and
    either half of a two-leg split are legs whose parts have to agree on one gun state.

    ``record``, if given, is extended with each leg's route as the sampling planner
    returned it, one entry per returned leg and in the same order.  Failed attempts leave
    nothing behind: only the openings that were actually used contribute.

    ``fallback_runs`` caps the sampling-planner budget for everything past the direct
    transit -- see ``OmplBudget.capped``.  That tail is where nearly all of the worst case
    sits, and it is also where the full budget buys least.

    ``deadline`` is the wall clock for this segment.  Every budget here is a budget for
    a part of the search and they multiply, so it is the one figure that bounds what a
    transit with no route can cost: past it nothing further is started, whatever is
    already in hand is still refined and shipped, and a transit with nothing in hand
    raises like any other failure -- the caller's loop records it and moves on to the next
    segment.

    ``fallback_via`` is either the poses to detour through or a zero-argument callable
    returning them.  The callable form exists because finding them is now a search of its
    own -- see :mod:`weldpath.fallback` -- and most transits solve directly and never need
    one.  It is called at most once, and only after the direct transit has failed at every
    opening, so a transit that solves pays nothing for the search it did not use.
    """
    ompl = ompl or OmplBudget()
    reduced = ompl.capped(fallback_runs)
    deadline = deadline or Deadline()

    def gave_up(exc: PlanningError) -> PlanningError:
        """The failure to report, naming the clock where that is what stopped it.

        Worth distinguishing: a transit that ran out of time may well have a route, and
        the segment's log then reads as though none exists.
        """
        if not deadline.expired:
            return exc
        return PlanningError(
            f"gave up after {deadline.spent / 60:.0f} minutes, the limit for one "
            f"segment; the last failure was: {exc}")

    def lists(*poses) -> GunOpenings:
        return _opening_lists(cell, openings, list(poses), main_count=main_openings,
                              extra_count=extra_openings, round_mm=opening_round_mm,
                              log=log)

    direct = lists(("start", qa), ("goal", qb))
    if not direct.main:
        raise PlanningError("no gun opening leaves both ends of the transit clear")
    # Into a list of its own, and handed on only once the leg is one: an attempt that
    # captures its raw route and then fails to reduce it would otherwise leave an entry
    # behind for a leg that never shipped, and the caller pairs the two by position.
    raw: list = []
    try:
        runs, opening = _plan_direct(cell, qa, qb, ompl=ompl,
                                     segment_length=segment_length,
                                     check_step=check_step,
                                     continuous_check=continuous_check,
                                     shortcut_seconds=shortcut_seconds,
                                     polish_seconds=polish_seconds,
                                     zone=zone, cartesian=cartesian, relocate=relocate,
                                     openings=direct, deadline=deadline, log=log,
                                     record=raw)
        if record is not None:
            record.extend(raw)
        return [(runs, opening)]
    except PlanningError as exc:
        log(f"      no direct route over {_openings_note(direct.all)}: {exc}")
        last = exc

    if deadline.ran_out(log, "this segment before looking for a fallback pose"):
        raise gave_up(last)
    poses = list((fallback_via() if callable(fallback_via) else fallback_via) or [])
    if not poses:
        raise last

    # Through a fallback pose, one opening at a time.  Directness is still the outer
    # question -- a detour costs two full solves against one, so it is not started
    # anywhere until the cheap answer has been exhausted everywhere -- but the gun is no
    # longer the inner question for the transit itself, only for these.
    walked: dict[int, GunOpenings] = {}
    for v, via in enumerate(poses, 1):
        log(f"      retrying via fallback pose {v}/{len(poses)}")
        walked[v] = lists(("start", qa), ("fallback", via), ("goal", qb))
        for opening in walked[v].all:
            if deadline.ran_out(log, f"this segment at fallback pose {v}/{len(poses)}"):
                raise gave_up(last)
            log(f"      retrying{_gun_note(opening)}{_effort_note(reduced, ompl)}")
            through: list = []
            try:
                with cell.gun_opening(opening):
                    runs = _plan_via(cell, qa, via, qb, ompl=reduced,
                                     segment_length=segment_length,
                                     check_step=check_step,
                                     continuous_check=continuous_check,
                                     shortcut_seconds=shortcut_seconds,
                                     polish_seconds=polish_seconds, zone=zone,
                                     openings=GunOpenings.pinned(opening),
                                     deadline=deadline, relocate=relocate, log=log,
                                     record=through)
            except PlanningError as exc:
                log(f"      no route{_gun_note(opening)}: {exc}")
                last = exc
                continue
            if record is not None:
                record.extend(through)
            return [(runs, opening)]

    # No single opening reaches: split the move and change the gun partway, at a pose the
    # robot is already passing through and stationary at.
    for i, mid in enumerate(poses, 1):
        options = walked[i].all
        if len(options) < 2:
            # A split that does not change the gun is the single-opening route that has
            # already failed, and the halves cost two solves to establish it again.
            continue
        if cell.in_stop_band(mid):
            # The robot stands still here while the gun changes, and nothing downstream
            # moves an endpoint.
            log(f"      fallback pose {i}/{len(poses)} holds joint 5 inside the "
                f"stop band; not changing the gun there")
            continue
        for first_open in options:
            if deadline.ran_out(log, "this segment before splitting it in two"):
                raise gave_up(last)
            raw_first: list = []
            try:
                with cell.gun_opening(first_open):
                    first, _ = _plan_direct(cell, qa, mid, ompl=reduced,
                                            segment_length=segment_length,
                                            check_step=check_step,
                                            continuous_check=continuous_check,
                                            shortcut_seconds=0.0, polish_seconds=0.0,
                                            zone=zone, cartesian=cartesian,
                                            relocate=relocate,
                                            openings=GunOpenings.pinned(first_open),
                                            deadline=deadline, log=log,
                                            record=raw_first)
            except PlanningError:
                continue
            for second_open in options:
                if abs(second_open - first_open) < OPENING_TOL_MM:
                    continue                    # already ruled out as a single opening
                raw_second: list = []
                try:
                    with cell.gun_opening(second_open):
                        second, _ = _plan_direct(cell, mid, qb, ompl=reduced,
                                                 segment_length=segment_length,
                                                 check_step=check_step,
                                                 continuous_check=continuous_check,
                                                 shortcut_seconds=0.0, polish_seconds=0.0,
                                                 zone=zone, cartesian=cartesian,
                                                 relocate=relocate,
                                                 openings=GunOpenings.pinned(second_open),
                                                 deadline=deadline, log=log,
                                                 record=raw_second)
                except PlanningError:
                    continue
                if record is not None:
                    record.extend(raw_first + raw_second)
                log(f"      no single gun opening reaches; changing from "
                    f"{first_open:g} mm to {second_open:g} mm at fallback pose "
                    f"{i}/{len(poses)}")
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
    raise gave_up(last or PlanningError("freespace transit failed at every gun opening"))


def _gun_note(opening: float | None) -> str:
    """" with the gun at 40 mm", or nothing at all where the cell has no gun joint.

    ``_opening_lists`` holds ``None`` for a cell without one, and there is no opening to
    name in that case -- naming one anyway raised a TypeError out of the format string,
    turning a transit that merely failed into a crash.
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


@dataclass
class GunOpenings:
    """The gun openings one transit may be planned at, in the order they are tried.

    Two lists, because the two phases are asked different questions.  ``main`` is the
    rotation phase one and the Cartesian tree deal their runs round.  Every phase-one run
    is spent whether or not earlier ones solved, so spreading them over several openings
    costs nothing and buys a choice the old order could not make: there the whole
    phase-one budget went to one opening, and the next was reached only once that one had
    failed outright, by which time most of what the transit had to spend was gone.

    ``extra`` holds openings ``main`` does not, and is what phase two walks.  Phase one
    has already had runs at everything in ``main``; what phase two adds is a longer search
    per run, and spending it where three short ones just failed is the narrower of the two
    bets available.  So it is spent somewhere new, and an empty ``extra`` -- the gun ran
    out of distinct openings, or none were asked for -- means phase two has nothing to do.

    Both lists hold only openings the leg's own poses are clear at; see ``_opening_lists``.

    ``pinned`` is the degenerate case, and there are two of them.  A leg that has to fly at
    one opening throughout cannot rotate -- a route through a fallback pose is one leg of
    the output whose two halves are one gun state, and so is either half of a two-leg
    split -- and neither can a cell with no gun joint at all.  Both lists then hold that
    single value, so each phase spends its own runs on it, which is what every opening got
    before this rotation existed.
    """
    main: list[float | None] = field(default_factory=list)
    extra: list[float | None] = field(default_factory=list)

    @classmethod
    def pinned(cls, opening: float | None) -> "GunOpenings":
        """One opening for both phases: the gun is fixed for the whole leg."""
        return cls([opening], [opening])

    @property
    def all(self) -> list[float | None]:
        """Every opening, in preference order, for a caller that fixes one per attempt."""
        return [*self.main, *self.extra]


# Bisection stops splitting a range narrower than this where no rounding is in force.  The
# gun is a mechanism with backlash and 200 mm of swing, so openings a fraction of a
# millimetre apart are the same command; without a floor the halving never ends.
OPENING_FLOOR_MM = 1.0


def _opening_stream(named: list[float] | None, widest: float, round_mm: float,
                    tried: list[float]):
    """Openings to consider, best first, for as long as the caller keeps asking.

    The named ones come first -- the transit's own two, then closed, widest and half open,
    which between them cover the useful shapes of the gun -- and after them bisections:
    the middle of the widest range nothing has been tried in yet, which is the choice that
    leaves the largest remaining hole as small as possible.  There is no geometry to
    reason from here.  The tip is 200 mm of swing and which openings clear a fixture is not
    a function of the number, so these are chosen to cover the range rather than to be
    individually plausible.

    ``tried`` is the caller's own list and is read live, so an opening it rejects still
    shapes the gaps exactly as a kept one does -- nothing needs to be offered that value
    twice, whichever way it was turned down.  Rounding to ``round_mm`` keeps the values
    sayable on the shop floor, 45 mm rather than 43.7 mm, at the cost of the split being
    slightly off centre.
    """
    for value in [*(named or []), 0.0, widest, widest / 2.0]:
        yield float(value)
    floor = round_mm if round_mm > 0.0 else OPENING_FLOOR_MM
    while True:
        # Closed and widest bound the search whether or not either was kept, so the
        # bisections cover the gun's whole travel even where the named openings cluster.
        known = sorted({0.0, float(widest), *tried})
        gaps = sorted(zip(known, known[1:]), key=lambda g: g[1] - g[0], reverse=True)
        for lo, hi in gaps:
            if hi - lo < 2.0 * floor:
                return                      # the widest gap left is narrower than a step
            mid = (lo + hi) / 2.0
            if round_mm > 0.0:
                mid = round(mid / round_mm) * round_mm
            mid = float(min(max(mid, 0.0), widest))
            if any(abs(mid - seen) < OPENING_TOL_MM for seen in tried):
                continue                    # rounded onto a neighbour: try a wider gap
            yield mid
            break
        else:
            return


def _openings_note(cycle: list[float | None]) -> str:
    """``3 gun openings``, for a log line that has to say how wide a rotation is."""
    if len(cycle) == 1:
        return "the gun as it stands" if cycle[0] is None else "one gun opening"
    return f"{len(cycle)} gun openings"


def _pose_block(cell: Cell, poses: list[tuple[str, np.ndarray]]) -> str | None:
    """Why one of these poses is unusable at the gun's current opening, or ``None``.

    The sampling planner discovers this for itself -- "Goal state is in collision", then a
    run spent failing to seed the goal tree -- but only after the whole time budget has
    gone, and every further run at that opening reaches the same answer just as slowly.
    Two contact queries settle it first.

    Every pose the leg is pinned to is worth checking, not just the goal.  A transit
    leaving a weld is planned at the opening the gun leaves with, and it is the *start*
    that the panel constrains there; a route through a fallback pose has a third.
    """
    for label, q in poses:
        if not cell.in_collision(q):
            continue
        touching = sorted(cell.contact_pairs(q).items(), key=lambda kv: kv[1])
        if touching:
            (first, second), distance = touching[0]
            return (f"the {label} pose has {first} {-distance / cell.man.scale:.1f} mm "
                    f"inside {second}")
        return f"the {label} pose is in collision"
    return None


def _opening_lists(cell: Cell, named: list[float] | None,
                   poses: list[tuple[str, np.ndarray]], *, main_count: int,
                   extra_count: int, round_mm: float = 0.0, log=print) -> GunOpenings:
    """The openings this leg can be planned at, screened and split into the two lists.

    An opening is kept only if it is a distinct value **and** every pose the leg is pinned
    to is clear with the gun held there.  An opening that puts the tip through the panel at
    one end is not a route waiting to be found, and nothing downstream can rescue it: the
    endpoints are where every route at that opening begins and ends.

    A skipped opening costs the lists nothing -- the stream is asked for another -- so the
    two counts are counts of openings that can be planned at.  That is the point of
    screening here rather than inside the run.  "Five openings" meaning "five proposals,
    two of them hopeless" is not a budget anyone can reason about, and under the old order
    the hopeless ones were paid for at full price, one wasted solve each.
    """
    if not cell.gun_joint_name:
        # Nothing to rotate through.  One entry in each list, so both phases run exactly
        # as they did before any of this existed.
        return GunOpenings.pinned(None)
    main_count = max(int(main_count), 1)
    extra_count = max(int(extra_count), 0)
    widest = float(cell.man.gun_opening_max)
    tried: list[float] = []
    kept: list[float] = []
    for value in _opening_stream(named, widest, round_mm, tried):
        if len(kept) >= main_count + extra_count:
            break
        value = float(min(max(value, 0.0), widest))
        if any(abs(value - seen) < OPENING_TOL_MM for seen in tried):
            continue                        # already offered, kept or not
        tried.append(value)
        with cell.gun_opening(value):
            blocked = _pose_block(cell, poses)
        if blocked:
            log(f"      not planning with the gun at {value:g} mm: {blocked}")
            continue
        kept.append(value)
    out = GunOpenings(kept[:main_count], kept[main_count:])
    if log:
        note = ", ".join(f"{v:g}" for v in out.main) or "none"
        extra = ", ".join(f"{v:g}" for v in out.extra)
        log(f"      gun openings: {note} mm for phase one"
            + (f", then {extra} mm for phase two" if extra else ", none left for phase two")
            + (f" ({len(tried) - len(kept)} screened out)"
               if len(tried) > len(kept) else ""))
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


# Which search a solution came from.  Reporting only: they are ranked on cost alone.
SAMPLED = "the sampling planner"
STEERED = "the cartesian tree"


@dataclass
class _Solution:
    """A scored route: penalised cost, plain time, the states, and the gun it was found at.

    The opening travels with the route because the route is only valid at it -- the tip is
    part of the machine that has to fit through the gap -- and because the phases now
    solve at several of them, so which one a route came from is no longer something the
    caller can infer from the order it asked in.

    ``source`` is which search found it, and is reporting only.  Both searches now run on
    every transit and are ranked together, so the winning route no longer says which of
    them produced it -- and that is the one figure that says whether running both is
    buying anything.
    """
    cost: float
    plain: float
    route: list[np.ndarray]
    opening: float | None = None
    source: str = SAMPLED


def _cheapest(candidates: list[_Solution], log) -> _Solution:
    """The lowest-cost solution of a set, with the spread reported where there was one.

    Worth printing rather than just taking: it is the one number that says whether phase
    one's runs are buying anything.  A set whose best and worst are the same cost is a set
    that found the same route every time, and the budget spent sampling for a choice could
    have gone to the phases that search for one at all.

    Ties go to the earlier candidate, which is the earlier opening, the runs having been
    made in the rotation's order.  A route no better than one at the opening the robot
    already holds does not earn a gun change.

    Where the set holds routes from both searches the winner's is named.  The two are
    ranked on one number and the cheaper wins, but which one that was is the only evidence
    there is that the tree is worth the time it now spends on every transit.
    """
    best = min(candidates, key=lambda c: c.cost)
    if log and len(candidates) > 1:
        worst = max(c.cost for c in candidates)
        mixed = (f", from {best.source}"
                 if len({c.source for c in candidates}) > 1 else "")
        log(f"      keeping the best of {len(candidates)} solutions: cost "
            f"{best.cost:.2f} s against {worst:.2f} s for the worst"
            f"{_gun_note(best.opening)}{mixed}")
    return best


def _interleave(first: int, second: int) -> list[bool]:
    """``first`` of one kind and ``second`` of another, each kind evenly spread.

    True for the first kind.  Whichever kind is next is the one whose next turn falls
    earliest as a fraction of its own count, so three of one against two of the other come
    out as 1 2 1 2 1 rather than 1 1 1 2 2.  It matters because the phase these order
    stops at the first solution: queueing one kind behind the other would let the ordering
    decide which kind ever got a turn, rather than the clock.
    """
    out: list[bool] = []
    a = b = 0
    while a < first or b < second:
        if b >= second or (a < first and (a + 0.5) / first <= (b + 0.5) / second):
            out.append(True)
            a += 1
        else:
            out.append(False)
            b += 1
    return out


@dataclass
class _Sampler:
    """The two sampling-planner phases, asked of whichever pair of endpoints is wanted.

    ``_plan_direct`` asks them of the whole transit; ``_recut_outside_band`` asks them
    again of the stretches of a Cartesian route that lie outside the band.  Both want the
    same thing -- every phase-one run spent and the cheapest kept, then phase two only if
    nothing solved at all -- under the same budgets, the same profile settings and the
    same reporting.  Holding that here is what lets the recut use *the* phase one and
    phase two rather than a second copy of them that drifts as this one is tuned.

    ``openings`` is what the two phases draw the gun from: phase one deals its runs round
    ``main``, phase two walks ``extra``.  A caller whose route is already committed to one
    opening passes its own single-value rotation instead -- the recut does, its stretches
    belonging to a route that will be flown at one gun state whatever this finds for them.

    ``message`` is the planner's own words for the last failure.  It is left on the object
    rather than returned because only the last one is ever quoted, and threading it back
    out of every call to say the same thing is noise.
    """
    cell: Cell
    ompl: OmplBudget
    segment_length: float
    check_step: float
    openings: GunOpenings = field(default_factory=lambda: GunOpenings.pinned(None))
    deadline: Deadline = field(default_factory=Deadline)
    continuous_check: bool = False
    log: object = print
    message: str = ""

    def phase_one(self, qa: np.ndarray, qb: np.ndarray,
                  label: str = "phase 1",
                  rotation: list | None = None) -> list[_Solution]:
        """Every run, spent whether or not earlier ones solved.  See ``OmplBudget``.

        The runs are dealt round the rotation, one opening each and back to the first when
        it runs out, so a transit that is hopeless at the opening it arrives with no longer
        spends its whole first phase proving it.  Nothing carries between runs -- each
        builds its own tree from scratch -- so the order decides only which openings get
        the odd extra run where the two counts do not divide.
        """
        if self.ompl.phase_one_runs <= 0:
            return []
        cycle = list(rotation if rotation is not None else self.openings.main) or [None]
        self.log(f"      {label}: {self.ompl.phase_one_runs} runs of "
                 f"{self.ompl.phase_one_seconds:g}s each over {_openings_note(cycle)}, "
                 f"keeping the cheapest that solves")
        found: list[_Solution] = []
        for attempt in range(1, self.ompl.phase_one_runs + 1):
            if self.deadline.ran_out(self.log, f"{label} after {attempt - 1} runs"):
                break
            got = self._run(qa, qb, self.deadline.clamp(self.ompl.phase_one_seconds),
                            f"{label} run {attempt}", cycle[(attempt - 1) % len(cycle)])
            if got is not None:
                found.append(got)
        return found

    def phase_two(self, qa: np.ndarray, qb: np.ndarray,
                  label: str = "phase 2", rotation: list | None = None,
                  cartesian_runs: int = 0, cartesian_run=None) -> list[_Solution]:
        """Runs until one solves, there being nothing to choose between.

        Where phase one cycles the openings it was given, this walks the ones it was not:
        phase one has already had runs at everything in its own rotation, and a longer
        search where three short ones just failed is the narrower of the two bets on
        offer.  With nothing left to walk there is nothing here to do, and saying so is
        better than a run that repeats one.

        ``cartesian_run`` is the short Cartesian search, spread evenly through the
        sampling runs rather than queued behind them.  Either kind ends the phase by
        solving, so an order that ran one kind out first would decide by the ordering
        which kind ever got a turn.  Each kind walks the openings from the top of the
        list, so both spend their first and best attempt on the best opening left.
        """
        cycle = list(rotation if rotation is not None else self.openings.extra)
        if not cycle:
            self.log(f"      {label}: no gun opening left that phase one has not already "
                     f"had its runs at")
            return []
        ompl_runs = max(self.ompl.phase_two_max_runs, 0)
        tree_runs = max(cartesian_runs, 0) if cartesian_run is not None else 0
        if not ompl_runs and not tree_runs:
            return []
        self.log(f"      {label}: up to {ompl_runs} runs of "
                 f"{self.ompl.phase_two_seconds:g}s each over {_openings_note(cycle)}"
                 + (f", with {tree_runs} Cartesian search"
                    f"{'' if tree_runs == 1 else 'es'} spread through them" if tree_runs
                    else "")
                 + ", stopping at the first solution")
        used = tree_used = 0
        for is_ompl in _interleave(ompl_runs, tree_runs):
            if self.deadline.ran_out(self.log, f"{label} after {used + tree_used} runs"):
                return []
            if is_ompl:
                used += 1
                found = self._run(qa, qb,
                                  self.deadline.clamp(self.ompl.phase_two_seconds),
                                  f"{label} run {used}", cycle[(used - 1) % len(cycle)])
            else:
                tree_used += 1
                found = cartesian_run(cycle[(tree_used - 1) % len(cycle)], tree_used)
            if found is not None:
                return [found]
        return []

    def solve(self, qa: np.ndarray, qb: np.ndarray, *,
              label: str = "", rotation: list | None = None) -> _Solution | None:
        """Both phases in order, cheapest solution or ``None``.

        For a caller that has nothing to insert between them.  ``_plan_direct`` does --
        the Cartesian tree goes there -- so it drives the two phases itself.
        """
        prefix = f"{label} " if label else ""
        candidates = (self.phase_one(qa, qb, f"{prefix}phase 1", rotation=rotation)
                      or self.phase_two(qa, qb, f"{prefix}phase 2", rotation=rotation))
        return _cheapest(candidates, self.log) if candidates else None

    def _run(self, qa: np.ndarray, qb: np.ndarray, planning_time: float,
             label: str, opening: float | None = None) -> _Solution | None:
        """One sampling-planner solve, scored and reported.  ``None`` when it did not solve."""
        with self.cell.gun_opening(opening):
            # The gun is not an inverse-kinematics variable but it is very much a state
            # one, and the planner reads the state the environment is holding rather than
            # anything in the program it is handed.  Loading a pose here is what puts this
            # run's opening in front of it; before, the run inherited whatever the last
            # collision query happened to have left there.
            self.cell.set_state(qa)
            profiles = ProfileDictionary()
            profiles.addProfile(OMPL_NAMESPACE, "DEFAULT",
                                _ompl_profile(self.segment_length, planning_time,
                                              self.continuous_check, self.log))
            request = PlannerRequest()
            request.env = self.cell.env
            request.instructions = _make_program(self.cell, qa, qb)
            request.profiles = profiles
            t0 = time.time()
            response = OMPLMotionPlanner(OMPL_NAMESPACE).solve(request)
            dt = time.time() - t0
            note = _gun_note(opening)
            if not response.successful:
                self.message = str(response.message)
                self.log(f"      {label}{note}: {self.message} ({dt:.1f}s)")
                return None
            raw = _extract(response.results)
            fault = _route_fault(self.cell, raw, self.check_step)
            if fault is not None:
                self.message = (f"returned a route that is not clear under this cell's "
                                f"own check ({fault})")
                self.log(f"      {label}{note}: {self.message} ({dt:.1f}s)")
                return None
            cost, plain = _path_cost(self.cell, raw, self.check_step)
        self.log(f"      {label}{note}: solved in {dt:.1f}s ({len(raw)} raw points, "
                 f"cost {cost:.2f} s against {plain:.2f} s unpenalised)")
        return _Solution(cost, plain, raw, opening)


def _plan_direct(cell: Cell, qa: np.ndarray, qb: np.ndarray, *, ompl: OmplBudget,
                 segment_length: float, check_step: float,
                 continuous_check: bool = False,
                 shortcut_seconds: float, polish_seconds: float,
                 zone: LinearZone | None = None,
                 cartesian: "CartesianBudget | None" = None,
                 relocate: "Relocation | None" = None,
                 openings: "GunOpenings | None" = None,
                 deadline: "Deadline | None" = None, log=print,
                 record: list | None = None) -> tuple[list[Run], float | None]:
    """A route from ``qa`` to ``qb``, and the gun opening it is to be flown at.

    The opening comes back with the route because it is chosen here: the phases solve at
    several and the cheapest solution wins whichever one it came from, so the caller no
    longer knows it from the order it asked in.  What the caller decides is which openings
    are on offer, and a caller that has already fixed one passes ``GunOpenings.pinned``.
    """
    openings = openings or GunOpenings.pinned(None)
    deadline = deadline or Deadline()

    # The clear straight line first, at each opening in turn.  It is normally the best
    # answer there is -- a sampling planner asked to improve on it can only return it
    # again -- and a collision sweep is nothing beside a phase of solves, so it is worth
    # asking of every opening before any of them is searched.
    for opening in openings.main:
        with cell.gun_opening(opening):
            if cell.segment_collides(qa, qb, max_step=check_step):
                continue
            # "Clear" and "sensible" part company when the line grazes a panel, so when
            # the penalty says this one does, the same pass that stands other routes off
            # is given a chance to bow it away -- there is nothing for OMPL to do here,
            # but plenty for relocation.
            stand_off = False
            if cell.penalty is not None and cell.penalty.enabled and shortcut_seconds > 0:
                raw = cell.move_time(qa, qb)
                cost = cell.segment_cost(qa, qb, max_step=check_step, stops=True)
                stand_off = cost > raw * 1.05
                if stand_off:
                    log(f"      direct move is clear but runs close to the parts (cost "
                        f"{cost:.2f} s against {raw:.2f} s unpenalised)"
                        f"{_gun_note(opening)}; standing it off")
            try:
                # _finish fills the interior in, which a two-point path needs before
                # relocation has anything to move.
                runs = _finish(cell, [qa, qb], zone=zone, relocate=relocate,
                               shortcut_seconds=shortcut_seconds if stand_off else 0.0,
                               polish_seconds=polish_seconds if stand_off else 0.0,
                               check_step=check_step, log=log)
            except PlanningError as exc:
                # Clear as a joint chord is not the same as flyable: with both ends in the
                # band the move ships linear, and _finish verifies it along the tool's
                # line.  This used to raise straight out and write the whole gun opening
                # off without a single sampling run, where a route round the obstacle may
                # well exist.  The search below covers the remaining openings as well, so
                # there is nothing lost in leaving the sweep here.
                log(f"      the direct move is clear as a joint move but not as planned"
                    f"{_gun_note(opening)} ({exc}); searching for a route instead")
                break
            _capture(record, [qa, qb])
            return runs, opening

    # RRTConnect returns the first path it finds, and which homotopy class that lands in
    # is luck -- one run goes over the fixture, the next threads behind it.  The
    # optimisation pass afterwards can shorten a route and stand it off, but it cannot move
    # it to the other side of an obstacle, so whichever class arrives here is the one that
    # ships.  Sampling several solutions and keeping the best-scoring one is therefore the
    # only stage that can make that choice at all -- and since the runs are being spent
    # anyway, dealing them round the openings makes the same choice over a wider set.
    sampler = _Sampler(cell=cell, ompl=ompl, segment_length=segment_length,
                       check_step=check_step, openings=openings, deadline=deadline,
                       continuous_check=continuous_check, log=log)
    candidates = sampler.phase_one(qa, qb)

    tree = cartesian if (cartesian is not None and cartesian.enabled
                         and zone is not None and zone.enabled) else None
    if tree is not None and not deadline.expired:
        # Run whatever phase one did, and rank what it finds against phase one's own.
        #
        # The two searches do not differ in whether they succeed so much as in what they
        # come back with.  OMPL samples joint space uniformly and returns the first route
        # its seed leads it to, so beside a panel it usually returns the one that stands
        # well off it -- there is nothing drawing the search into the gap.  The tree
        # steers along the tool's own line with orientations drawn near the endpoints',
        # which concentrates it into exactly that corridor.  So a phase one that solved is
        # not evidence that the tree had nothing better to offer, and the only way to find
        # out is to run both and score them together, which _cheapest below does over the
        # whole set.  Before this the tree was a fallback and a transit phase one solved
        # never saw it, which meant the cheaper route of the two was never even costed.
        #
        # What that costs is the tree's full budget on every transit that reaches here
        # rather than only on the ones nothing else could reach: --cartesian-solve-seconds
        # until it solves and then --cartesian-min-seconds looking for better ones,
        # whether or not a route is already in hand.  The segment clock is what bounds it.
        #
        # Still before phase two, which is unchanged in running only when nothing has been
        # found at all: every route the tree finds is also a joint-space route, so it
        # searches a strict *subset* of what phase two searches and cannot stand in for it.
        # Still gated on the zone, because a route made of straight moves only earns its
        # cost where linear motion was wanted in the first place.
        found = _cartesian_solutions(cell, qa, qb, tree, check_step, openings.main, log,
                                     deadline=deadline)
        if found:
            # The best of the set is the only one recut: the recut runs the sampling
            # phases on every stretch outside the band, far too dear to spend on routes
            # that are then thrown away.  It is also what makes the tree's route
            # comparable with phase one's -- the stretches where it has no business being
            # a chain of straight moves are replanned before anything is scored.
            best = _cheapest(found, log)
            if tree.recut:
                best = _recut_solution(cell, best, zone=zone, sampler=sampler,
                                       check_step=check_step, log=log)
            candidates.append(best)

    if not candidates:
        # Nothing to choose between at this point, so the goal changes from a good route to
        # any route, and the first one that arrives ends the phase.
        tree_run = None
        if tree is not None and tree.phase_two_enabled:
            def tree_run(opening, run):
                """One short Cartesian search, for phase two to spread among its own."""
                got = _cartesian_route(cell, qa, qb, tree,
                                       seconds=deadline.clamp(tree.phase_two_seconds),
                                       seed=tree.seed + PHASE_TWO_SEED_OFFSET + run,
                                       opening=opening, check_step=check_step,
                                       label=f"phase 2 cartesian run {run}", log=log)
                if got is not None and tree.recut:
                    got = _recut_solution(cell, got, zone=zone, sampler=sampler,
                                          check_step=check_step, log=log)
                return got
        candidates = sampler.phase_two(
            qa, qb, cartesian_runs=tree.phase_two_runs if tree_run is not None else 0,
            cartesian_run=tree_run)

    if not candidates:
        raise PlanningError(
            f"freespace transit failed after {ompl.worst_case_runs} attempts: "
            f"{sampler.message}")

    best = _cheapest(candidates, log)
    _capture(record, best.route)
    with cell.gun_opening(best.opening):
        # Shortcut before reducing: the dense path gives the cuts somewhere to land.
        out = _finish(cell, best.route, zone=zone, relocate=relocate,
                      shortcut_seconds=shortcut_seconds,
                      polish_seconds=polish_seconds, check_step=check_step, log=log)
    log(f"      reduced to {sum(len(r.states) for r in out)} points in "
        f"{len(out)} run{'' if len(out) == 1 else 's'}{_gun_note(best.opening)}")
    return out, best.opening


# Phase two's Cartesian searches start from seeds nothing else uses, so a transit whose
# earlier searches all failed does not spend phase two repeating them exactly.  They run at
# other openings in any case; this makes them a different search of the cell as well.
PHASE_TWO_SEED_OFFSET = 1000


def _cartesian_route(cell: Cell, qa: np.ndarray, qb: np.ndarray, budget: "CartesianBudget",
                     *, seconds: float, seed: int, opening: float | None,
                     check_step: float, label: str, log) -> _Solution | None:
    """One Cartesian search at one gun opening, scored as phase one scores its own.

    The gun is not a detail of the check here but part of the shape being steered around:
    a tip 200 mm open sweeps a corridor a tip closed never enters, and the tree's every
    edge is a straight move of that tool.  So a search that failed says something about
    the opening it ran at and nothing about any other.
    """
    from .cartesian import plan_cartesian    # deferred: cartesian.py reads this module
    t0 = time.time()
    with cell.gun_opening(opening):
        cell.set_state(qa)
        try:
            route = plan_cartesian(cell, qa, qb, max_step=check_step, log=log,
                                   budget=replace(budget, seconds=seconds, seed=seed))
        except PlanningError as exc:
            log(f"      {exc} ({label}{_gun_note(opening)}, {time.time() - t0:.1f}s)")
            return None
        cost, plain = _path_cost(cell, route, check_step)
    log(f"      {label}{_gun_note(opening)}: solved in {time.time() - t0:.1f}s "
        f"({len(route)} points, cost {cost:.2f} s against {plain:.2f} s unpenalised)")
    return _Solution(cost, plain, route, opening, STEERED)


def _recut_solution(cell: Cell, best: _Solution, *, zone: "LinearZone",
                    sampler: _Sampler, check_step: float, log) -> _Solution:
    """``best`` with its out-of-band stretches replanned, or ``best`` where none were.

    Held at the route's own opening throughout.  Every question the recut asks -- which
    states read near the panel, whether a chord between two cuts is clear, what a joining
    route costs -- is a question about a cell with the tip in a particular place, and the
    answers are being spliced into a route that will be flown with it there.
    """
    with cell.gun_opening(best.opening):
        route = _recut_outside_band(cell, best.route, zone=zone, sampler=sampler,
                                    check_step=check_step, opening=best.opening, log=log)
        if route is best.route:
            return best
        cost, plain = _path_cost(cell, route, check_step)
    log(f"      cartesian tree after the recut: {len(route)} points, cost {cost:.2f} s "
        f"against {plain:.2f} s unpenalised")
    return _Solution(cost, plain, route, best.opening, best.source)


def _cartesian_solutions(cell: Cell, qa: np.ndarray, qb: np.ndarray,
                         budget: "CartesianBudget", check_step: float,
                         rotation: list, log,
                         deadline: "Deadline | None" = None) -> list[_Solution]:
    """Cartesian-tree routes for one transit, each scored as phase one scores its own.

    Searches run from successive seeds until one solves or ``budget.seconds`` has passed,
    then carry on until ``budget.min_seconds`` has passed, both counted from the start of
    the first.  A tree search, like RRTConnect, settles for whichever route its seed leads
    it to first, so more of them is the only way to have a choice.  Each search is given
    only the time left, so neither figure is overrun by more than the search in hand takes
    to notice.

    The runs are dealt round the same rotation phase one used, one opening each.  There is
    no endpoint screen here any more: ``_opening_lists`` has already established that both
    ends are clear at every opening in that rotation, which is the one failure no further
    seed could change.

    Called on every transit, not only on the ones phase one failed, so these budgets are
    now spent whether or not a route is already in hand -- see ``_plan_direct``.
    """
    rotation = list(rotation) or [None]
    deadline = deadline or Deadline()
    log(f"      cartesian tree: up to {budget.seconds:g}s searching linear space for a "
        f"solution, then until {budget.min_seconds:g}s for better ones, over "
        f"{_openings_note(rotation)}")
    start = time.time()
    found: list[_Solution] = []
    run = 0
    while True:
        left = deadline.clamp(
            start + (budget.min_seconds if found else budget.seconds) - time.time())
        if left <= 0.0:
            if deadline.expired:
                deadline.ran_out(log, f"the cartesian tree after {run} runs")
            return found
        run += 1
        got = _cartesian_route(cell, qa, qb, budget, seconds=left,
                               seed=budget.seed + run - 1,
                               opening=rotation[(run - 1) % len(rotation)],
                               check_step=check_step,
                               label=f"cartesian tree run {run}", log=log)
        if got is not None:
            found.append(got)


def _far_stretches(near: list[bool]) -> list[tuple[int, int]]:
    """Index pairs bounding each stretch of the route that reads outside the band.

    The bounds are the first and last state of the stretch itself, not the near states
    either side of it, which is what puts the cut *just outside* the band: the near
    stretches keep every state the tree validated for them, and what is given up is only
    what was already out of range.

    A stretch with nothing between its bounds is left out.  There is no route to replan
    there -- the two cuts are adjacent, the move between them is already the only move --
    and it is the shape the apex of a retract makes as it clips out of the band for an
    instant, so it would otherwise buy a full sampling search for no change at all.
    """
    return [(i, j) for i, j in _runs_of(near, False) if j - i >= 2]


def _runs_of(flags: list[bool], value: bool) -> list[tuple[int, int]]:
    """``(first, last)`` index of every maximal run of ``value`` in ``flags``."""
    out: list[tuple[int, int]] = []
    i = 0
    while i < len(flags):
        if flags[i] != value:
            i += 1
            continue
        j = i
        while j + 1 < len(flags) and flags[j + 1] == value:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def _recut_outside_band(cell: Cell, route: list[np.ndarray], *, zone: LinearZone,
                        sampler: _Sampler, check_step: float,
                        opening: float | None = None, log=print) -> list[np.ndarray]:
    """Cut a Cartesian-tree route at the band boundary and re-plan what lies outside it.

    The tree searches linear space over the whole transit, band or no band, because a
    straight tool move is the only edge it has and it still has to reach the far endpoint.
    Only the near-panel half of what comes back was ever the point.  Outside the band a
    route made of straight tool moves is a route drawn from a strict subset of what the
    arm can do, and nothing recommends it there: the long withdrawal a transit makes
    between two welds a few millimetres apart is exactly the shape a joint move crosses
    directly, and the tree cannot produce that move because the tool would not travel in a
    straight line while it was made.

    So each stretch that reads far is handed back to the sampling planner, under the same
    phase one and phase two the transit itself gets.  The two ends of the stretch become
    waypoints of the finished route; every state between them is dropped, whatever profile
    it would have flown under, because a linear move outside the band is precisely what
    this is here to be rid of.

    Two things it declines to search rather than search badly:

    * **A stretch whose chord is already clear** is answered by that chord, with no run
      spent.  The straight joint move between two states is the quickest route there can
      be between them, so a planner asked to improve on it can only return it again --
      the same reasoning ``_plan_direct`` opens with.
    * **A route with no near states at all** is left alone entirely.  There would be
      nothing to keep, so the "stretch" is the whole transit, and that is the query phase
      one has just failed on. Asking it again with the same budget is the one thing here
      guaranteed to be a waste.

    A stretch the planner cannot join keeps its Cartesian states.  That is a route which
    is known to work, and a worse shape than a joint move is not a reason to have no route
    at all.

    ``opening`` is the gun state the route was found at, and the phases are pinned to it
    rather than left to rotate.  A joining route is spliced into this route, so one found
    with the tip somewhere else is not a route at all; it is two moves nothing has
    checked.  The caller holds the cell at that opening as well -- see ``_recut_solution``
    -- which is what the clearance reads and chord checks here are answered under.
    """
    if len(route) < 3 or not zone.enabled:
        return route
    if log and len(route) > 200:
        log(f"      recut: reading clearance at {len(route)} points to find where the "
            f"route leaves the {zone.near_mm:g} mm band")
    near = [_reads_near(cell, q, zone.near_mm) for q in route]
    if not any(near):
        log("      recut: no state of the route reads near the panel, so there is nothing "
            "to keep and nothing phase one has not already tried; leaving it as it is")
        return route

    cuts = _far_stretches(near)
    if not cuts:
        log("      recut: the route holds no stretch outside the band worth replanning")
        return route
    one = len(cuts) == 1
    log(f"      recut: the route leaves the band in {len(cuts)} "
        f"stretch{'' if one else 'es'}, {'which is' if one else 'each'} replanned as "
        f"joint motion")

    plans: list[tuple[int, int, list[np.ndarray]]] = []
    for n, (i, j) in enumerate(cuts, 1):
        dropped = route[i:j + 1]
        head = (f"      recut {n} of {len(cuts)}: points {i} to {j} of {len(route)}, "
                f"{_tcp_travel(cell, dropped):.0f} mm of tool travel")
        if not cell.segment_collides(route[i], route[j], max_step=check_step):
            log(f"{head}; the joint move between the cuts is clear, so it is the answer")
            plans.append((i, j, [route[i], route[j]]))
            continue
        log(f"{head}; searching for a joining route")
        found = sampler.solve(route[i], route[j], label=f"recut {n}",
                              rotation=[opening])
        if found is None:
            log(f"      recut {n}: no joining route, so this stretch keeps the Cartesian "
                f"one ({sampler.message})")
            continue
        cost, bridge = found.cost, found.route
        # The splice assumes the joining route begins and ends at the states it was asked
        # for.  It does -- the program is built from them -- but a route that did not
        # would leave two moves in the finished path that nothing has ever checked, and
        # that is worth a line of arithmetic rather than a trust.
        if not (np.allclose(bridge[0], route[i]) and np.allclose(bridge[-1], route[j])):
            log(f"      recut {n}: the joining route does not start and end at the cuts, "
                f"so it cannot be spliced in; keeping the Cartesian stretch")
            continue
        was, _ = _path_cost(cell, dropped, check_step)
        log(f"      recut {n}: joined with {len(bridge)} points at cost {cost:.2f} s, "
            f"against {was:.2f} s for the {len(dropped)} Cartesian points it replaces")
        plans.append((i, j, bridge))

    if not plans:
        return route
    # Spliced back to front so that the indices of the stretches still to come are the
    # ones they were found at.  Each bridge already begins and ends on the cut states, so
    # the slice it replaces includes them.
    out = list(route)
    for i, j, bridge in reversed(plans):
        out[i:j + 1] = bridge
    log(f"      recut: {len(plans)} of {len(cuts)} "
        f"stretch{'' if one else 'es'} replanned, {len(route)} points now {len(out)}")
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
    """Settings for turning the near-panel parts of a transit into linear motion.

    The band is the whole of the decision.  A move is linear when either of its ends reads
    within ``near_mm`` of the parts, asked of every move and of nothing larger; there is no
    per-leg qualification a route has to pass before any of its moves may be linear.
    """
    near_mm: float = 0.0            # clearance at or under which the route counts as near
    linear_speed_mm_s: float = 0.0  # tool speed cap; 0 leaves linear moves costed on joints
    crossing_penalty_s: float = 0.0  # flat costing-only surcharge on each move that
                                     # reaches into the band from outside it; 0 charges none
    introduce_mm: float = 0.0       # furthest clearance a refinement pass may introduce a
                                    # linear move from; 0 places no limit.  See reach_mm

    @property
    def enabled(self) -> bool:
        return self.near_mm > 0.0

    @property
    def reach_mm(self) -> float:
        """How far out a pass may introduce a linear move, in mm.  See ``overreaches``.

        Never under the band itself.  A move with both ends inside the band is linear by
        the endpoint rule wherever it runs, so there is nothing for a pass to introduce
        there; a limit tighter than the band would refuse the reshaping ``demotes``
        deliberately permits and leave the passes unable to touch a near-panel stretch at
        all.  0 switches the rule off.
        """
        return max(self.introduce_mm, self.near_mm) if self.introduce_mm > 0.0 else 0.0


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
    model = MotionModel(cell, max_step=check_step, zone=zone)
    stagetrace.route("refining a leg planned earlier without a budget")
    stagetrace.note(_trace_zone(zone))
    stagetrace.stage("input", model, pts)
    refined = _refine(model, pts, shortcut_seconds=shortcut_seconds,
                      polish_seconds=polish_seconds, relocate=relocate, log=log)
    runs = _split_runs(model, refined)
    _traced_verify(model, runs, "refined leg", log=log)
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
    fused = d.get("clearance_fused", 0)
    # What the shared walk saved, in the currency it saved it in: a fused query rode on a
    # state load the collision check had already paid for, so without it the loads would
    # have been this much higher.
    share = f", {fused} of them off a walk already made" if fused else ""
    log(f"      work: {d['clearance']} clearance queries ({served:.0f}% of {asked} served "
        f"from cache{share}), {d['collision_tests']} collision tests, "
        f"{d['state_loads']} state loads, {d['fk']} forward kinematics, "
        f"in {elapsed:.1f}s")


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
    # One sampling pass.  There were two while a per-leg gate stood here: the fill has to
    # follow the curve each move will really be flown along, which needs a model, and the
    # model could not be built until the gate had said whether the leg was to have a zone
    # at all -- so the route was sampled once along the chord to answer that, and again
    # along the real curves afterwards.  Nothing asks that question now.  The band answers
    # it per move through MotionModel.motion, so the model exists before anything is
    # sampled and the profile-aware fill is the only fill.
    model = MotionModel(cell, max_step=check_step, zone=zone)
    dense, anchors = _densify_marked(cell, path, check_step, model=model)
    stagetrace.route(f"planned route, refinement budgets shortcut {shortcut_seconds:g} s, "
                     f"polish {polish_seconds:g} s"
                     + ("" if shortcut_seconds > 0 or polish_seconds > 0 else
                        " (unrefined for now: may be refined again later or discarded)"))
    stagetrace.note(_trace_zone(zone))
    stagetrace.stage("solver", model, path, sources=range(len(path)))
    stagetrace.stage("densify", model, dense, sources=anchors)
    refined = _refine(model, dense, shortcut_seconds=shortcut_seconds,
                      polish_seconds=polish_seconds, relocate=relocate,
                      anchors=anchors, log=log)
    runs = _split_runs(model, refined)
    _traced_verify(model, runs, "planned route", log=log)
    _report_work(cell, before, time.time() - t0, log)
    if log:
        lin = sum(1 for r in runs if r.motion == LIN)
        if lin:
            moves = sum(len(r.states) - 1 for r in runs if r.motion == LIN)
            log(f"      {lin} linear runs over {moves} moves, "
                f"{len(runs) - lin} joint runs, {len(refined)} waypoints")
    return runs


def _trace_zone(zone: "LinearZone | None") -> str:
    """Which band is in force, as the stage trace reports it."""
    if zone is None or not zone.enabled:
        return "linear band: off, every move is PTP"
    return (f"linear band: {zone.near_mm:g} mm, a move is LIN when either end reads "
            f"inside it (reach {zone.reach_mm:g} mm)")


def _traced_verify(model: "MotionModel", runs: list[Run], where: str, log=None) -> None:
    """:func:`_verify_runs`, with its verdict and the shipped split in the stage trace."""
    try:
        _verify_runs(model, runs, where, log=log)
    except PlanningError as exc:
        stagetrace.note(f"verify: FAILED, route discarded: {exc}")
        raise
    stagetrace.note("verify: passed; ships as " + ", ".join(
        f"{r.motion} x{len(r.states) - 1}" for r in runs))


def _verify_runs(model: "MotionModel", runs: list[Run], where: str, log=None) -> None:
    """Check a finished route along the path each of its moves will really take.

    The optimisation passes check the moves they *propose*, and only those.  ``shortcut``
    never offers an adjacent pair -- ``_try_cut`` returns early on ``j - i < 2`` -- and
    ``simplify``'s reach loop runs ``while j > i + 1``, so the pair it finally settles on
    is appended without a check.  A move that came out of ``_densify_marked`` and that
    nothing happened to replace therefore leaves here carrying only the guarantee the
    route arrived with.

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
    pa, pb = cell.pose_mm(a), cell.pose_mm(b)
    travel = float(np.linalg.norm(pb[:3, 3] - pa[:3, 3]))
    step = model.tool_step
    chord = ("clear" if not cell.segment_collides(a, b, max_step=model.max_step)
             else "also blocked")
    head = (f"{travel:.0f} mm of tool travel against a {step:g} mm linear step, "
            f"joint chord {chord}")
    try:
        chain = plan_linear(cell, pa, pb, a, step_mm=step, step_rad=model.max_step)
    except PlanningError as exc:
        return f"{head}; {exc}"
    ends = _line_end_fault(chain, a, b)
    if ends is not None:
        return f"{head}; {ends}"
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
    points = [cell.tcp_at(q) for q in states]
    raw = float(sum(np.linalg.norm(b - a) for a, b in zip(points, points[1:])))
    return raw / cell.man.scale


def _chain_scan(cell: Cell, chain: list[np.ndarray], check_step: float,
                factors: bool = False) -> tuple[bool, list[float] | None]:
    """Whether the joint gaps between a tracked line's stations are blocked, and the
    penalty factors read off the same walk.

    ``plan_linear`` clears the stations it places; this clears the gaps between them.

    The linear counterpart of :meth:`weldpath.cell.Cell.segment_scan`, and it exists for
    the same reason: a caller that installs this chain wants a factor at every point of
    it, and asking for them afterwards loads every state a second time to measure a
    clearance the contact test had the transforms for.  The gaps are walked here either
    way, so the second question is answered where the first one already is.

    The factors returned are those of the **chain points**, one per point, and not of the
    grid inside each gap.  The chain is what gets installed -- its points become path
    points and are what the cost is summed over -- while the grid inside a gap exists to
    prove the gap clear and is thrown away.  Returning the finer sampling would price a
    move by states that are nowhere in it.

    Each gap's scan reports both its ends, so the shared point between two gaps is read
    from the first of them and the last point from the final gap; taking both would be
    the same figure from the same cache, but the pairing would no longer be one for one
    with ``chain``, which is what the caller indexes by.
    """
    out: list[float] | None = [] if factors else None
    for a, b in zip(chain, chain[1:]):
        hit, facs = cell.segment_scan(a, b, max_step=check_step, factors=factors)
        if hit:
            return True, None
        if factors:
            out.append(facs[0])
    if factors:
        out.append(facs[-1] if len(chain) > 1 else cell.penalty_factor(chain[0]))
    return False, out


def _reads_near(cell: Cell, q: np.ndarray, near_mm: float) -> bool:
    """Whether ``q`` is within ``near_mm`` of the parts, as far as the clearance query knows.

    ``clearance_mm`` returns the probe distance itself when nothing lies within the probe,
    so a reading at the probe is "nothing seen", not a distance, and it has to be kept out
    of an at-or-under test.  Compared as a distance it passed whenever the probe was no
    wider than the band: ``ToolpathPlanner`` sized the two equal, every state of every leg
    read as near, and nearly every move shipped linear.

    The caller has to have sized the probe past ``near_mm`` -- ``ToolpathPlanner`` does --
    or states clear by more than the probe and less than the band are missed.  The probe
    only ever grows, so once it is past the band this answer no longer depends on it.
    """
    reading = cell.clearance_mm(q)
    return reading < cell.probe_mm and reading <= near_mm


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
def rotation_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Shortest rotation angle, in radians, taking ``Ra`` onto ``Rb``.

    Taken as ``atan2(2 sin(theta), 2 cos(theta))`` from the relative rotation's skew part
    and trace, not as ``arccos`` of the trace alone.  ``arccos`` is flat at 1, so it cannot
    resolve a small angle: a rotation compared with itself, whose trace is 3 give or take
    rounding, came back as anything up to 5.6e-8 rad -- above 1e-9 for 480 of 2000 random
    rotations.  The Cartesian tree decides its two halves have met on a turn under 1e-9,
    so a quarter of the time the test could not pass at all: the connect sat on the goal
    pose adding zero-length steps until the budget ran out, 3982 nodes from one sample on
    a 300 mm straight retract that joins in 6 once the angle reads true.

    The skew part is itself a difference of near-equal entries, but one that cancels to
    rounding error, so the angle it gives near zero is of that order rather than its
    square root.
    """
    R = Ra.T @ Rb
    s = np.linalg.norm([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return float(np.arctan2(s, np.trace(R) - 1.0))


def interpolate_pose(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear in position, shortest-arc slerp in orientation."""
    out = np.eye(4)
    out[:3, 3] = a[:3, 3] + t * (b[:3, 3] - a[:3, 3])
    Ra, Rb = a[:3, :3], b[:3, :3]
    R = Ra.T @ Rb
    angle = rotation_angle(Ra, Rb)
    if angle < 1e-9:
        out[:3, :3] = Ra
        return out
    skew = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    norm = float(np.linalg.norm(skew))
    if norm > 1e-6:
        axis = skew / norm                  # |skew| = 2 sin(angle)
    else:
        # A half turn: the skew part vanishes and carries no axis.  R = 2 a a^T - I there,
        # so the axis is the largest column of (R + I), whichever sign it comes out with --
        # a half turn about a and about -a are the same rotation.
        M = R + np.eye(3)
        axis = M[:, int(np.argmax(np.linalg.norm(M, axis=0)))]
        axis = axis / np.linalg.norm(axis)
    th = angle * t
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    out[:3, :3] = Ra @ (np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K))
    return out


# Most a joint may move between two stations of one straight tool move, and furthest a
# tracked line may end from the state it was asked to reach.  Measured on Path 1's linear
# moves the stations moved 0.031 rad at most and the ends landed within 1.1e-4 rad, so
# both leave wide headroom; what they catch is a branch flip or a whole extra turn, which
# the controller cannot make inside one straight move.
LINE_JUMP_RAD = 0.5
LINE_END_TOL_RAD = 1e-2


def line_stations(pose_a: np.ndarray, pose_b: np.ndarray, step_mm: float,
                  step_rad: float = 0.0) -> int:
    """How many equal steps a straight tool move is divided into.

    Enough that no step carries the tool further than ``step_mm`` or turns it further than
    ``step_rad``.  Travel alone used to decide it, so a move that mostly turned the tool --
    a wrist reorientation on the spot -- got its two ends and nothing between them, and was
    checked only along the joint chord joining those.  An infinite ``step_mm`` is
    ``--check-step-mm 0``, which collapses the stations onto the ends; the turn is not
    allowed to bring them back, since that setting asks for the joint grid alone.
    """
    dist = float(np.linalg.norm(pose_b[:3, 3] - pose_a[:3, 3]))
    n = int(np.ceil(dist / max(step_mm, 1e-6)))
    if np.isfinite(step_mm) and step_rad > 0.0:
        turn = rotation_angle(pose_a[:3, :3], pose_b[:3, :3])
        n = max(n, int(np.ceil(turn / step_rad)))
    return max(1, n)


def _line_end_fault(chain: list[np.ndarray], a: np.ndarray, b: np.ndarray) -> str | None:
    """Why a tracked line is not the move from ``a`` to ``b``, or ``None`` if it is.

    ``plan_linear`` solves every station afresh, the two ends included, and callers pin
    the ends back onto ``a`` and ``b`` before checking the gaps.  Pinning without looking
    hid the case where tracking the line from ``a`` lands on another arm configuration, or
    on the same one a whole turn round, than ``b``: the last gap then held a wrist flip,
    checked as a joint chord and passed, in a move the controller drives as a line.
    """
    for label, got, want in (("starts", chain[0], a), ("ends", chain[-1], b)):
        off = np.abs(np.asarray(got, dtype=float) - np.asarray(want, dtype=float))
        if float(off.max()) > LINE_END_TOL_RAD:
            j = int(np.argmax(off))
            return (f"the line {label} on a different arm configuration, joint {j + 1} "
                    f"{np.degrees(off[j]):.1f} deg away")
    return None


def plan_linear(cell: Cell, pose_a: np.ndarray, pose_b: np.ndarray, seed: np.ndarray, *,
                step_mm: float = 50.0, step_rad: float = 0.0,
                allow_collision: bool = False) -> list[np.ndarray]:
    """Joint states following the straight Cartesian line from ``pose_a`` to ``pose_b``.

    Poses are world 4x4 in manifest units.  Stations are placed by :func:`line_stations`.
    A station further than ``LINE_JUMP_RAD`` from the one before it on any joint is a
    branch flip the controller cannot make mid-line, and is refused like a station with no
    solution -- the rule ``cartesian._steer`` already held its edges to.
    """
    n = line_stations(pose_a, pose_b, step_mm, step_rad)
    dist = float(np.linalg.norm(pose_b[:3, 3] - pose_a[:3, 3]))
    out: list[np.ndarray] = []
    current = np.asarray(seed, dtype=float)
    for k in range(n + 1):
        pose = interpolate_pose(pose_a, pose_b, k / n)
        q = cell.solve_pose(pose, [current], require_collision_free=not allow_collision)
        if q is None:
            raise PlanningError(
                f"no collision-free IK at {100.0 * k / n:.0f}% along a {dist:.0f} mm "
                f"linear move")
        if k and float(np.max(np.abs(q - current))) > LINE_JUMP_RAD:
            raise PlanningError(
                f"the arm changes configuration at {100.0 * k / n:.0f}% along a "
                f"{dist:.0f} mm linear move")
        out.append(q)
        current = q
    return out
