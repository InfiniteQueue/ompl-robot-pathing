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
def simplify(cell: Cell, path: list[np.ndarray], max_step: float = 0.05
             ) -> list[np.ndarray]:
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
            factors[k] = cell.penalty_factor(path[k])
        return factors[k]

    def polyline_cost(i: int, j: int) -> float:
        # Each retained hop is its own stop-to-stop move.  The hops are short, so the
        # endpoint factors describe them well enough without sampling their interiors.
        return sum(cell.move_time(path[k], path[k + 1]) * max(factor(k), factor(k + 1))
                   for k in range(i, j))

    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if cell.segment_collides(path[i], path[j], max_step=max_step):
                j -= 1
                continue
            chord = cell.segment_cost(path[i], path[j], max_step=max_step,
                                      fa=factor(i), fb=factor(j), stops=True)
            if chord > polyline_cost(i, j) + 1e-9:
                j -= 1
                continue
            break
        out.append(path[j])
        i = j
    return out


def _hop(cell: Cell, a: np.ndarray, b: np.ndarray, fa: float, fb: float,
         max_step: float) -> float:
    """Penalised time of one emitted move: full bang-bang, ramps included."""
    return cell.segment_cost(a, b, max_step=max_step, fa=fa, fb=fb, stops=True)


def polish(cell: Cell, path: list[np.ndarray], *, time_budget: float = 5.0,
           max_step: float = 0.05, rng: np.random.Generator | None = None,
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

    rng = rng or np.random.default_rng(1)
    pts = [np.asarray(p, dtype=float).copy() for p in path]
    fac = [cell.penalty_factor(p) for p in pts]
    costs = [_hop(cell, pts[i], pts[i + 1], fac[i], fac[i + 1], max_step)
             for i in range(len(pts) - 1)]
    before = sum(costs)
    deadline = time.time() + time_budget
    dropped = moved = tried = 0

    def drop_sweep() -> None:
        nonlocal dropped
        k = 1
        while k < len(pts) - 1 and time.time() < deadline:
            if cell.segment_collides(pts[k - 1], pts[k + 1], max_step=max_step):
                k += 1
                continue
            direct = _hop(cell, pts[k - 1], pts[k + 1], fac[k - 1], fac[k + 1], max_step)
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
        # candidates before paying for a collision check or a clearance query.
        budget = costs[k - 1] + costs[k]
        if (cell.move_time(pts[k - 1], candidate)
                + cell.move_time(candidate, pts[k + 1])) >= budget - 1e-9:
            continue
        if cell.segment_collides(pts[k - 1], candidate, max_step=max_step):
            continue
        if cell.segment_collides(candidate, pts[k + 1], max_step=max_step):
            continue
        f = cell.penalty_factor(candidate)
        first = _hop(cell, pts[k - 1], candidate, fac[k - 1], f, max_step)
        second = _hop(cell, candidate, pts[k + 1], f, fac[k + 1], max_step)
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


def _refine(cell: Cell, path: list[np.ndarray], *, shortcut_seconds: float,
            polish_seconds: float, check_step: float, log) -> list[np.ndarray]:
    """The whole post-processing chain, in the order the three passes need to run.

    Shortcutting reshapes the route while it is still dense, reduction picks which of those
    points are actually worth stopping at, and polishing then judges those stops under the
    time they really cost.
    """
    improved = shortcut(cell, path, time_budget=shortcut_seconds,
                        max_step=check_step, log=log)
    reduced = simplify(cell, improved, max_step=check_step)
    return polish(cell, reduced, time_budget=polish_seconds,
                  max_step=check_step, log=log)


def _resample(cell: Cell, a: np.ndarray, b: np.ndarray, step: float) -> list[np.ndarray]:
    """Points strictly between a and b, spaced no further apart than ``step``."""
    n = max(1, int(np.ceil(float(np.max(np.abs(b - a))) / max(step, 1e-9))))
    return [a + (b - a) * (k / n) for k in range(1, n)]


def shortcut(cell: Cell, path: list[np.ndarray], *, time_budget: float = 2.0,
             max_step: float = 0.05, rng: np.random.Generator | None = None,
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

    dense: list[np.ndarray] = [np.asarray(path[0], dtype=float)]
    for a, b in zip(path, path[1:]):
        dense.extend(_resample(cell, np.asarray(a, dtype=float),
                               np.asarray(b, dtype=float), max_step))
        dense.append(np.asarray(b, dtype=float))

    # One clearance query per waypoint, cached: the geometry query dominates, so scoring a
    # candidate has to be arithmetic over remembered factors rather than fresh queries.
    penalised = cell.penalty is not None and cell.penalty.enabled
    fac = [cell.penalty_factor(p) if penalised else 1.0 for p in dense]

    def step_cost(x: np.ndarray, y: np.ndarray, fx: float, fy: float) -> float:
        return cell.cruise_time(x, y) * max(fx, fy)

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
            cuts += _try_cut(cell, dense, fac, rng, max_step, penalised, span_cost)
        else:
            moves += _try_relocate(cell, dense, fac, rng, max_step, penalised, step_cost)

    if log:
        after = span_cost(0, len(dense) - 1)
        gain = 100.0 * (1.0 - after / before) if before > 0 else 0.0
        detail = "penalised cruise time" if penalised else "cruise time"
        log(f"      shortcut: {cuts} cuts and {moves} relocations kept from {tried} "
            f"attempts, {detail} {before:.2f} -> {after:.2f} s ({gain:.0f}% better)")
    return dense


def _try_cut(cell: Cell, dense, fac, rng, max_step, penalised, span_cost) -> int:
    """Replace dense[i..j] with the straight move between the ends, if that is cheaper."""
    i, j = sorted(rng.integers(0, len(dense), size=2))
    if j - i < 2:
        return 0
    span = span_cost(i, j)
    # Every factor is at least 1 and cruise time is additive, so the unpenalised cruise
    # time of the direct move is a valid lower bound on what the replacement can cost.
    # Rejecting on that first keeps the expensive checks off the many candidates that
    # were never going to win.
    if cell.cruise_time(dense[i], dense[j]) >= span - 1e-9:
        return 0
    if cell.segment_collides(dense[i], dense[j], max_step=max_step):
        return 0
    points = _resample(cell, dense[i], dense[j], max_step)
    if penalised:
        factors = [cell.penalty_factor(p) for p in points]
        chain = [dense[i]] + points + [dense[j]]
        chain_f = [fac[i]] + factors + [fac[j]]
        direct = sum(cell.cruise_time(x, y) * max(fx, fy)
                     for x, y, fx, fy in zip(chain, chain[1:], chain_f, chain_f[1:]))
        if direct >= span - 1e-9:
            return 0
    else:
        factors = [1.0] * len(points)
    dense[i + 1:j] = points
    fac[i + 1:j] = factors
    return 1


def _try_relocate(cell: Cell, dense, fac, rng, max_step, penalised, step_cost) -> int:
    """Displace one interior waypoint and keep the move if it lowers the local cost.

    The displacement is drawn in joint space but scaled by the cell's joint weights, so a
    given attempt moves the tool about as far whichever joints it uses -- otherwise almost
    every sample would be a wrist twiddle that changes nothing.
    """
    if len(dense) < 3:
        return 0
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
    if cell.segment_collides(dense[k - 1], candidate, max_step=max_step):
        return 0
    if cell.segment_collides(candidate, dense[k + 1], max_step=max_step):
        return 0

    f = cell.penalty_factor(candidate) if penalised else 1.0
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


def _plan_at_opening(cell: Cell, qa: np.ndarray, qb: np.ndarray, *, attempts: int,
                     runs: int, segment_length: float, check_step: float,
                     fallback_via: list[np.ndarray] | None,
                     shortcut_seconds: float, polish_seconds: float,
                     planning_time: float = DEFAULT_PLANNING_TIME, log=print,
                     record: list | None = None) -> list[np.ndarray]:
    """Collision-free joint path from ``qa`` to ``qb`` at the gun's current opening.

    If the direct transit cannot be found, the move is retried in two legs through each
    of ``fallback_via`` -- normally the cell's start pose.  Routing a difficult transit
    through a known-clear pose is what a robot programmer would do by hand, and it turns
    a long detour around the panels into two easy problems.
    """
    try:
        return _plan_direct(cell, qa, qb, attempts=attempts, runs=runs,
                            segment_length=segment_length, check_step=check_step,
                            shortcut_seconds=shortcut_seconds,
                            polish_seconds=polish_seconds,
                            planning_time=planning_time, log=log, record=record)
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
            first = _plan_direct(cell, qa, mid, attempts=attempts, runs=runs,
                                 segment_length=segment_length, check_step=check_step,
                                 shortcut_seconds=0.0, polish_seconds=0.0,
                                 planning_time=planning_time, log=log, record=halves)
            second = _plan_direct(cell, mid, qb, attempts=attempts, runs=runs,
                                  segment_length=segment_length, check_step=check_step,
                                  shortcut_seconds=0.0, polish_seconds=0.0,
                                  planning_time=planning_time, log=log, record=halves)
        except PlanningError:
            continue
        if len(halves) == 2:
            _capture(record, halves[0] + halves[1][1:])
        # Refine the joined route rather than each leg: the detour through the fallback
        # pose is exactly the kind of corner these passes exist to cut.
        return _refine(cell, first + second[1:], shortcut_seconds=shortcut_seconds,
                       polish_seconds=polish_seconds, check_step=check_step, log=log)
    raise PlanningError("freespace transit failed, including via fallback poses")


def plan_freespace(cell: Cell, qa: np.ndarray, qb: np.ndarray, *, attempts: int = 3,
                   runs: int = 1, segment_length: float = 0.02,
                   check_step: float = 0.05,
                   fallback_via: list[np.ndarray] | None = None,
                   shortcut_seconds: float = 2.0, polish_seconds: float = 5.0,
                   planning_time: float = DEFAULT_PLANNING_TIME,
                   openings: list[float] | None = None,
                   record: list | None = None,
                   log=print) -> list[tuple[list[np.ndarray], float | None]]:
    """Plan a transit, choosing a gun opening for it when the natural one will not do.

    Some destinations simply cannot be reached at the opening the robot arrives with: the
    tip is 200 mm of swing, so an opening that clears a fixture on the way out fouls it on
    the way back.  ``openings`` lists the openings to consider, most preferred first --
    normally the opening carried over from the previous locator, then closed, then wide.

    Returns one ``(path, opening)`` leg per output phase.  A single leg is always
    preferred, and a two-leg answer is only produced when no single opening works: the
    gun then changes at the intermediate pose, where the robot is stationary and the
    change costs no motion.

    ``record``, if given, is extended with each leg's route as the sampling planner
    returned it, one entry per returned leg and in the same order.  Failed attempts leave
    nothing behind: only the openings that were actually used contribute.
    """
    def attempt(opening, a, b, budget, into=None):
        with cell.gun_opening(opening):
            return _plan_at_opening(cell, a, b, attempts=attempts, runs=runs,
                                    segment_length=segment_length,
                                    check_step=check_step, fallback_via=fallback_via,
                                    shortcut_seconds=budget,
                                    polish_seconds=polish_seconds if budget else 0.0,
                                    planning_time=planning_time, log=log, record=into)

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
                    first = _refine(cell, first, shortcut_seconds=shortcut_seconds,
                                    polish_seconds=polish_seconds,
                                    check_step=check_step, log=log)
                with cell.gun_opening(second_open):
                    second = _refine(cell, second, shortcut_seconds=shortcut_seconds,
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


def _plan_direct(cell: Cell, qa: np.ndarray, qb: np.ndarray, *, attempts: int,
                 runs: int, segment_length: float, check_step: float,
                 shortcut_seconds: float, polish_seconds: float,
                 planning_time: float = DEFAULT_PLANNING_TIME, log=print,
                 record: list | None = None) -> list[np.ndarray]:
    if not cell.segment_collides(qa, qb, max_step=check_step):
        # A clear straight line is normally the best answer there is, and a sampling
        # planner asked to improve on it would only return it again.  But "clear" and
        # "sensible" part company when the line grazes a panel, so when the penalty says
        # this one does, the same pass that stands other routes off is given a chance to
        # bow it away -- there is nothing for OMPL to do here, but plenty for relocation.
        raw = cell.move_time(qa, qb)
        if not (cell.penalty is not None and cell.penalty.enabled) or shortcut_seconds <= 0:
            return _capture(record, [qa, qb])
        cost = cell.segment_cost(qa, qb, max_step=check_step, stops=True)
        if cost <= raw * 1.05:
            return _capture(record, [qa, qb])
        log(f"      direct move is clear but runs close to the parts "
            f"(cost {cost:.2f} s against {raw:.2f} s unpenalised); standing it off")
        # Densified here rather than left to shortcut: a two-point path has no interior
        # waypoint, and relocation is the only move that can help.  The unrefined route is
        # still the bare straight line; the densification is part of the refinement.
        _capture(record, [qa, qb])
        seeded = [qa] + _resample(cell, np.asarray(qa, dtype=float),
                                  np.asarray(qb, dtype=float), check_step) + [qb]
        return _refine(cell, seeded, shortcut_seconds=shortcut_seconds,
                       polish_seconds=polish_seconds, check_step=check_step, log=log)

    # RRTConnect returns the first path it finds, and which homotopy class that lands in
    # is luck -- one run goes over the fixture, the next threads behind it.  The
    # optimisation pass afterwards can shorten a route and stand it off, but it cannot move
    # it to the other side of an obstacle, so whichever class arrives here is the one that
    # ships.  Sampling several solutions and keeping the best-scoring one is therefore the
    # only stage that can make that choice at all.
    candidates: list[tuple[float, float, list[np.ndarray]]] = []
    last = ""
    for attempt in range(1, max(runs, attempts) + 1):
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
        if response.successful:
            raw = _extract(response.results)
            cost, plain = _path_cost(cell, raw, check_step)
            candidates.append((cost, plain, raw))
            log(f"      OMPL run {attempt}: solved in {dt:.1f}s ({len(raw)} raw points, "
                f"cost {cost:.2f} s against {plain:.2f} s unpenalised)")
        else:
            last = str(response.message)
            log(f"      OMPL run {attempt}: {last} ({dt:.1f}s)")
        # The minimum is a floor, not a cap: keep going past it only while nothing at all
        # has solved, up to the attempts limit.
        if attempt >= runs and candidates:
            break

    if not candidates:
        raise PlanningError(
            f"freespace transit failed after {max(runs, attempts)} attempts: {last}")

    cost, plain, raw = min(candidates, key=lambda c: c[0])
    _capture(record, raw)
    if len(candidates) > 1:
        worst = max(c[0] for c in candidates)
        log(f"      keeping the best of {len(candidates)} solutions: cost {cost:.2f} s "
            f"against {worst:.2f} s for the worst")
    # Shortcut before reducing: the dense path gives the cuts somewhere to land.
    path = _refine(cell, raw, shortcut_seconds=shortcut_seconds,
                   polish_seconds=polish_seconds, check_step=check_step, log=log)
    log(f"      reduced to {len(path)} points")
    return path


def validate(cell: Cell, path: list[np.ndarray], max_step: float = 0.05) -> str | None:
    """Re-check a finished path; returns a description of the first bad move, or None."""
    for k, (a, b) in enumerate(zip(path, path[1:])):
        if cell.segment_collides(a, b, max_step=max_step):
            return f"move {k} -> {k + 1} is not collision free"
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
