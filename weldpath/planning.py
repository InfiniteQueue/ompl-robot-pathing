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

    Where a clearance penalty is in force this also refuses reductions that push the route
    nearer the parts.  Without that check this pass runs last and would quietly undo the
    standoff the shortcut pass just bought: the chord across a corner the robot took wide
    is shorter, collision free, and hard against the panel.
    """
    if len(path) < 3:
        return [p.copy() for p in path]
    penalised = cell.penalty is not None and cell.penalty.enabled
    factors: dict[int, float] = {}

    def factor(k: int) -> float:
        if k not in factors:
            factors[k] = cell.penalty_factor(path[k])
        return factors[k]

    def polyline_cost(i: int, j: int) -> float:
        return sum(cell.distance(path[k], path[k + 1]) * max(factor(k), factor(k + 1))
                   for k in range(i, j))

    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if cell.segment_collides(path[i], path[j], max_step=max_step):
                j -= 1
                continue
            if penalised:
                chord = cell.segment_cost(path[i], path[j], max_step=max_step,
                                          fa=factor(i), fb=factor(j))
                if chord > polyline_cost(i, j) + 1e-9:
                    j -= 1
                    continue
            break
        out.append(path[j])
        i = j
    return out


def _resample(cell: Cell, a: np.ndarray, b: np.ndarray, step: float) -> list[np.ndarray]:
    """Points strictly between a and b, spaced no further apart than ``step``."""
    n = max(1, int(np.ceil(float(np.max(np.abs(b - a))) / max(step, 1e-9))))
    return [a + (b - a) * (k / n) for k in range(1, n)]


def shortcut(cell: Cell, path: list[np.ndarray], *, time_budget: float = 2.0,
             max_step: float = 0.05, rng: np.random.Generator | None = None,
             log=None) -> list[np.ndarray]:
    """Improve a path under the weighted metric, penalised for running close to the parts.

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

    Cost counts how far the tool actually travels, so a wide J1 excursion is cut before a
    wrist rotation that costs the same in raw joint space, and time spent near a panel is
    multiplied by :class:`~weldpath.penalty.ClearancePenalty`.

    The path is densified first, so changes can land between the planner's own waypoints
    instead of only at them.  Work is bounded by ``time_budget`` seconds; the result is
    always collision free, since every replacement is checked before it is kept.
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
        return cell.distance(x, y) * max(fx, fy)

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
        detail = "penalised cost" if penalised else "weighted path cost"
        log(f"      shortcut: {cuts} cuts and {moves} relocations kept from {tried} "
            f"attempts, {detail} {before:.2f} -> {after:.2f} m ({gain:.0f}% better)")
    return dense


def _try_cut(cell: Cell, dense, fac, rng, max_step, penalised, span_cost) -> int:
    """Replace dense[i..j] with the straight move between the ends, if that is cheaper."""
    i, j = sorted(rng.integers(0, len(dense), size=2))
    if j - i < 2:
        return 0
    span = span_cost(i, j)
    # Every factor is at least 1, so the unpenalised length is a valid lower bound on what
    # the replacement can cost.  Rejecting on that first keeps the expensive checks off
    # the many candidates that were never going to win.
    if cell.distance(dense[i], dense[j]) >= span - 1e-9:
        return 0
    if cell.segment_collides(dense[i], dense[j], max_step=max_step):
        return 0
    points = _resample(cell, dense[i], dense[j], max_step)
    if penalised:
        factors = [cell.penalty_factor(p) for p in points]
        chain = [dense[i]] + points + [dense[j]]
        chain_f = [fac[i]] + factors + [fac[j]]
        direct = sum(cell.distance(x, y) * max(fx, fy)
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
    floor = (cell.distance(dense[k - 1], candidate)
             + cell.distance(candidate, dense[k + 1]))
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
    """Penalised length of a whole path, and its plain weighted travel.

    With no penalty in force the two are equal, so ranking on the first still ranks on
    length and the choice degrades to "shortest raw solution" rather than to nothing.
    Endpoint factors are threaded from one segment to the next so each waypoint costs one
    clearance query rather than two.
    """
    if len(path) < 2:
        return 0.0, 0.0
    penalised = cell.penalty is not None and cell.penalty.enabled
    total = travel = 0.0
    prev_f = cell.penalty_factor(path[0]) if penalised else None
    for a, b in zip(path, path[1:]):
        next_f = cell.penalty_factor(b) if penalised else None
        total += cell.segment_cost(a, b, max_step=max_step, fa=prev_f, fb=next_f)
        travel += cell.distance(a, b)
        prev_f = next_f
    return total, travel


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


def _ompl_profile(segment_length: float) -> OMPLRealVectorMoveProfile:
    profile = OMPLRealVectorMoveProfile()
    profile.collision_check_config.longest_valid_segment_length = segment_length
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
                     shortcut_seconds: float, log,
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
                            shortcut_seconds=shortcut_seconds, log=log, record=record)
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
                                 shortcut_seconds=0.0, log=log, record=halves)
            second = _plan_direct(cell, mid, qb, attempts=attempts, runs=runs,
                                  segment_length=segment_length, check_step=check_step,
                                  shortcut_seconds=0.0, log=log, record=halves)
        except PlanningError:
            continue
        if len(halves) == 2:
            _capture(record, halves[0] + halves[1][1:])
        # Shortcut the joined route rather than each leg: the detour through the fallback
        # pose is exactly the kind of corner this pass exists to cut.
        joined = shortcut(cell, first + second[1:], time_budget=shortcut_seconds,
                          max_step=check_step, log=log)
        return simplify(cell, joined, max_step=check_step)
    raise PlanningError("freespace transit failed, including via fallback poses")


def plan_freespace(cell: Cell, qa: np.ndarray, qb: np.ndarray, *, attempts: int = 3,
                   runs: int = 1, segment_length: float = 0.02,
                   check_step: float = 0.05,
                   fallback_via: list[np.ndarray] | None = None,
                   shortcut_seconds: float = 2.0,
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
                                    shortcut_seconds=budget, log=log, record=into)

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
                    first = simplify(cell, shortcut(cell, first,
                                                    time_budget=shortcut_seconds,
                                                    max_step=check_step, log=log),
                                     max_step=check_step)
                with cell.gun_opening(second_open):
                    second = simplify(cell, shortcut(cell, second,
                                                     time_budget=shortcut_seconds,
                                                     max_step=check_step, log=log),
                                      max_step=check_step)
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
                 shortcut_seconds: float, log,
                 record: list | None = None) -> list[np.ndarray]:
    if not cell.segment_collides(qa, qb, max_step=check_step):
        # A clear straight line is normally the best answer there is, and a sampling
        # planner asked to improve on it would only return it again.  But "clear" and
        # "sensible" part company when the line grazes a panel, so when the penalty says
        # this one does, the same pass that stands other routes off is given a chance to
        # bow it away -- there is nothing for OMPL to do here, but plenty for relocation.
        raw = cell.distance(qa, qb)
        if not (cell.penalty is not None and cell.penalty.enabled) or shortcut_seconds <= 0:
            return _capture(record, [qa, qb])
        cost = cell.segment_cost(qa, qb, max_step=check_step)
        if cost <= raw * 1.05:
            return _capture(record, [qa, qb])
        log(f"      direct move is clear but runs close to the parts "
            f"(cost {cost:.2f} m against {raw:.2f} m of travel); standing it off")
        # Densified here rather than left to shortcut: a two-point path has no interior
        # waypoint, and relocation is the only move that can help.  The unrefined route is
        # still the bare straight line; the densification is part of the refinement.
        _capture(record, [qa, qb])
        seeded = [qa] + _resample(cell, np.asarray(qa, dtype=float),
                                  np.asarray(qb, dtype=float), check_step) + [qb]
        improved = shortcut(cell, seeded, time_budget=shortcut_seconds,
                            max_step=check_step, log=log)
        return simplify(cell, improved, max_step=check_step)

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
        profiles.addProfile(OMPL_NAMESPACE, "DEFAULT", _ompl_profile(segment_length))
        request = PlannerRequest()
        request.env = cell.env
        request.instructions = _make_program(cell, qa, qb)
        request.profiles = profiles
        t0 = time.time()
        response = OMPLMotionPlanner(OMPL_NAMESPACE).solve(request)
        dt = time.time() - t0
        if response.successful:
            raw = _extract(response.results)
            cost, travel = _path_cost(cell, raw, check_step)
            candidates.append((cost, travel, raw))
            log(f"      OMPL run {attempt}: solved in {dt:.1f}s ({len(raw)} raw points, "
                f"cost {cost:.2f} m over {travel:.2f} m of travel)")
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

    cost, travel, raw = min(candidates, key=lambda c: c[0])
    _capture(record, raw)
    if len(candidates) > 1:
        worst = max(c[0] for c in candidates)
        log(f"      keeping the best of {len(candidates)} solutions: cost {cost:.2f} m "
            f"against {worst:.2f} m for the worst")
    # Shortcut before reducing: the dense path gives the cuts somewhere to land.
    cut = shortcut(cell, raw, time_budget=shortcut_seconds,
                   max_step=check_step, log=log)
    path = simplify(cell, cut, max_step=check_step)
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
