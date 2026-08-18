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
    """
    if len(path) < 3:
        return [p.copy() for p in path]
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and cell.segment_collides(path[i], path[j], max_step=max_step):
            j -= 1
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
    """Shorten a path by replacing detours with direct moves, under the weighted metric.

    ``simplify`` can only delete waypoints that a straight move already bypasses, so it
    never changes the route: a sampled path that swings the arm around the base to reach a
    point beside it keeps that swing.  This pass repeatedly picks two states on the path
    and, if the straight move between them is collision free *and* cheaper under the
    cell's weighted joint metric, splices it in.  Because the metric counts how far the
    tool actually travels, a wide J1 excursion is what gets cut first rather than a wrist
    rotation that costs the same in raw joint space.

    The path is densified first, so cuts can start and end between the planner's own
    waypoints instead of only at them.  Work is bounded by ``time_budget`` seconds; the
    result is always collision free, since every replacement is checked before it is kept.
    """
    if len(path) < 3 or time_budget <= 0:
        return [np.asarray(p, dtype=float).copy() for p in path]

    dense: list[np.ndarray] = [np.asarray(path[0], dtype=float)]
    for a, b in zip(path, path[1:]):
        dense.extend(_resample(cell, np.asarray(a, dtype=float),
                               np.asarray(b, dtype=float), max_step))
        dense.append(np.asarray(b, dtype=float))

    def cost(points: list[np.ndarray]) -> float:
        return sum(cell.distance(x, y) for x, y in zip(points, points[1:]))

    rng = rng or np.random.default_rng(0)
    before = cost(dense)
    deadline = time.time() + time_budget
    tried = kept = 0

    while time.time() < deadline and len(dense) > 2:
        i, j = sorted(rng.integers(0, len(dense), size=2))
        if j - i < 2:
            continue
        tried += 1
        span = cost(dense[i:j + 1])
        direct = cell.distance(dense[i], dense[j])
        if direct >= span - 1e-9:
            continue
        if cell.segment_collides(dense[i], dense[j], max_step=max_step):
            continue
        dense[i + 1:j] = _resample(cell, dense[i], dense[j], max_step)
        kept += 1

    if log:
        after = cost(dense)
        gain = 100.0 * (1.0 - after / before) if before > 0 else 0.0
        log(f"      shortcut: {kept}/{tried} cuts kept, weighted path cost "
            f"{before:.2f} -> {after:.2f} m ({gain:.0f}% shorter)")
    return dense


# ---------------------------------------------------------------------------
# freespace
# ---------------------------------------------------------------------------
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


def _plan_at_opening(cell: Cell, qa: np.ndarray, qb: np.ndarray, *, attempts: int,
                     segment_length: float, check_step: float,
                     fallback_via: list[np.ndarray] | None,
                     shortcut_seconds: float, log) -> list[np.ndarray]:
    """Collision-free joint path from ``qa`` to ``qb`` at the gun's current opening.

    If the direct transit cannot be found, the move is retried in two legs through each
    of ``fallback_via`` -- normally the cell's start pose.  Routing a difficult transit
    through a known-clear pose is what a robot programmer would do by hand, and it turns
    a long detour around the panels into two easy problems.
    """
    try:
        return _plan_direct(cell, qa, qb, attempts=attempts,
                            segment_length=segment_length, check_step=check_step,
                            shortcut_seconds=shortcut_seconds, log=log)
    except PlanningError:
        if not fallback_via:
            raise

    for i, mid in enumerate(fallback_via):
        if cell.in_collision(mid):
            continue
        log(f"      retrying via fallback pose {i + 1}")
        try:
            first = _plan_direct(cell, qa, mid, attempts=attempts,
                                 segment_length=segment_length, check_step=check_step,
                                 shortcut_seconds=0.0, log=log)
            second = _plan_direct(cell, mid, qb, attempts=attempts,
                                  segment_length=segment_length, check_step=check_step,
                                  shortcut_seconds=0.0, log=log)
        except PlanningError:
            continue
        # Shortcut the joined route rather than each leg: the detour through the fallback
        # pose is exactly the kind of corner this pass exists to cut.
        joined = shortcut(cell, first + second[1:], time_budget=shortcut_seconds,
                          max_step=check_step, log=log)
        return simplify(cell, joined, max_step=check_step)
    raise PlanningError("freespace transit failed, including via fallback poses")


def plan_freespace(cell: Cell, qa: np.ndarray, qb: np.ndarray, *, attempts: int = 3,
                   segment_length: float = 0.02, check_step: float = 0.05,
                   fallback_via: list[np.ndarray] | None = None,
                   shortcut_seconds: float = 2.0,
                   openings: list[float] | None = None,
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
    """
    def attempt(opening, a, b, budget):
        with cell.gun_opening(opening):
            return _plan_at_opening(cell, a, b, attempts=attempts,
                                    segment_length=segment_length,
                                    check_step=check_step, fallback_via=fallback_via,
                                    shortcut_seconds=budget, log=log)

    candidates = _opening_candidates(cell, openings)

    # One opening for the whole transit, in preference order: changing the gun is a real
    # operation on the machine, so it is a last resort rather than a free parameter.
    last = None
    for n, opening in enumerate(candidates):
        try:
            if n:
                log(f"      retrying with the gun at {opening:g} mm")
            return [(attempt(opening, qa, qb, shortcut_seconds), opening)]
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
            try:
                first = attempt(first_open, qa, mid, 0.0)
            except PlanningError:
                continue
            for second_open in candidates:
                if second_open == first_open:
                    continue                    # already ruled out as a single opening
                try:
                    second = attempt(second_open, mid, qb, 0.0)
                except PlanningError:
                    continue
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
                 segment_length: float, check_step: float, shortcut_seconds: float,
                 log) -> list[np.ndarray]:
    if not cell.segment_collides(qa, qb, max_step=check_step):
        return [qa, qb]

    last = ""
    for attempt in range(1, attempts + 1):
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
            log(f"      OMPL attempt {attempt}: solved in {dt:.1f}s ({len(raw)} raw points)")
            # Shortcut before reducing: the dense path gives the cuts somewhere to land.
            cut = shortcut(cell, raw, time_budget=shortcut_seconds,
                           max_step=check_step, log=log)
            path = simplify(cell, cut, max_step=check_step)
            log(f"      reduced to {len(path)} points")
            return path
        last = str(response.message)
        log(f"      OMPL attempt {attempt}: {last} ({dt:.1f}s)")
    raise PlanningError(f"freespace transit failed after {attempts} attempts: {last}")


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
