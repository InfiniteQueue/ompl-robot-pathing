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


def plan_freespace(cell: Cell, qa: np.ndarray, qb: np.ndarray, *, attempts: int = 3,
                   segment_length: float = 0.02, check_step: float = 0.05,
                   fallback_via: list[np.ndarray] | None = None,
                   log=print) -> list[np.ndarray]:
    """Collision-free joint path from ``qa`` to ``qb``.

    If the direct transit cannot be found, the move is retried in two legs through each
    of ``fallback_via`` -- normally the cell's start pose.  Routing a difficult transit
    through a known-clear pose is what a robot programmer would do by hand, and it turns
    a long detour around the panels into two easy problems.
    """
    try:
        return _plan_direct(cell, qa, qb, attempts=attempts,
                            segment_length=segment_length, check_step=check_step, log=log)
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
                                 log=log)
            second = _plan_direct(cell, mid, qb, attempts=attempts,
                                  segment_length=segment_length, check_step=check_step,
                                  log=log)
        except PlanningError:
            continue
        return simplify(cell, first + second[1:], max_step=check_step)
    raise PlanningError("freespace transit failed, including via fallback poses")


def _plan_direct(cell: Cell, qa: np.ndarray, qb: np.ndarray, *, attempts: int,
                 segment_length: float, check_step: float, log) -> list[np.ndarray]:
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
            path = simplify(cell, raw, max_step=check_step)
            log(f"      OMPL attempt {attempt}: solved in {dt:.1f}s "
                f"({len(raw)} raw -> {len(path)} points)")
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
