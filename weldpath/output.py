"""Serialisation of planned motion to the output waypoint file.

The schema is fixed by the consumer (``TesseractWaypoints.vb``), and two details of it are
easy to get wrong:

* ``joints`` is a name -> value map in the *planner's* angular units -- radians for
  revolute joints, millimetres for prismatic -- even though ``units`` reads ``"mm/deg"``.
  The values are the robot's *register* values rather than the planner's, which differ for
  joint 3, whose linkage holds it against the floor.  See
  :func:`weldpath.profile.commanded`.
* ``tcp_world_mm`` is the full 4x4 row-major world pose with the translation in
  millimetres in the fourth column, in the same frame as the manifest's
  ``locators[].pose_world``, so the consumer can place it back in the cell untransformed.

``gun_opening_mm`` and ``contact_allowed`` belong to the phase, not the waypoint: a phase
is a run of waypoints sharing one motion type and one gun state.  Only members the schema
declares are written.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .cell import Cell
from .manifest import Manifest
from .profile import JointDynamics, commanded
from .toolpath import LIN, Segment

OUTPUT_NAME = "waypoints.json"
UNREFINED_NAME = "waypoints-unrefined.json"

# Cartesian speed cap on LIN moves. The joint limits in weldpath.profile govern every move;
# this is an additional commanded ceiling on tool speed, not a second dynamics model.
DEFAULT_LINEAR_SPEED_MM_S = 250.0


class Timing:
    """Schedules waypoints under the robot's real per-joint velocity and acceleration.

    Every waypoint is a full stop, so each move is its own bang-bang profile and the times
    simply accumulate -- unlike the old model, which spread one profile across a whole phase
    and therefore made a move's duration exactly linear in its distance. That linearity is
    why splitting a move used to cost nothing; it now costs a real ramp, which is what makes
    surplus waypoints expensive.

    A ``LIN`` move is additionally capped by the commanded tool speed. No Cartesian
    acceleration is modelled, because the manifest supplies none -- the cap is a floor on
    the move's duration, and the joint profile still governs whenever it is slower.
    """

    def __init__(self, dynamics: JointDynamics,
                 linear_speed_mm_s: float = DEFAULT_LINEAR_SPEED_MM_S):
        self.dynamics = dynamics
        self.linear_speed = max(linear_speed_mm_s, 1e-6)

    def phase_times(self, motion: str, states: list[np.ndarray],
                    positions: list[np.ndarray]) -> list[float]:
        if len(states) < 2:
            return [0.0] * len(states)
        times = [0.0]
        for i in range(len(states) - 1):
            dt = self.dynamics.move_time(states[i], states[i + 1])
            if motion == LIN:
                travel = float(np.linalg.norm(positions[i + 1] - positions[i]))
                dt = max(dt, travel / self.linear_speed)
            times.append(times[-1] + dt)
        return times


def _joint_scales(cell: Cell, man: Manifest) -> np.ndarray:
    """Factor turning each joint's planner value into the unit the output is written in.

    Revolute joints are written in radians, as planned, which the consumer expects despite
    ``units`` reading ``"mm/deg"``.  Prismatic ones are planned in metres and written in the
    manifest's own length unit, so those do get converted.
    """
    prismatic = {j.name: j.is_prismatic for d in man.devices for j in d.joints}
    return np.array([1.0 / man.scale if prismatic.get(name) else 1.0
                     for name in cell.joint_names], dtype=float)


def _pose_rows(cell: Cell, q: np.ndarray) -> list[list[float]]:
    """World TCP pose as 4x4 row-major, translation in manifest units (mm)."""
    T = cell.fk(q).copy()
    T[:3, 3] /= cell.man.scale
    return [[round(float(v), 6) for v in row] for row in T]


def build_document(cell: Cell, man: Manifest, segments: list[Segment],
                   timing: Timing, unrefined: bool = False) -> dict:
    """Serialise the planned motion.

    ``unrefined`` writes each segment's raw phases instead -- the sampling planner's own
    output, before shortcutting and waypoint reduction -- in the same schema, so the two
    files can be diffed or replayed against each other.
    """
    out_segments = []
    scales = _joint_scales(cell, man)

    for seg in segments:
        phases = []
        for ph in (seg.raw_phases if unrefined else seg.phases):
            poses = [_pose_rows(cell, q) for q in ph.states]
            positions = [np.array([r[0][3], r[1][3], r[2][3]]) for r in poses]
            times = timing.phase_times(ph.motion, ph.states, positions)
            waypoints = [
                {
                    "joints": {name: round(float(value), 9)
                               for name, value
                               in zip(cell.joint_names, commanded(q) * scales)},
                    "tcp_world_mm": rows,
                    "time": round(t, 6),
                }
                for q, rows, t in zip(ph.states, poses, times)
            ]
            phases.append({
                "motion": ph.motion,
                "contact_allowed": bool(ph.contact_allowed),
                "gun_opening_mm": round(float(ph.gun_opening_mm), 6),
                "waypoints": waypoints,
            })
        entry = {"from": seg.source, "to": seg.target, "phases": phases}
        if seg.error:
            entry["error"] = seg.error
        out_segments.append(entry)

    return {
        "units": "mm/deg",
        "joint_names": list(cell.joint_names),
        "segments": out_segments,
    }


def check_endpoints(document: dict, man: Manifest, tol_mm: float = 1.0) -> list[str]:
    """Verify each planned segment runs between the two locators it names.

    The consumer relies on the end of a segment being its ``to`` locator, and the path is
    required to begin at the first locator rather than at the robot's start pose, so both
    ends are worth asserting rather than assuming.

    Welds are checked against their *imported* pose, which is what the file claims and what
    the consumer will place back in the cell -- not the shifted pose used for planning.
    """
    poses = {loc.name: np.array(loc.export_pose, dtype=float) for loc in man.locators}
    problems = []

    def position(waypoint) -> np.ndarray:
        rows = waypoint["tcp_world_mm"]
        return np.array([rows[0][3], rows[1][3], rows[2][3]])

    for seg in document["segments"]:
        if seg.get("error") or not seg["phases"]:
            continue
        ends = (("starts", seg["from"], position(seg["phases"][0]["waypoints"][0])),
                ("ends", seg["to"], position(seg["phases"][-1]["waypoints"][-1])))
        for verb, locator, got in ends:
            error = float(np.linalg.norm(got - poses[locator][:3, 3]))
            if error > tol_mm:
                problems.append(f"{seg['from']} -> {seg['to']} {verb} {error:.2f} mm "
                                f"from locator '{locator}'")
    return problems


def write(directory: str, document: dict, name: str = OUTPUT_NAME) -> str:
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(document, fh, indent=1)
    return path


def summarise(document: dict, segments: list[Segment]) -> str:
    lines = []
    total = 0
    ok = 0
    for seg, planned in zip(document["segments"], segments):
        count = sum(len(p["waypoints"]) for p in seg["phases"])
        total += count
        if seg.get("error"):
            lines.append(f"  {seg['from']} -> {seg['to']}: FAILED ({seg['error']})")
            continue
        ok += 1
        detail = ", ".join(
            f"{phase.kind}/{written['motion']}:{len(written['waypoints'])}"
            for phase, written in zip(planned.phases, seg["phases"]))
        lines.append(f"  {seg['from']} -> {seg['to']}: {count} waypoints [{detail}]")
    head = (f"{ok}/{len(document['segments'])} segments planned, "
            f"{total} waypoints total")
    return "\n".join([head] + lines)
