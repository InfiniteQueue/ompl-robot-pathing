"""Serialisation of planned motion to the output waypoint file.

The schema is fixed by the consumer (``TesseractWaypoints.vb``), and two details of it are
easy to get wrong:

* ``joints`` is a name -> value map in the *planner's* units -- radians for revolute
  joints, millimetres for prismatic -- even though ``units`` reads ``"mm/deg"``.
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
from .profile import MotionProfile
from .toolpath import LIN, Segment

OUTPUT_NAME = "waypoints.json"

# Nominal peak speeds behind the per-waypoint `time` field. They are commanded maxima, not
# averages: with a trapezoidal profile the ramps mean the mean speed is lower, so a move
# takes correspondingly longer.
DEFAULT_JOINT_SPEED_DEG_S = 180.0
DEFAULT_LINEAR_SPEED_MM_S = 250.0


class Timing:
    """Schedules each phase's waypoints along a trapezoidal velocity profile.

    The profile spans a whole phase, starting and finishing at rest, which is what the
    schema's per-phase ``time`` origin implies: each phase is its own motion block.
    Waypoint positions are untouched -- only the times they are reached change.

    Distance along a phase is measured in the metric that governs the move: for ``LIN``
    that is tool travel in millimetres, and for ``PTP`` it is the largest joint excursion
    of each step, so every step is timed by whichever joint has furthest to go.
    """

    def __init__(self, joint_speed_deg_s: float = DEFAULT_JOINT_SPEED_DEG_S,
                 linear_speed_mm_s: float = DEFAULT_LINEAR_SPEED_MM_S,
                 profile: MotionProfile | None = None):
        self.joint_speed = np.deg2rad(max(joint_speed_deg_s, 1e-6))
        self.linear_speed = max(linear_speed_mm_s, 1e-6)
        self.profile = profile or MotionProfile()

    def phase_times(self, motion: str, states: list[np.ndarray],
                    positions: list[np.ndarray]) -> list[float]:
        if len(states) < 2:
            return [0.0] * len(states)
        if motion == LIN:
            steps = [float(np.linalg.norm(positions[i + 1] - positions[i]))
                     for i in range(len(positions) - 1)]
            speed = self.linear_speed
        else:
            steps = [float(np.max(np.abs(states[i + 1] - states[i])))
                     for i in range(len(states) - 1)]
            speed = self.joint_speed
        return self.profile.schedule(steps, speed)


def _pose_rows(cell: Cell, q: np.ndarray) -> list[list[float]]:
    """World TCP pose as 4x4 row-major, translation in manifest units (mm)."""
    T = cell.fk(q).copy()
    T[:3, 3] /= cell.man.scale
    return [[round(float(v), 6) for v in row] for row in T]


def build_document(cell: Cell, man: Manifest, segments: list[Segment],
                   timing: Timing | None = None) -> dict:
    timing = timing or Timing()
    out_segments = []

    for seg in segments:
        phases = []
        for ph in seg.phases:
            poses = [_pose_rows(cell, q) for q in ph.states]
            positions = [np.array([r[0][3], r[1][3], r[2][3]]) for r in poses]
            times = timing.phase_times(ph.motion, ph.states, positions)
            waypoints = [
                {
                    "joints": {name: round(float(value), 9)
                               for name, value in zip(cell.joint_names, q)},
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


def write(directory: str, document: dict) -> str:
    path = os.path.join(directory, OUTPUT_NAME)
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
