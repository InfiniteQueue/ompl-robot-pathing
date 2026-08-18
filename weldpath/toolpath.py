"""Turn the manifest's ordered locators into planned, phased motion.

Locators are visited in manifest order and one output segment is produced per consecutive
pair, matching the sample output.

Weld locators get a linear approach and depart of ``planning.linear_zone_mm``, so the gun
slides onto and off the joint in a straight line instead of arriving on a curve.  Per the
brief the gun is treated as stationary through the weld: the robot stops at the locator,
the opening changes from ``gun_opening_arrive`` to ``gun_opening_leave``, and no motion is
planned for the closing itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cell import Cell
from .manifest import Locator, Manifest
from .planning import PlanningError, plan_freespace, plan_linear, validate

AXES = {"+x": np.array([1.0, 0, 0]), "-x": np.array([-1.0, 0, 0]),
        "+y": np.array([0, 1.0, 0]), "-y": np.array([0, -1.0, 0]),
        "+z": np.array([0, 0, 1.0]), "-z": np.array([0, 0, -1.0])}


PTP = "PTP"
LIN = "LIN"


@dataclass
class Phase:
    """A run of waypoints sharing one motion type and one gun state.

    ``kind`` is internal bookkeeping for the run log; it is not serialised, because the
    consumer's schema has no member for it.
    """
    kind: str                       # freespace | approach | weld | depart
    motion: str                     # PTP | LIN
    states: list[np.ndarray]
    gun_opening_mm: float = 0.0
    # True where the gun is deliberately up against a panel -- the weld itself and the
    # linear moves on and off it. Freespace transits are not meant to touch anything.
    contact_allowed: bool = False


@dataclass
class Segment:
    source: str
    target: str
    phases: list[Phase] = field(default_factory=list)
    error: str | None = None


def retract_axis(override: str | None = None) -> np.ndarray:
    """Direction, in a weld locator's own frame, that the robot retracts along.

    The locator carries its own orientation, and its z axis is the one authored against the
    joint, so the retract is defined relative to that rather than derived from the tool.
    Deriving it from the gun's stroke -- as this did while the gun was prismatic -- made
    the direction a property of the machine instead of the weld, and gave no answer at all
    once the gun became angular.

    The default is ``-z``, matching ``--weld-shift-mm``: the shift backs the tool off the
    panel along the locator's -z, so the 300 mm linear retract has to travel the same way.
    """
    return AXES[override.lower()] if override else AXES["-z"]


def offset_pose(pose: np.ndarray, axis_tcp: np.ndarray, distance: float) -> np.ndarray:
    """Shift ``pose`` along a direction expressed in its own frame."""
    out = np.array(pose, dtype=float).copy()
    out[:3, 3] = out[:3, 3] + out[:3, :3] @ (axis_tcp * distance)
    return out


class ToolpathPlanner:
    def __init__(self, cell: Cell, man: Manifest, *, approach_axis: str | None = None,
                 linear_step_mm: float = 50.0, ompl_attempts: int = 3,
                 segment_length: float = 0.02, check_step_deg: float = 3.0,
                 shortcut_seconds: float = 2.0, weld_clearance_mm: float | None = None,
                 log=print):
        self.cell = cell
        self.man = man
        self.log = log
        self.axis = retract_axis(approach_axis)
        self.linear_step_mm = linear_step_mm
        self.ompl_attempts = ompl_attempts
        self.segment_length = segment_length
        self.check_step = np.deg2rad(check_step_deg)
        self.shortcut_seconds = shortcut_seconds
        # None means "no separate weld rule", i.e. the cell's own clearance throughout.
        self.weld_clearance = (None if weld_clearance_mm is None
                               else weld_clearance_mm * man.scale)
        self.start_q = np.array([man.start_state[n] for n in cell.joint_names], dtype=float)

    def _clearance_for(self, *locators: Locator):
        """Clearance context for work that touches these locators.

        A move with a weld at either end is governed by the weld clearance: the gun has to
        be able to reach the panel it is welding, while a transit between two ordinary
        vias has no reason to be that permissive.
        """
        if self.weld_clearance is not None and any(l.is_weld for l in locators):
            return self.cell.clearance(self.weld_clearance)
        return self.cell.clearance(self.cell.obstacle_clearance)

    # -- per-locator anchor states ------------------------------------------
    def _solve_locator(self, loc: Locator, seed: np.ndarray) -> np.ndarray:
        with self._clearance_for(loc):
            q = self.cell.solve_pose(loc.pose_world, [seed, self.start_q])
        if q is None:
            raise PlanningError(f"no collision-free IK for locator '{loc.name}'")
        return q

    def _approach_pose(self, loc: Locator) -> np.ndarray:
        return offset_pose(loc.pose_world, self.axis, self.man.linear_zone_mm)

    # -- main ----------------------------------------------------------------
    def run(self) -> list[Segment]:
        man = self.man
        locators = man.locators
        if len(locators) < 2:
            return []

        anchors: dict[str, np.ndarray] = {}
        seed = self.start_q
        for loc in locators:
            try:
                anchors[loc.name] = self._solve_locator(loc, seed)
                seed = anchors[loc.name]
            except PlanningError as exc:
                self.log(f"  ! {exc}")

        segments: list[Segment] = []
        for a, b in zip(locators, locators[1:]):
            seg = Segment(a.name, b.name)
            self.log(f"  segment {a.name} -> {b.name}")
            try:
                if a.name not in anchors:
                    raise PlanningError(f"no reachable joint solution for '{a.name}'")
                if b.name not in anchors:
                    raise PlanningError(f"no reachable joint solution for '{b.name}'")
                with self._clearance_for(a, b):
                    seg.phases = self._plan_pair(a, b, anchors)
            except PlanningError as exc:
                seg.error = str(exc)
                self.log(f"    ! {exc}")
            segments.append(seg)
        return segments

    def _plan_pair(self, a: Locator, b: Locator, anchors) -> list[Phase]:
        # The path starts at the first locator, not at start_state. start_state is still
        # used to seed inverse kinematics and as a known-clear pose to route a difficult
        # transit through, but it never contributes a waypoint of its own.
        phases: list[Phase] = []
        qa, qb = anchors[a.name], anchors[b.name]
        transit_start, transit_end = qa, qb

        depart_states: list[np.ndarray] = []
        approach_states: list[np.ndarray] = []

        if a.is_weld and self.man.linear_zone_mm > 0:
            depart_states = plan_linear(
                self.cell, a.pose_world, self._approach_pose(a), qa,
                step_mm=self.linear_step_mm)
            transit_start = depart_states[-1]
        if b.is_weld and self.man.linear_zone_mm > 0:
            approach_states = plan_linear(
                self.cell, self._approach_pose(b), b.pose_world, qb,
                step_mm=self.linear_step_mm)
            transit_end = approach_states[0]

        # The robot holds still at a weld while the opening steps from arrive to leave, so
        # that is a phase boundary: same pose, new gun state.
        if a.is_weld:
            phases.append(Phase("weld", LIN, [qa], a.gun_opening_leave, True))
        if depart_states:
            phases.append(Phase("depart", LIN, depart_states, a.gun_opening_leave, True))

        transit = plan_freespace(
            self.cell, transit_start, transit_end,
            attempts=self.ompl_attempts, segment_length=self.segment_length,
            check_step=self.check_step, fallback_via=[self.start_q],
            shortcut_seconds=self.shortcut_seconds, log=self.log)
        opening = a.gun_opening_leave if a.is_weld else 0.0
        phases.append(Phase("freespace", PTP, transit, opening, False))

        if approach_states:
            phases.append(Phase("approach", LIN, approach_states,
                                b.gun_opening_arrive, True))

        # The output is what the controller will execute, so check the reduced path
        # rather than trusting that reduction preserved what the planner found.
        full = [q for ph in phases for q in ph.states]
        problem = validate(self.cell, full, max_step=self.check_step)
        if problem:
            raise PlanningError(f"planned path failed validation: {problem}")

        # Planning may have used a weld pose backed off from the panel; the program has to
        # name the weld where the study put it, so those waypoints go back.  Done after
        # validation because this last step is the gun deliberately closing on the part.
        self._restore_weld_poses(a, b, phases, qa, qb)
        return phases

    def _restore_weld_poses(self, a: Locator, b: Locator, phases: list[Phase],
                            qa: np.ndarray, qb: np.ndarray) -> None:
        """Put the waypoints that sit *at* a weld back onto its imported pose."""
        for loc, anchor, which in ((a, qa, "first"), (b, qb, "last")):
            if not loc.is_weld or loc.pose_world_import is None:
                continue
            # Seeded from the planned anchor, so this stays on the same arm configuration;
            # collision checking is off because contact with the panel is the point.
            q = self.cell.solve_pose(loc.export_pose, [anchor],
                                     require_collision_free=False, branch_seeds=0)
            if q is None:
                self.log(f"    ! cannot reach the imported pose of '{loc.name}'; "
                         f"leaving it at the shifted pose")
                continue
            for ph in phases:
                if which == "first" and ph.kind in ("weld", "depart"):
                    ph.states[0] = q          # the weld itself, and where depart begins
                elif which == "last" and ph.kind == "approach":
                    ph.states[-1] = q
