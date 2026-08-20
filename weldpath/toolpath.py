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
from .planning import (LIN, PTP, LinearZone, PlanningError, plan_freespace, plan_linear,
                       validate)

AXES = {"+x": np.array([1.0, 0, 0]), "-x": np.array([-1.0, 0, 0]),
        "+y": np.array([0, 1.0, 0]), "-y": np.array([0, -1.0, 0]),
        "+z": np.array([0, 0, 1.0]), "-z": np.array([0, 0, -1.0])}



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
    # The same motion with each freespace transit as the sampling planner returned it,
    # before shortcutting and reduction. The linear phases are shared with ``phases``,
    # since nothing optimises those. Only populated when it is asked for.
    raw_phases: list[Phase] = field(default_factory=list)


def retract_axis(override: str | None = None) -> np.ndarray:
    """Direction, in a weld locator's own frame, that the robot retracts along.

    The locator carries its own orientation, and its z axis is the one authored against the
    joint, so the retract is defined relative to that rather than derived from the tool.
    Deriving it from the gun's stroke -- as this did while the gun was prismatic -- made
    the direction a property of the machine instead of the weld, and gave no answer at all
    once the gun became angular.

    The default is ``-x``.  Note that ``--weld-shift-mm`` still backs the tool off along the
    locator's **z**, so the two are no longer the same direction; see the README.
    """
    return AXES[override.lower()] if override else AXES["-x"]


def offset_pose(pose: np.ndarray, axis_tcp: np.ndarray, distance: float) -> np.ndarray:
    """Shift ``pose`` along a direction expressed in its own frame."""
    out = np.array(pose, dtype=float).copy()
    out[:3, 3] = out[:3, 3] + out[:3, :3] @ (axis_tcp * distance)
    return out


class ToolpathPlanner:
    def __init__(self, cell: Cell, man: Manifest, *, approach_axis: str | None = None,
                 linear_step_mm: float = 50.0, ompl_attempts: int = 3,
                 ompl_runs: int = 1,
                 segment_length: float = 0.02, check_step_deg: float = 3.0,
                 ompl_seconds: float = 5.0, linear_zone: bool = True,
                 shortcut_seconds: float = 2.0, polish_seconds: float = 5.0,
                 near_panel_mm: float = 0.0, near_panel_min_mm: float = 0.0,
                 weld_clearance_mm: float | None = None,
                 keep_unrefined: bool = False, log=print):
        self.cell = cell
        self.man = man
        self.log = log
        self.axis = retract_axis(approach_axis)
        # 0 turns the straight lead-in and lead-out off entirely: the transit then runs
        # weld to weld, free to curve away from the panel from the first millimetre.
        self.linear_zone_mm = man.linear_zone_mm if linear_zone else 0.0
        self.linear_step_mm = linear_step_mm
        self.ompl_attempts = ompl_attempts
        self.ompl_runs = ompl_runs
        self.ompl_seconds = ompl_seconds
        self.segment_length = segment_length
        self.check_step = np.deg2rad(check_step_deg)
        self.shortcut_seconds = shortcut_seconds
        self.polish_seconds = polish_seconds
        # Stretches of a transit that run this close to a panel or to tooling come out as
        # linear motion instead of joint motion.  The clearance query has to be able to
        # see that far, which it will not do on the penalty's probe alone.
        self.zone = LinearZone(near_mm=near_panel_mm, min_run_mm=near_panel_min_mm,
                               step_mm=linear_step_mm)
        if self.zone.enabled:
            cell.require_proximity(near_panel_mm)
        self.keep_unrefined = keep_unrefined
        # None means "no separate weld rule", i.e. the cell's own clearance throughout.
        self.weld_clearance = (None if weld_clearance_mm is None
                               else weld_clearance_mm * man.scale)
        self.start_q = np.array([man.start_state[n] for n in cell.joint_names], dtype=float)
        # Gun opening each locator turned out to be reachable at, filled in by run().
        self.openings: dict[str, float] = {}

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
    def _locator_openings(self, loc: Locator) -> list[float]:
        """Gun openings worth trying when reaching this locator, most preferred first.

        A weld's openings are process data: the gun has to be where the weld schedule says,
        so if the robot cannot reach the pose at that opening the answer is a failure, not
        a wider gun.  An ordinary via carries no such requirement -- its declared opening is
        simply zero -- so if the tip fouls something there, opening or closing it is a
        legitimate way through and is tried.
        """
        declared = loc.gun_opening_arrive if loc.is_weld else 0.0
        if loc.is_weld or not self.cell.gun_joint_name:
            return [declared]
        widest = self.man.gun_opening_max
        return [declared, widest, widest / 2.0]

    def _diagnose(self, loc: Locator, seed: np.ndarray) -> str:
        """Why the pose was rejected.  Call inside the clearance and gun-opening context.

        Worth the extra solve: "no solution" and "solution rejected by the collision check"
        need completely different fixes -- a reachability problem against a geometry one --
        and a message that covers both sends the reader after the wrong one.  Naming the
        blocking pair and its depth turns a modelling artefact into something obvious,
        since convex hulls over-report contact badly on the C-shaped castings here.
        """
        q = self.cell.solve_pose(loc.pose_world, [seed, self.start_q],
                                 require_collision_free=False)
        if q is None:
            return "no inverse-kinematics solution inside the joint limits"
        hits = self.cell.contact_pairs(q)
        if not hits:
            return "reachable, but the pose reads as in collision"
        (a, b), worst = min(hits.items(), key=lambda kv: kv[1])
        more = f" (+{len(hits) - 1} more)" if len(hits) > 1 else ""
        return (f"reachable, but {a} <-> {b} reads {worst / self.man.scale:+.1f} mm"
                f"{more} against a {self.cell.obstacle_clearance / self.man.scale:+.1f} mm "
                f"clearance")

    def _solve_locator(self, loc: Locator, seed: np.ndarray) -> tuple[np.ndarray, float]:
        """Joint solution for a locator, plus the gun opening it was reached at."""
        problems = []
        for opening in self._locator_openings(loc):
            with self._clearance_for(loc), self.cell.gun_opening(opening):
                q = self.cell.solve_pose(loc.pose_world, [seed, self.start_q])
                if q is None:
                    problems.append(f"at {opening:g} mm, {self._diagnose(loc, seed)}")
            if q is not None:
                if opening:
                    self.log(f"    '{loc.name}' needs the gun at {opening:g} mm to be "
                             f"reachable")
                return q, opening
        detail = "; ".join(problems)
        hint = ("" if "reachable, but" not in detail else
                ". Hulls over-report contact on concave parts: check the pair against the "
                "raw geometry before trusting it, and see --hull-cell-mm")
        raise PlanningError(f"cannot place the robot at locator '{loc.name}' -- "
                            f"{detail}{hint}")

    def _approach_pose(self, loc: Locator) -> np.ndarray:
        return offset_pose(loc.pose_world, self.axis, self.linear_zone_mm)

    def _leave_opening(self, loc: Locator) -> float:
        """Gun opening in force as the robot leaves this locator."""
        return loc.gun_opening_leave if loc.is_weld else self.openings.get(loc.name, 0.0)

    def _arrive_opening(self, loc: Locator) -> float:
        """Gun opening the robot must be holding as it reaches this locator."""
        return loc.gun_opening_arrive if loc.is_weld else self.openings.get(loc.name, 0.0)

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
                anchors[loc.name], self.openings[loc.name] = self._solve_locator(loc, seed)
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
                    seg.phases, seg.raw_phases = self._plan_pair(a, b, anchors)
            except PlanningError as exc:
                seg.error = str(exc)
                self.log(f"    ! {exc}")
            segments.append(seg)
        return segments

    def _plan_pair(self, a: Locator, b: Locator, anchors
                   ) -> tuple[list[Phase], list[Phase]]:
        # The path starts at the first locator, not at start_state. start_state is still
        # used to seed inverse kinematics and as a known-clear pose to route a difficult
        # transit through, but it never contributes a waypoint of its own.
        phases: list[Phase] = []
        qa, qb = anchors[a.name], anchors[b.name]
        transit_start, transit_end = qa, qb

        depart_states: list[np.ndarray] = []
        approach_states: list[np.ndarray] = []
        leave_open, arrive_open = self._leave_opening(a), self._arrive_opening(b)

        # Each stretch of motion is planned with the tip where it will actually be. The gun
        # is 200 mm of swinging geometry, so a linear retract that clears with it closed can
        # foul with it open, and planning both against one arbitrary opening proves nothing.
        if a.is_weld and self.linear_zone_mm > 0:
            with self.cell.gun_opening(leave_open):
                depart_states = plan_linear(
                    self.cell, a.pose_world, self._approach_pose(a), qa,
                    step_mm=self.linear_step_mm)
            transit_start = depart_states[-1]
        if b.is_weld and self.linear_zone_mm > 0:
            with self.cell.gun_opening(arrive_open):
                approach_states = plan_linear(
                    self.cell, self._approach_pose(b), b.pose_world, qb,
                    step_mm=self.linear_step_mm)
            transit_end = approach_states[0]

        # The robot holds still at a weld while the opening steps from arrive to leave, so
        # that is a phase boundary: same pose, new gun state.  That gun motion is not
        # simulated -- the robot is stationary and the tip's own travel is clear by
        # inspection, and not simulating it is what keeps the weld from needing the panel
        # collisions switched off.
        if a.is_weld:
            phases.append(Phase("weld", LIN, [qa], leave_open, True))
        if depart_states:
            phases.append(Phase("depart", LIN, depart_states, leave_open, True))

        # Phases up to here are shared with the unrefined copy: nothing optimises a linear
        # move, so it is the same motion in both files.
        raw_phases: list[Phase] = list(phases) if self.keep_unrefined else []

        raw_legs: list | None = [] if self.keep_unrefined else None
        legs = plan_freespace(
            self.cell, transit_start, transit_end,
            attempts=self.ompl_attempts, runs=self.ompl_runs,
            segment_length=self.segment_length,
            check_step=self.check_step, fallback_via=[self.start_q],
            shortcut_seconds=self.shortcut_seconds,
            polish_seconds=self.polish_seconds, planning_time=self.ompl_seconds,
            zone=self.zone,
            openings=[leave_open, arrive_open], record=raw_legs, log=self.log)
        for i, (runs, opening) in enumerate(legs):
            opening_mm = 0.0 if opening is None else opening
            for run in runs:
                # A linear run is the gun working its way over the panel rather than
                # crossing open space, so it is named for what it is in the run log.
                kind = "freespace" if run.motion == PTP else "traverse"
                phases.append(Phase(kind, run.motion, run.states, opening_mm, False))
            if raw_legs is not None and i < len(raw_legs):
                raw_phases.append(Phase("freespace", PTP, raw_legs[i], opening_mm, False))

        if approach_states:
            approach = Phase("approach", LIN, approach_states, arrive_open, True)
            phases.append(approach)
            if self.keep_unrefined:
                raw_phases.append(approach)

        # The output is what the controller will execute, so check the reduced path rather
        # than trusting that reduction preserved what the planner found.  Checked per phase,
        # under that phase's own gun opening: a single sweep at one opening would validate a
        # tip position the robot never holds.
        for ph in phases:
            with self.cell.gun_opening(ph.gun_opening_mm):
                problem = validate(self.cell, ph.states, max_step=self.check_step)
            if problem:
                raise PlanningError(f"planned {ph.kind} failed validation: {problem}")
        problem = self._validate_junctions(phases)
        if problem:
            raise PlanningError(f"planned path failed validation: {problem}")

        # Planning may have used a weld pose backed off from the panel; the program has to
        # name the weld where the study put it, so those waypoints go back.  Done after
        # validation because this last step is the gun deliberately closing on the part.
        # This only rewrites the weld, depart and approach phases, which the unrefined copy
        # holds by reference rather than by value, so it lands on both at once.
        self._restore_weld_poses(a, b, phases, qa, qb)
        return phases, raw_phases

    def _validate_junctions(self, phases: list[Phase]) -> str | None:
        """Check the handover between consecutive phases.

        By construction each phase begins at the state the previous one ended on, so a
        junction is normally the gun changing while the robot stands still -- deliberately
        not simulated.  Where the states are *not* identical the robot really does move
        across the boundary, and that move is checked at both openings, since which one is
        in force during it is a matter for the controller rather than this planner.
        """
        for first, second in zip(phases, phases[1:]):
            if not first.states or not second.states:
                continue
            end, start = first.states[-1], second.states[0]
            if np.allclose(end, start, atol=1e-9):
                continue
            for opening in {first.gun_opening_mm, second.gun_opening_mm}:
                with self.cell.gun_opening(opening):
                    if self.cell.segment_collides(end, start, max_step=self.check_step):
                        return (f"the move from {first.kind} into {second.kind} is not "
                                f"collision free with the gun at {opening:g} mm")
        return None

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
