"""Turn the manifest's ordered locators into planned, phased motion.

Locators are visited in manifest order and one output segment is produced per consecutive
pair, matching the sample output.

A transit runs weld to weld and is free to curve away from the panel immediately.  Welds
used to get a straight lead-in and lead-out of a fixed length along one fixed direction,
which asked for a clear tunnel a weld set deep in panelling often has no room for; where a
straight run near the panel is wanted it is now found by measuring the route, in
:func:`~weldpath.planning._finish`, rather than assumed at the weld.  Per the brief the
gun is treated as stationary through the weld: the robot stops at the locator, the opening
changes from ``gun_opening_arrive`` to ``gun_opening_leave``, and no motion is planned for
the closing itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cell import Cell
from .manifest import Locator, Manifest
from .cartesian import CartesianBudget
from .planning import (LIN, PTP, LinearZone, OmplBudget, PlanningError, Relocation,
                       plan_freespace,
                       validate)



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


# Headroom past the clearance a locator has to meet, so the report can distinguish a pose
# that just scrapes past from one with room around it.  Without it the query stops at the
# threshold and every success reads the same.
CLEARANCE_REPORT_HEADROOM_MM = 25.0


class ToolpathPlanner:
    def __init__(self, cell: Cell, man: Manifest, *,
                 ompl: OmplBudget | None = None,
                 cartesian: CartesianBudget | None = None,
                 fallback_runs: int = 0, relocate: Relocation | None = None,
                 segment_length: float = 0.02, check_step_deg: float = 3.0,
                 shortcut_seconds: float = 2.0, polish_seconds: float = 5.0,
                 near_panel_mm: float = 0.0, near_panel_min_mm: float = 0.0,
                 near_panel_min_pct: float = 0.0, linear_speed_mm_s: float = 0.0,
                 linear_crossing_penalty_s: float = 0.0,
                 weld_clearance_mm: float | None = None,
                 export_dir: str | None = None,
                 keep_unrefined: bool = False, log=print):
        self.cell = cell
        self.man = man
        self.log = log
        self.ompl = ompl or OmplBudget()
        # Only ever reached when the sampling planner's first phase came back empty, and
        # only where the near-panel band is in force.  A cell whose transits solve never
        # pays for it.
        self.cartesian = cartesian
        self.fallback_runs = fallback_runs
        # How far the shortcut and polish passes displace a waypoint when they try
        # relocating one, and how the distance is drawn between those bounds.
        self.relocate = relocate or Relocation()
        self.segment_length = segment_length
        self.check_step = np.deg2rad(check_step_deg)
        self.shortcut_seconds = shortcut_seconds
        self.polish_seconds = polish_seconds
        # Stretches of a transit that run this close to a panel or to tooling come out as
        # linear motion instead of joint motion.  The clearance query has to be able to
        # see that far, which it will not do on the penalty's probe alone.
        self.zone = LinearZone(near_mm=near_panel_mm, min_run_mm=near_panel_min_mm,
                               min_run_pct=near_panel_min_pct,
                               linear_speed_mm_s=linear_speed_mm_s,
                               crossing_penalty_s=linear_crossing_penalty_s)
        # Every locator reports its measured clearance against the one it has to meet,
        # placed or not, so the query has to see past the larger threshold with room to
        # spare.  A probe that stops at the threshold can only ever answer "at least the
        # requirement", which is the half of the question already known.
        report_probe = max(cell.obstacle_clearance / man.scale,
                           weld_clearance_mm or 0.0) + CLEARANCE_REPORT_HEADROOM_MM
        wanted = max(near_panel_mm if self.zone.enabled else 0.0, report_probe)
        if wanted > 0.0:
            cell.require_proximity(wanted, log=log)
        # Set when --export-collision-geometry is on: a pose that cannot be placed then
        # also writes the two links that blocked it, as they sat when it was rejected.
        self.export_dir = export_dir
        self.keep_unrefined = keep_unrefined
        # None means "no separate weld rule", i.e. the cell's own clearance throughout.
        self.weld_clearance = (None if weld_clearance_mm is None
                               else weld_clearance_mm * man.scale)
        self.start_q = np.array([man.start_state[n] for n in cell.joint_names], dtype=float)
        # Gun opening each locator turned out to be reachable at, filled in by run().
        self.openings: dict[str, float] = {}
        # Poses to route a difficult transit through; filled in by run() once the locators
        # have been solved, since they are locator configurations rather than a constant.
        self.fallback_via: list[np.ndarray] = []

    def _fallback_poses(self, anchors: dict[str, np.ndarray]) -> list[np.ndarray]:
        """Poses worth routing a difficult transit through, most preferred first.

        The first via, in the configuration the robot actually reaches it in -- which is
        what ``anchors`` holds, inverse kinematics having been solved for it and seeded
        from ``start_state`` through the chain of locators before it.

        A via is a pose the path already visits in open space, so it is known reachable,
        known clear, and known to be somewhere the job wants the robot to be.  A weld is
        not a candidate: it is a pose at the panel, which is the last place to send a
        transit that is struggling for room.

        ``start_state`` is deliberately not used here.  It is a seed for inverse
        kinematics and the manifest makes no claim that it is a home position, a staging
        pose, or related to the path at all -- so routing through it was reading a promise
        into the file that the file does not make.  Where there is no via to use, this
        returns nothing and the two-leg fallback is simply unavailable, rather than
        substituting a pose whose suitability nothing has established.
        """
        for loc in self.man.locators:
            if loc.is_weld:
                continue
            q = anchors.get(loc.name)
            if q is None:
                continue
            self.log(f"  difficult transits will be routed through via '{loc.name}'")
            return [q]
        self.log("  no via is available to route difficult transits through; transits "
                 "that need one will fail rather than detour through the seed pose")
        return []

    def _clearance_for(self, *locators: Locator):
        """Clearance context for work that touches these locators.

        A move with a weld at either end is governed by the weld clearance: the gun has to
        be able to reach the panel it is welding, while a transit between two ordinary
        vias has no reason to be that permissive.
        """
        if self.weld_clearance is not None and any(l.is_weld for l in locators):
            return self.cell.clearance(self.weld_clearance)
        return self.cell.clearance(self.cell.obstacle_clearance)

    def _report_clearance(self, loc: Locator, q: np.ndarray | None,
                          opening: float, placed: bool) -> None:
        """State the clearance the pose actually has beside the one it has to meet.

        Printed for every locator on the first pass whether it was placed or not, because
        the two cases raise the same question and neither answered it before.  A weld that
        was placed with 0.3 mm to spare is a weld the next change to the geometry will
        break, and nothing said so; a weld that was rejected by 30 mm is a different
        problem from one rejected by 0.3, and the blocking pair alone does not separate
        them.  Call inside the clearance and gun-opening context, so the requirement read
        here is the one the solve was actually held to.
        """
        required = self.cell.obstacle_clearance / self.man.scale
        verdict = "placed" if placed else "rejected"
        if q is None:
            self.log(f"    '{loc.name}' at {opening:g} mm: no pose to measure "
                     f"({required:+.1f} mm required) -- {verdict}")
            return
        got = self.cell.clearance_mm(q)
        probe = self.cell.probe_mm
        if not np.isfinite(got):
            shown = "unmeasured"
        elif probe > 0.0 and got >= probe:
            shown = f"beyond {probe:+.1f} mm"     # the query ran out of range, not a reading
        else:
            shown = f"{got:+.1f} mm"
        self.log(f"    '{loc.name}' at {opening:g} mm: clearance {shown} against "
                 f"{required:+.1f} mm required -- {verdict}")

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

    def _diagnose(self, loc: Locator, seed: np.ndarray, tag: str = "",
                  opening: float = 0.0) -> str:
        """Why the pose was rejected.  Call inside the clearance and gun-opening context.

        Worth the extra solve: "no solution" and "solution rejected by the collision check"
        need completely different fixes -- a reachability problem against a geometry one --
        and a message that covers both sends the reader after the wrong one.  Naming the
        blocking pair and its depth turns a modelling artefact into something obvious,
        since convex hulls over-report contact badly on the C-shaped castings here.
        """
        q = self.cell.solve_pose(loc.pose_world, [seed, self.start_q],
                                 require_collision_free=False)
        # Measured off this pose rather than a third solve of its own: the collision-free
        # solve returned nothing to measure, and this is the same pose the pair below is
        # named from, so the two lines describe one state.
        self._report_clearance(loc, q, opening, placed=False)
        if q is None:
            return "no inverse-kinematics solution inside the joint limits"
        hits = self.cell.contacts(q)
        if not hits:
            return "reachable, but the pose reads as in collision"
        (a, b), (worst, point) = min(hits.items(), key=lambda kv: kv[1][0])
        more = f" (+{len(hits) - 1} more)" if len(hits) > 1 else ""
        # Where the contact sits, said in the frame the reader is looking at: the TCP's
        # own axes are the ones the approach and the lead-in are defined along, so "40 mm
        # behind and 12 mm off to the side" locates it on the gun without a viewer.
        local = self.cell.in_tcp_frame(q, point)
        self._export_blocked(q, (a, b), tag or loc.name)
        where = ("" if not np.all(np.isfinite(local)) else
                 f"; closest at ({local[0]:+.1f}, {local[1]:+.1f}, {local[2]:+.1f}) mm "
                 f"from the TCP, in the TCP's own axes")
        return (f"reachable, but {a} <-> {b} reads {worst / self.man.scale:+.1f} mm"
                f"{more} against a "
                f"{self.cell.obstacle_clearance / self.man.scale:+.1f} mm clearance{where}")

    def _export_blocked(self, q: np.ndarray, pair: tuple[str, str], tag: str) -> None:
        """Write the blocking geometry, if the run was asked for collision geometry.

        The state has to be pushed into the environment first: the contact test that found
        this pair went through the contact manager, which carries its own transforms, and
        the exporter reads the environment's.
        """
        if not self.export_dir:
            return
        from . import hullexport
        try:
            self.cell.set_state(q)
            hullexport.export_pair(self.cell.env, self.man, self.export_dir, pair, tag,
                                   log=self.log)
        except Exception as exc:                # diagnostics must not mask the failure
            self.log(f"      could not write the blocking geometry: "
                     f"{type(exc).__name__}: {exc}")

    def _solve_locator(self, loc: Locator, seed: np.ndarray) -> tuple[np.ndarray, float]:
        """Joint solution for a locator, plus the gun opening it was reached at."""
        problems = []
        for opening in self._locator_openings(loc):
            with self._clearance_for(loc), self.cell.gun_opening(opening):
                q = self.cell.solve_pose(loc.pose_world, [seed, self.start_q])
                if q is None:
                    tag = f"{loc.name}_at_{opening:g}mm"
                    problems.append(f"at {opening:g} mm, "
                                    f"{self._diagnose(loc, seed, tag, opening)}")
                else:
                    self._report_clearance(loc, q, opening, placed=True)
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
        self.fallback_via = self._fallback_poses(anchors)

        segments: list[Segment] = []
        pairs = list(zip(locators, locators[1:]))
        for index, (a, b) in enumerate(pairs):
            seg = Segment(a.name, b.name)
            self.log(f"  segment {a.name} -> {b.name}")
            try:
                if a.name not in anchors:
                    raise PlanningError(f"no reachable joint solution for '{a.name}'")
                if b.name not in anchors:
                    raise PlanningError(f"no reachable joint solution for '{b.name}'")
                with self._clearance_for(a, b):
                    seg.phases, seg.raw_phases = self._plan_pair(
                        a, b, anchors, final=index == len(pairs) - 1)
            except PlanningError as exc:
                seg.error = str(exc)
                self.log(f"    ! {exc}")
            segments.append(seg)
        return segments

    @staticmethod
    def _transit_openings(a: Locator, b: Locator, leave_open: float,
                          arrive_open: float) -> list[float]:
        """Gun openings to try for the transit between two locators, best first.

        The weld end is the constrained one: its opening is process data, and it is the
        opening the anchor and the linear approach were actually proved reachable at.  Try
        the free end's opening first and the planner spends a full time budget discovering
        that the pose it has to finish at was never checked in that gun state.

        With a weld at both ends or neither, the departure opening leads: it is the one
        already in force, so using it costs no gun change at the start of the move.
        """
        if b.is_weld and not a.is_weld:
            return [arrive_open, leave_open]
        return [leave_open, arrive_open]

    def _plan_pair(self, a: Locator, b: Locator, anchors, final: bool = False
                   ) -> tuple[list[Phase], list[Phase]]:
        # The path starts at the first locator, not at start_state. start_state seeds
        # inverse kinematics and sets the gun's initial opening; it never contributes a
        # waypoint of its own, and it is not what a difficult transit detours through --
        # see _fallback_poses.
        phases: list[Phase] = []
        qa, qb = anchors[a.name], anchors[b.name]
        transit_start, transit_end = qa, qb
        leave_open, arrive_open = self._leave_opening(a), self._arrive_opening(b)

        # The robot holds still at a weld while the opening steps from arrive to leave, so
        # that is a phase boundary: same pose, new gun state.  That gun motion is not
        # simulated -- the robot is stationary and the tip's own travel is clear by
        # inspection, and not simulating it is what keeps the weld from needing the panel
        # collisions switched off.
        if a.is_weld:
            phases.append(Phase("weld", LIN, [qa], leave_open, True))

        # Phases up to here are shared with the unrefined copy: nothing optimises a linear
        # move, so it is the same motion in both files.
        raw_phases: list[Phase] = list(phases) if self.keep_unrefined else []

        raw_legs: list | None = [] if self.keep_unrefined else None
        legs = plan_freespace(
            self.cell, transit_start, transit_end,
            ompl=self.ompl, cartesian=self.cartesian,
            segment_length=self.segment_length,
            check_step=self.check_step, fallback_via=self.fallback_via,
            fallback_runs=self.fallback_runs, relocate=self.relocate,
            shortcut_seconds=self.shortcut_seconds,
            polish_seconds=self.polish_seconds,
            zone=self.zone,
            openings=self._transit_openings(a, b, leave_open, arrive_open),
            record=raw_legs, log=self.log)
        for i, (runs, opening) in enumerate(legs):
            opening_mm = 0.0 if opening is None else opening
            for run in runs:
                # A linear run is the gun working its way over the panel rather than
                # crossing open space, so it is named for what it is in the run log.
                kind = "freespace" if run.motion == PTP else "traverse"
                phases.append(Phase(kind, run.motion, run.states, opening_mm, False))
            if raw_legs is not None and i < len(raw_legs):
                raw_phases.append(Phase("freespace", PTP, raw_legs[i], opening_mm, False))


        # The output is what the controller will execute, so check the reduced path rather
        # than trusting that reduction preserved what the planner found.  Checked per phase,
        # under that phase's own gun opening: a single sweep at one opening would validate a
        # tip position the robot never holds.
        for ph in phases:
            with self.cell.gun_opening(ph.gun_opening_mm):
                problem = validate(self.cell, ph.states, max_step=self.check_step,
                                   motion=ph.motion, zone=self.zone)
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
        self._restore_weld_poses(a, phases, qa)

        # A weld the tour ends on is never any segment's ``a``, so it never gets a phase of
        # its own and its imported pose would appear nowhere at all -- the run would stop on
        # the stand-off, a weld shift short of the weld it names.  Only the last segment
        # needs this: every other arriving weld is the next segment's departing one.  Built
        # after validation for the same reason as the restore above, so the short move onto
        # the panel is neither planned nor checked.
        if final and b.is_weld:
            q = self._imported_state(b, qb)
            if q is not None:
                closing = Phase("weld", LIN, [q], self._leave_opening(b), True)
                phases.append(closing)
                if self.keep_unrefined:
                    # By reference, as the opening weld phase is: nothing optimises it, so
                    # the two files describe the same motion here.
                    raw_phases.append(closing)
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

    def _imported_state(self, loc: Locator, seed: np.ndarray) -> np.ndarray | None:
        """Joint state standing on a weld's imported pose, or None if it cannot be reached.

        Seeded from the planned anchor, so this stays on the same arm configuration;
        collision checking is off because contact with the panel is the point.
        """
        if not loc.is_weld or loc.pose_world_import is None:
            return None
        q = self.cell.solve_pose(loc.export_pose, [seed],
                                 require_collision_free=False, branch_seeds=0)
        if q is None:
            self.log(f"    ! cannot reach the imported pose of '{loc.name}'; "
                     f"leaving it at the shifted pose")
        return q

    def _restore_weld_poses(self, a: Locator, phases: list[Phase],
                            qa: np.ndarray) -> None:
        """Put the waypoint that sits *at* a weld back onto its imported pose.

        Only the departing weld has such a waypoint.  ``a`` is where this segment starts,
        and the ``weld`` phase holds the robot there while the gun closes, so that is the
        one state in the segment standing on the locator itself.  The arriving weld is
        reached by the transit, whose last state is the planned -- shifted -- anchor, and
        nothing here rewrites it: that stand-off is the gun clear of the panel, and it is
        half of the pair the weld sits between.
        """
        q = None if not a.is_weld else self._imported_state(a, qa)
        if q is None:
            return
        for ph in phases:
            if ph.kind == "weld":
                ph.states[0] = q
