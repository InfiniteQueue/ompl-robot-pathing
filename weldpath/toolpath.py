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

from . import stagetrace
from .cell import STOP_BAND_JOINT, Cell
from .manifest import Locator, Manifest
from .cartesian import CartesianBudget
from .fallback import FallbackFinder
from .planning import (LIN, PTP, Deadline, LinearZone, OmplBudget, PlanningError,
                       Relocation, unique_openings,
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

# How far past --near-panel-mm the clearance query has to see.  A reading at the probe means
# nothing was found within it, so a probe no wider than the band cannot tell a state inside
# the band from one far outside it.
NEAR_PANEL_HEADROOM_MM = 25.0


class ToolpathPlanner:
    def __init__(self, cell: Cell, man: Manifest, *,
                 ompl: OmplBudget | None = None,
                 cartesian: CartesianBudget | None = None,
                 fallback_runs: int = 0, relocate: Relocation | None = None,
                 segment_seconds: float = 0.0,
                 main_openings: int = 5, extra_openings: int = 0,
                 opening_round_mm: float = 5.0,
                 fallback_mm: float = 100.0, fallback_step_mm: float = 40.0,
                 segment_length: float = 0.02, check_step_deg: float = 3.0,
                 continuous_check: bool = False,
                 shortcut_seconds: float = 2.0, polish_seconds: float = 5.0,
                 near_panel_mm: float = 0.0, near_panel_min_mm: float = 0.0,
                 near_panel_min_pct: float = 0.0, linear_speed_mm_s: float = 0.0,
                 linear_crossing_penalty_s: float = 0.0,
                 linear_introduce_mm: float = 0.0,
                 weld_clearance_mm: float | None = None,
                 stand_off_search: bool = True, stand_off_scan_mm: float = 0.5,
                 stand_off_resolution_mm: float = 0.05,
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
        # Wall clock for one segment's search.  Every budget below it bounds a part of the
        # search and they multiply, so this is what bounds a segment that has no route at
        # all: past it the search stops and the segment is reported as one that found
        # none, which the run loop already knows how to carry on from.  See Deadline.
        self.segment_seconds = segment_seconds
        # How many gun openings a transit is planned over: the rotation phase one deals
        # its runs round, the reserve phase two walks afterwards, and the multiple the
        # values found by bisection are rounded to.  See planning._opening_lists.
        self.main_openings = main_openings
        self.extra_openings = extra_openings
        self.opening_round_mm = opening_round_mm
        # How far the shortcut and polish passes displace a waypoint when they try
        # relocating one, and how the distance is drawn between those bounds.
        self.relocate = relocate or Relocation()
        self.segment_length = segment_length
        self.continuous_check = bool(continuous_check)
        self.check_step = np.deg2rad(check_step_deg)
        self.shortcut_seconds = shortcut_seconds
        self.polish_seconds = polish_seconds
        # Stretches of a transit that run this close to a panel or to tooling come out as
        # linear motion instead of joint motion.  The clearance query has to see past the
        # band, not just to it: a reading at the probe means nothing was found, so a probe
        # equal to the band reads every state outside it as sitting on its edge.
        self.zone = LinearZone(near_mm=near_panel_mm, min_run_mm=near_panel_min_mm,
                               min_run_pct=near_panel_min_pct,
                               linear_speed_mm_s=linear_speed_mm_s,
                               crossing_penalty_s=linear_crossing_penalty_s,
                               introduce_mm=linear_introduce_mm)
        # Every locator reports its measured clearance against the one it has to meet,
        # placed or not, so the query has to see past the larger threshold with room to
        # spare.  A probe that stops at the threshold can only ever answer "at least the
        # requirement", which is the half of the question already known.
        report_probe = max(cell.obstacle_clearance / man.scale,
                           weld_clearance_mm or 0.0) + CLEARANCE_REPORT_HEADROOM_MM
        # ``--linear-introduce-mm`` is read by the same at-or-under test as the band, so
        # the query has to see past whichever of the two is the wider.  Past, not to: a
        # reading at the probe is "nothing found", and a state exactly at the limit would
        # otherwise be indistinguishable from one a metre out.
        wanted = max(max(near_panel_mm, self.zone.reach_mm) + NEAR_PANEL_HEADROOM_MM
                     if self.zone.enabled else 0.0,
                     report_probe)
        if wanted > 0.0:
            cell.require_proximity(wanted, log=log)
        # Set when --export-collision-geometry is on: a pose that cannot be placed then
        # also writes the two links that blocked it, as they sat when it was rejected.
        self.export_dir = export_dir
        self.keep_unrefined = keep_unrefined
        # None means "no separate weld rule", i.e. the cell's own clearance throughout.
        self.weld_clearance = (None if weld_clearance_mm is None
                               else weld_clearance_mm * man.scale)
        # A weld blocked at its shifted pose searches the shorter stand-offs back towards the
        # imported one; see _search_stand_off.
        if stand_off_scan_mm <= 0.0 or stand_off_resolution_mm <= 0.0:
            raise ValueError("the stand-off scan spacing and resolution must be positive")
        self.stand_off_search = bool(stand_off_search)
        self.stand_off_scan_mm = float(stand_off_scan_mm)
        self.stand_off_resolution_mm = float(stand_off_resolution_mm)
        self.start_q = np.array([man.start_state[n] for n in cell.joint_names], dtype=float)
        # Gun opening each locator turned out to be reachable at, filled in by run().
        self.openings: dict[str, float] = {}
        # Poses to route a difficult transit through are searched for per transit and only
        # when one is needed; see weldpath.fallback and _fallback_for.
        self.fallback = FallbackFinder(cell, man, fallback_mm=fallback_mm,
                                       step_mm=fallback_step_mm, log=self.log)

    def _fallback_for(self, a: Locator, b: Locator, qa: np.ndarray, qb: np.ndarray):
        """A callable giving this transit its fallback poses, run only if it is asked.

        Handed to ``plan_freespace`` rather than a list because the search costs real time
        -- three methods, each solving inverse kinematics at every step it takes -- and
        most transits solve at the first gun opening and never reach for a detour.  The
        planner calls this once, after every opening has failed directly.

        This replaces taking the study's first via and routing everything through it.  That
        was one pose for the whole run, chosen without reference to either end of the move
        it was rescuing or to how much room there was around it; it was only ever a guess
        that a pose the path already visited would be a clear one.
        """
        def find() -> list[np.ndarray]:
            return self.fallback.poses(qa, qb, a.pose_world, b.pose_world)
        return find

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
        # Deduped for the same reason a transit's list is: a gun with no travel to speak of
        # collapses all three of these onto zero, and solving the same pose three times
        # over means three identical failures, three diagnoses and three geometry exports
        # before the locator is given up on.
        return unique_openings([declared, widest, widest / 2.0], widest)

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
                q = self.cell.solve_pose(loc.pose_world, [seed, self.start_q],
                                         avoid_stop_band=True)
                searched = ""
                if q is None and self.stand_off_search and loc.pose_world_import is not None:
                    q, searched = self._search_stand_off(loc, seed)
                if q is None:
                    # Diagnosed at the full shift: the search only replaces pose_world when
                    # it finds somewhere clear.
                    tag = f"{loc.name}_at_{opening:g}mm"
                    problems.append(f"at {opening:g} mm, "
                                    f"{self._diagnose(loc, seed, tag, opening)}{searched}")
                else:
                    self._report_clearance(loc, q, opening, placed=True)
            if q is not None:
                if self.cell.in_stop_band(q):
                    # Every solution found was inside it.  A locator is an endpoint the
                    # passes never move, so this one stands.
                    self.log(f"    ! '{loc.name}' has no solution outside the joint 5 "
                             f"stop band; placed at "
                             f"{np.rad2deg(q[STOP_BAND_JOINT]):.1f} deg")
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

    def _search_stand_off(self, loc: Locator, seed: np.ndarray
                          ) -> tuple[np.ndarray | None, str]:
        """Find a shorter stand-off for a weld the full ``--weld-shift-mm`` could not place.

        Candidates lie on the line the shift moved the weld along, from the imported pose (0)
        to the full shift, orientation unchanged.  The one kept is the clear pose reading the
        most clearance, ties going to the larger stand-off.  On success ``loc.pose_world`` is
        replaced, so the transits, the fallback search and the endpoint check all read the
        pose that was actually placed; the imported pose is left alone.

        Not a bisection: clear is not monotonic in the distance.  Backing off frees the tip
        from the panel but can put the throat into tooling behind it, so there may be several
        clear windows, some narrower than any fixed scan.  So:

        1. scan the range every ``stand_off_scan_mm``;
        2. split every gap between two blocked samples in half, and again, until the spacing
           is at or below ``stand_off_resolution_mm`` -- carrying on after something is
           clear, since a better window may be narrower.  A window narrower than the final
           spacing can still fall between samples and be missed;
        3. in every clear window, climb from its best sample: try half the scan spacing
           either side, keep whichever reads more clearance, halve, and stop once the step is
           at or below the resolution.  This finds the best point near that sample, not
           necessarily a narrower peak elsewhere in a wide window.

        Returns the state, or None and a note for the failure message.  Call inside the
        clearance and gun-opening context the locator is being solved in.
        """
        imported = np.asarray(loc.pose_world_import, dtype=float)
        offset = np.asarray(loc.pose_world, dtype=float)[:3, 3] - imported[:3, 3]
        span = float(np.linalg.norm(offset))
        if span <= 0.0:
            return None, ""
        # Signed as --weld-shift-mm is, along the locator's own z.
        sign = 1.0 if float(offset @ imported[:3, 2]) >= 0.0 else -1.0
        cells = max(1, int(np.ceil(span / self.stand_off_scan_mm - 1e-9)))
        step = span / cells

        def pose_at(d: float) -> np.ndarray:
            P = imported.copy()
            P[:3, 3] += offset * (d / span)
            return P

        # Distance -> (state, clearance), a blocked pose reading -inf.  The full shift has
        # just failed, so it bounds the search without being solved again.
        samples: dict[float, tuple[np.ndarray | None, float]] = {
            round(span, 9): (None, -np.inf)}

        def sample(d: float) -> float:
            d = round(min(max(d, 0.0), span), 9)
            if d not in samples:
                q = self.cell.solve_pose(pose_at(d), [seed, self.start_q],
                                         avoid_stop_band=True)
                samples[d] = (q, -np.inf if q is None else self.cell.clearance_mm(q))
            return d

        def clear(d: float) -> bool:
            return samples[d][0] is not None

        def rank(d: float) -> tuple[float, float]:
            return samples[d][1], d

        for k in range(cells):
            sample(k * step)
        coarse = step
        # Split every gap between two blocked samples, whether or not something is already
        # clear elsewhere: the first window found need not be the best one, and stopping there
        # measured 1.2 mm of clearance with a 3.3 mm window sitting unsampled further out.
        # A gap with a clear end is not split -- anything clear inside it runs on from that
        # sample's window, which the climb below explores.  Blocked samples are cheap, being
        # rejected before any clearance is measured.
        while step > self.stand_off_resolution_mm + 1e-9:
            grid = sorted(samples)
            step /= 2.0
            gaps = [(a + b) / 2.0 for a, b in zip(grid, grid[1:])
                    if not clear(a) and not clear(b)]
            if not gaps:
                break
            for d in gaps:
                sample(d)

        # A run of clear samples with no blocked one between them is a window.
        windows: list[list[float]] = []
        run: list[float] = []
        for d in sorted(samples):
            if clear(d):
                run.append(d)
            elif run:
                windows.append(run)
                run = []
        if run:
            windows.append(run)
        if not windows:
            return None, (f"; no shorter stand-off is clear either ({len(samples) - 1} tried "
                          f"between 0 and {sign * span:+g} mm, {step:.3g} mm apart)")

        for run in windows:
            d = max(run, key=rank)
            h = coarse / 2.0
            while True:
                d = max((d, sample(d - h), sample(d + h)), key=rank)
                if h <= self.stand_off_resolution_mm + 1e-9:
                    break
                h /= 2.0

        best = max((d for d in samples if clear(d)), key=rank)
        shown = ", ".join(f"{sign * r[0]:+.2f}" if len(r) == 1 else
                          f"{sign * r[0]:+.2f} to {sign * r[-1]:+.2f}" for r in windows)
        self.log(f"    '{loc.name}' is blocked at the full {sign * span:+g} mm stand-off; "
                 f"clear at {shown} mm ({len(samples) - 1} tried), standing off "
                 f"{sign * best:+.2f} mm")
        loc.pose_world = pose_at(best)
        return samples[best][0], ""

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
        self.fallback.announce()

        segments: list[Segment] = []
        pairs = list(zip(locators, locators[1:]))
        for index, (a, b) in enumerate(pairs):
            seg = Segment(a.name, b.name)
            self.log(f"  segment {a.name} -> {b.name} [{index + 1}/{len(pairs)}]")
            stagetrace.section(f"segment {a.name} -> {b.name} [{index + 1}/{len(pairs)}]")
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

        The departure opening leads: it is the one already in force, so using it costs no
        gun change at the start of the move, and it is the state the robot was proved to
        stand at ``a`` in.  The arrival opening follows, being the state it has to be
        holding by the time it reaches ``b``.

        This used to put the arrival opening first where ``b`` was a weld and ``a`` was
        not, on the grounds that a weld's opening is process data and the pose was only
        ever checked in it.  That reasoning still holds for the destination pose; what it
        did not account for is that the transit has to leave ``a`` as well as arrive at
        ``b``, and the same argument applies at that end.  The order is now uniform, and
        the arrival opening is still tried second rather than not at all.
        """
        return [leave_open, arrive_open]

    def _plan_pair(self, a: Locator, b: Locator, anchors, final: bool = False
                   ) -> tuple[list[Phase], list[Phase]]:
        # The path starts at the first locator, not at start_state. start_state seeds
        # inverse kinematics and sets the gun's initial opening; it never contributes a
        # waypoint of its own, and it is not what a difficult transit detours through --
        # see weldpath.fallback.
        # Started here rather than at the transit, so that everything this segment does
        # counts against it -- placing the weld, searching for a fallback pose, and the
        # refinement -- even though the search loops are what actually read it.
        deadline = Deadline(self.segment_seconds)
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
            continuous_check=self.continuous_check,
            check_step=self.check_step,
            fallback_via=self._fallback_for(a, b, transit_start, transit_end),
            fallback_runs=self.fallback_runs, relocate=self.relocate,
            deadline=deadline,
            shortcut_seconds=self.shortcut_seconds,
            polish_seconds=self.polish_seconds,
            zone=self.zone,
            openings=self._transit_openings(a, b, leave_open, arrive_open),
            main_openings=self.main_openings,
            extra_openings=self.extra_openings,
            opening_round_mm=self.opening_round_mm,
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
                                 require_collision_free=False, branch_seeds=0,
                                 avoid_stop_band=True)
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
