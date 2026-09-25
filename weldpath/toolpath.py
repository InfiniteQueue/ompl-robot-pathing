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
class _Scan:
    """What one sweep of a parameter found: the winner, every sample, and the windows."""
    best: float | None              # the value kept, or None where nothing was clear
    samples: dict                   # value -> (state or None, clearance)
    windows: list                   # runs of consecutive clear samples, in order
    spacing: float                  # how fine the sweep ended up

    @property
    def tried(self) -> int:
        return len(self.samples)

    def describe(self, scale: float = 1.0, plus: bool = False) -> str:
        """The clear windows, as a log line says them.  ``scale`` carries a sign.

        ``plus`` prints the sign on positive values, which a stand-off wants -- it is a
        displacement, and the direction is half of what it says -- and an opening does not.
        """
        fmt = "{:+.2f}" if plus else "{:.2f}"
        return ", ".join(
            fmt.format(scale * r[0]) if len(r) == 1 else
            f"{fmt.format(scale * r[0])} to {fmt.format(scale * r[-1])}"
            for r in self.windows)


def _scan_for_clear(lo: float, hi: float, *, scan: float, resolution: float,
                    measure, known: dict | None = None) -> _Scan:
    """The value in ``lo..hi`` placing the robot with the most room, and where else was clear.

    One algorithm on one parameter, shared by the two axes a blocked weld can be freed
    along: how far its pose stands off the panel, and how far the gun is open.  All it
    wants of the caller is ``measure(x) -> (state, clearance)``, a blocked value reading
    ``-inf``.  The two were the same code written twice before this; they are the same
    question asked of different numbers.

    Not a bisection, because clear is not monotonic in either parameter.  Backing a weld
    off frees the tip from the panel but can put the throat into tooling behind it, and
    opening the gun frees the throat but swings the moving electrode into whatever is
    beside it -- so in both cases there may be several clear windows with blocked ground
    between them, some narrower than any fixed scan.  So:

    1. scan the range every ``scan``;
    2. split every gap between two blocked samples in half, and again, until the spacing is
       at or below ``resolution`` -- carrying on after something is clear, since a better
       window may be narrower.  A window narrower than the final spacing can still fall
       between samples and be missed.  A gap with a clear end is not split: anything clear
       inside it runs on from that sample's window, which the climb below explores;
    3. in every clear window, climb from its best sample -- try half the scan spacing either
       side, keep whichever reads more clearance, halve, stop at the resolution.  That finds
       the best point near that sample, not necessarily a narrower peak elsewhere in a wide
       window.

    ``known`` pre-seeds values the caller has already measured, so a sweep bounded by a
    solve that has just failed does not pay for it twice.
    """
    samples: dict = dict(known or {})

    def sample(x: float) -> float:
        x = round(min(max(x, lo), hi), 9)
        if x not in samples:
            samples[x] = measure(x)
        return x

    def clear(x: float) -> bool:
        return samples[x][0] is not None

    def rank(x: float) -> tuple[float, float]:
        return samples[x][1], x

    cells = max(1, int(np.ceil((hi - lo) / scan - 1e-9)))
    step = (hi - lo) / cells
    for k in range(cells + 1):
        sample(lo + k * step)
    coarse = step

    while step > resolution + 1e-9:
        grid = sorted(samples)
        step /= 2.0
        gaps = [(a + b) / 2.0 for a, b in zip(grid, grid[1:])
                if not clear(a) and not clear(b)]
        if not gaps:
            break
        for x in gaps:
            sample(x)

    windows: list[list[float]] = []
    run: list[float] = []
    for x in sorted(samples):
        if clear(x):
            run.append(x)
        elif run:
            windows.append(run)
            run = []
    if run:
        windows.append(run)
    if not windows:
        return _Scan(None, samples, [], step)

    for run in windows:
        x = max(run, key=rank)
        h = coarse / 2.0
        while True:
            x = max((x, sample(x - h), sample(x + h)), key=rank)
            if h <= resolution + 1e-9:
                break
            h /= 2.0
    return _Scan(max((x for x in samples if clear(x)), key=rank), samples, windows, step)


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
                 near_panel_mm: float = 0.0, linear_speed_mm_s: float = 0.0,
                 linear_crossing_penalty_s: float = 0.0,
                 linear_introduce_mm: float = 0.0,
                 direct_clearance_mm: float = 0.0,
                 weld_clearance_mm: float | None = None,
                 stand_off_search: bool = True, stand_off_scan_mm: float = 0.5,
                 stand_off_resolution_mm: float = 0.05,
                 opening_search: bool = True, opening_scan_mm: float = 10.0,
                 opening_resolution_mm: float = 1.0,
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
        # linear motion instead of joint motion.  The band is the whole of that decision:
        # it is asked of each move, and no leg has to qualify before its moves may be
        # linear.  The clearance query has to see past the band, not just to it: a reading
        # at the probe means nothing was found, so a probe equal to the band reads every
        # state outside it as sitting on its edge.
        self.zone = LinearZone(near_mm=near_panel_mm,
                               linear_speed_mm_s=linear_speed_mm_s,
                               crossing_penalty_s=linear_crossing_penalty_s,
                               introduce_mm=linear_introduce_mm)
        # How much air the straight move between two waypoints has to keep before it is
        # taken in preference to searching.  A collision check answers whether the move
        # touches anything, not whether it is a move anybody would choose, and the two
        # part company on a transit that slides along a panel: clear at the margin, and
        # the one shape of route the searches would never have proposed.  0 leaves the
        # collision check as the whole of the test, which is what it was.
        self.direct_clearance_mm = max(0.0, float(direct_clearance_mm))
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
                     report_probe,
                     # Read by the same at-or-under test, and for the same reason it has
                     # to be seen past: a floor at the probe is one every state meets by
                     # the query running out of range.
                     (self.direct_clearance_mm + NEAR_PANEL_HEADROOM_MM
                      if self.direct_clearance_mm > 0.0 else 0.0))
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
        # A weld that declares no gun opening, or whose declared one places nowhere at all,
        # has one searched for over the gun's travel; see _search_opening.
        if opening_scan_mm <= 0.0 or opening_resolution_mm <= 0.0:
            raise ValueError("the gun-opening scan spacing and resolution must be positive")
        self.opening_search = bool(opening_search)
        self.opening_scan_mm = float(opening_scan_mm)
        self.opening_resolution_mm = float(opening_resolution_mm)
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

        A weld's declared opening is process data: the gun has to be where the weld
        schedule says, so a wider one is not offered here as an alternative to it.  What
        happens when it does not place is ``_search_opening``, which runs only after this
        list is exhausted -- so the schedule is still what is tried first and what ships
        wherever it works.  An empty list is a weld that declares nothing at all, where
        there is no schedule to honour and the search is the primary path rather than a
        fallback.

        An ordinary via carries no such requirement -- it declares nothing and is planned
        closed by default -- so if the tip fouls something there, opening or closing it is
        a legitimate way through and is tried straight away.
        """
        if not self.cell.gun_joint_name:
            return [0.0]
        if loc.is_weld:
            return [] if loc.gun_opening_arrive is None else [float(loc.gun_opening_arrive)]
        widest = self.man.gun_opening_max
        # Deduped for the same reason a transit's list is: a gun with no travel to speak of
        # collapses all three of these onto zero, and solving the same pose three times
        # over means three identical failures, three diagnoses and three geometry exports
        # before the locator is given up on.
        return unique_openings([0.0, widest, widest / 2.0], widest)

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
        # Snapshot: an inner stand-off sweep replaces ``pose_world`` on success, and the
        # opening search has to start from the stand-off the study asked for rather than
        # from wherever a failed attempt left the pose.
        full_shift = np.array(loc.pose_world, dtype=float)
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
        # A weld that declares no opening, or whose declared one places nowhere at all,
        # is what _search_opening is for.  Reached only once the declared opening has been
        # tried at every stand-off, so the schedule is honoured wherever it can be.
        if (loc.is_weld and self.opening_search and self.cell.gun_joint_name
                and self.man.gun_opening_max > 0.0):
            loc.pose_world = np.array(full_shift, dtype=float)
            with self._clearance_for(loc):
                q, opening, note = self._search_opening(loc, seed)
                if q is not None:
                    self._report_clearance(loc, q, opening, placed=True)
            if q is not None:
                if self.cell.in_stop_band(q):
                    self.log(f"    ! '{loc.name}' has no solution outside the joint 5 "
                             f"stop band; placed at "
                             f"{np.rad2deg(q[STOP_BAND_JOINT]):.1f} deg")
                return q, opening
            problems.append(f"no gun opening reaches it{note}")

        detail = ("; ".join(problems)
                  or "it declares no gun opening and --weld-opening-search is off")
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

        The sweep is :func:`_scan_for_clear`, which the gun-opening search runs too.  Why it
        is a scan with gap splitting rather than a bisection is argued there, and holds for
        the same reason on both axes.

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

        def pose_at(d: float) -> np.ndarray:
            P = imported.copy()
            P[:3, 3] += offset * (d / span)
            return P

        def measure(d: float):
            q = self.cell.solve_pose(pose_at(d), [seed, self.start_q],
                                     avoid_stop_band=True)
            return q, (-np.inf if q is None else self.cell.clearance_mm(q))

        # The full shift has just failed, so it bounds the sweep without being solved again.
        found = _scan_for_clear(0.0, span, scan=self.stand_off_scan_mm,
                                resolution=self.stand_off_resolution_mm, measure=measure,
                                known={round(span, 9): (None, -np.inf)})
        if found.best is None:
            return None, (f"; no shorter stand-off is clear either ({found.tried - 1} tried "
                          f"between 0 and {sign * span:+g} mm, {found.spacing:.3g} mm apart)")
        self.log(f"    '{loc.name}' is blocked at the full {sign * span:+g} mm stand-off; "
                 f"clear at {found.describe(sign, plus=True)} mm ({found.tried - 1} tried), "
                 f"standing off {sign * found.best:+.2f} mm")
        loc.pose_world = pose_at(found.best)
        return found.samples[found.best][0], ""

    def _search_opening(self, loc: Locator, seed: np.ndarray
                        ) -> tuple[np.ndarray | None, float, str]:
        """The gun opening to reach this weld at, where the declared one will not do.

        Two cases arrive here and they are the same question.  A weld the study gave no
        opening for, where there is nothing to honour and something has to be chosen; and a
        weld whose declared opening places the robot nowhere at all, neither at the
        stand-off it asked for nor at any shorter one.  Both ask "which openings is this
        pose reachable at", and it is asked the way the stand-off asks its own question,
        through :func:`_scan_for_clear`.

        **The two axes are searched in order, not as one grid, and the order is the point.**
        A stand-off is a planning device: ``_restore_weld_poses`` puts the waypoint back on
        the imported pose before anything is written, so deviating along it costs nothing
        anyone downstream can see.  An opening is process data that ships.  So the declared
        opening at the declared stand-off is still tried first and still ships wherever it
        works -- which is exactly what happened before this existed -- and only once no
        opening works at the stand-off already settled on is the pose allowed to move as
        well.  Searched as one grid, an opening reading a millimetre more clearance could
        displace the study's own choice, which is not a trade anyone asked for.

        Only the arrival opening is searched.  The departure one describes the *transit out
        of* the weld rather than anything geometric here -- see ``_leave_opening`` -- and is
        settled by that transit's own search.

        Returns the state, the opening it was reached at, and a note for the failure message.
        """
        widest = float(self.man.gun_opening_max)
        declared = loc.gun_opening_arrive

        def measure(o: float):
            with self.cell.gun_opening(o):
                q = self.cell.solve_pose(loc.pose_world, [seed, self.start_q],
                                         avoid_stop_band=True)
                return q, (-np.inf if q is None else self.cell.clearance_mm(q))

        found = _scan_for_clear(0.0, widest, scan=self.opening_scan_mm,
                                resolution=self.opening_resolution_mm, measure=measure)
        if found.best is not None:
            why = ("declares no gun opening" if declared is None else
                   f"cannot be placed at the {declared:g} mm the study asks for")
            # Flagged where it overrides the study, plain where it fills in a blank.
            mark = "" if declared is None else "! "
            self.log(f"    {mark}'{loc.name}' {why}; clear at {found.describe()} mm "
                     f"({found.tried} tried), reaching it at {found.best:g} mm")
            return found.samples[found.best][0], found.best, ""

        if not (self.stand_off_search and loc.pose_world_import is not None):
            return None, 0.0, (f"; and no opening between 0 and {widest:g} mm places it "
                               f"either ({found.tried} tried)")

        # Both wrong at once -- the tip through the panel *and* the throat in tooling, so
        # neither axis frees it alone.  One inner sweep of the stand-off per opening, on the
        # coarse grid only: each sample here is already a whole sweep of the other axis
        # rather than a single solve, so splitting this axis too would multiply the two.
        self.log(f"    '{loc.name}' is blocked at every gun opening at this stand-off; "
                 f"searching the stand-off at each of them")
        keep = np.array(loc.pose_world, dtype=float)
        cells = max(1, int(np.ceil(widest / self.opening_scan_mm - 1e-9)))
        pairs: list[tuple[float, float, np.ndarray, np.ndarray]] = []
        for k in range(cells + 1):
            opening = min(widest, k * widest / cells)
            with self.cell.gun_opening(opening):
                q, _ = self._search_stand_off(loc, seed)
                if q is not None:
                    pairs.append((self.cell.clearance_mm(q), opening, q,
                                  np.array(loc.pose_world, dtype=float)))
            # Restored between trials: the inner sweep replaces the pose on success, and the
            # next opening has to be measured from the stand-off this weld started at.
            loc.pose_world = np.array(keep, dtype=float)
        if not pairs:
            return None, 0.0, (f"; and no combination of opening and stand-off places it "
                               f"either ({cells + 1} openings swept)")
        _, opening, q, pose = max(pairs, key=lambda t: (t[0], t[1]))
        loc.pose_world = pose
        self.log(f"    ! '{loc.name}' needed both: reaching it at {opening:g} mm with the "
                 f"pose moved off the full stand-off as well")
        return q, opening, ""

    def _leave_opening(self, loc: Locator) -> float:
        """Gun opening the study asks the robot to leave this locator holding.

        A preference rather than a commitment.  It seeds the transit's opening list, and
        what that transit is actually solved at is written back over the weld phase once
        it is -- see ``_plan_pair``.

        The two openings a weld declares describe the **transits either side of it**, not
        the squeeze: the squeeze is not modelled at all, the robot being stationary through
        the weld.  A study chains them, one weld's leave being the next weld's arrive
        (measured across Paths 2 and 3: every pair agrees), which is the same statement
        read from the two ends of one transit.  So this is not searched -- there is nothing
        geometric to search for.  What decides it is which opening that transit can be
        flown at, which is the transit search's question and not this one's.
        """
        if loc.is_weld:
            return 0.0 if loc.gun_opening_leave is None else float(loc.gun_opening_leave)
        placed = self.openings.get(loc.name)
        return 0.0 if placed is None else float(placed)

    def _arrive_opening(self, loc: Locator) -> float:
        """Gun opening the robot must be holding as it reaches this locator.

        Read from what the locator was actually placed at rather than from what it
        declared.  A weld whose declared opening placed nowhere, or that declared none,
        has had one found for it, and it is that one every transit arriving here has to be
        holding -- the declared value is a pose the robot was never shown to reach.
        """
        placed = self.openings.get(loc.name)
        if placed is not None:
            return float(placed)
        return 0.0 if loc.gun_opening_arrive is None else float(loc.gun_opening_arrive)

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

        # The weld phase, corrected once the transit out of it has been solved.
        weld_phase = phases[-1] if a.is_weld else None
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
            direct_clearance=self.direct_clearance_mm,
            openings=self._transit_openings(a, b, leave_open, arrive_open),
            main_openings=self.main_openings,
            extra_openings=self.extra_openings,
            opening_round_mm=self.opening_round_mm,
            record=raw_legs, log=self.log)
        # What the robot leaves a weld holding is what the transit out of it was solved
        # at, which is only known now.  The declared value seeded that search and is a
        # preference, not a commitment: writing it out unchanged would have the gun change
        # at the moment the robot starts moving, instead of while it stands still at the
        # weld -- the one place the change is deliberately not simulated, because there the
        # robot is stationary and the tip's own travel is clear by inspection.  The arriving
        # side needs nothing done to it: the transit into a weld carries its opening on its
        # own phases, and the next segment corrects that weld's phase the same way.  Held by
        # reference in the unrefined copy, so this lands on both at once.
        if weld_phase is not None and legs and legs[0][1] is not None:
            flown = float(legs[0][1])
            if abs(flown - weld_phase.gun_opening_mm) > 1e-9:
                self.log(f"    '{a.name}' leaves at {flown:g} mm rather than the "
                         f"{weld_phase.gun_opening_mm:g} mm declared, that being what the "
                         f"transit out of it was solved at")
            weld_phase.gun_opening_mm = flown

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
