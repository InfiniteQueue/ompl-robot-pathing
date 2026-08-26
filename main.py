"""Generate a FANUC weld path from a study directory.

    python main.py <directory>

Reads ``<directory>/manifest.json`` (with meshes under ``<directory>/meshes``) and writes
``<directory>/waypoints.json``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time


TRUE_WORDS = ("true", "yes", "on", "1")
FALSE_WORDS = ("false", "no", "off", "0")


def boolean(text: str) -> bool:
    """Parse a spelled-out boolean argument value.

    Written out rather than left as a bare flag where the setting selects between two
    models that both exist: "--stepped-penalty false" says which curve is in force, where
    the absence of a flag only says which one is not.
    """
    word = str(text).strip().lower()
    if word in TRUE_WORDS:
        return True
    if word in FALSE_WORDS:
        return False
    raise argparse.ArgumentTypeError(
        f"expected one of {', '.join(TRUE_WORDS + FALSE_WORDS)}, got {text!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="weldpath",
        description="Plan a FANUC weld path from a Tesseract study directory.")
    #DIRECTORY PATH
    p.add_argument("directory",
                   help="study directory containing manifest.json and meshes/")

    #region ###WELD HANDLING###
    #LINEAR TUNNEL STEP LENGTH
    p.add_argument("--linear-step-mm", type=float, default=50.0,
                   help="spacing of points along linear approach/depart moves "
                        "(default: 50)")
#endregion
    #region ###COLLISIONS###
    #COLLISION JOINT STEP RESOLUTION
    p.add_argument("--check-step-deg", type=float, default=3.0,
                   help="joint-space resolution used when checking a move for collision; "
                        "smaller is safer and slower (default: 3)")
    #COLLISION STEP RESOLUTION AT THE TOOL
    p.add_argument("--check-step-mm", type=float, default=10.0, metavar="MM",
                   help="tool-space companion to --check-step-deg, applied as well as it "
                        "rather than instead of it. A joint step means different distances "
                        "at different poses -- a few degrees is millimetres at the wrist "
                        "and a hand's breadth at the base -- so an interval that carries "
                        "the gun further than this is split and rechecked. Never checks "
                        "less than --check-step-deg alone would, and costs nothing where "
                        "the two agree. 0 turns this off (default: 10)")
    #COLLISION CHECK RESOLUTION
    p.add_argument("--segment-length-rad", type=float, default=0.02,
                   help="collision checking resolution for the sampling planner "
                        "(default: 0.02)")
#endregion
    #region ###PATHFINDING###
    #PHASE ONE: CHOOSE BETWEEN ROUTES
    p.add_argument("--phase-one-runs", type=int, default=7, metavar="N",
                   help="sampling-planner runs made per transit in phase one. Every one "
                        "is spent whether or not earlier runs succeeded, and the "
                        "lowest-penalty solution of the set is the one that ships. The "
                        "planner returns whichever route it stumbles on first, and no "
                        "later pass can move a route to the other side of an obstacle, so "
                        "this is the only stage that can choose between them "
                        "(default: 7)")
    #PHASE ONE RUN TIME
    p.add_argument("--phase-one-solve-seconds", type=float, default=8.0,
                   metavar="SECONDS",
                   help="how long one phase-one run may search before giving up. Every "
                        "run costs this long in the worst case, so it multiplies with "
                        "--phase-one-runs (default: 8)")
    #PHASE TWO: FIND ANY ROUTE AT ALL
    p.add_argument("--phase-two-max-runs", type=int, default=8, metavar="N",
                   help="most sampling-planner runs allowed in phase two, which is "
                        "entered only when phase one found nothing at all. Phase two "
                        "stops at the first solution rather than sampling for a better "
                        "one; if it too comes back empty the transit is retried through "
                        "the fallback poses, starting again from phase one (default: 8)")
    #PHASE TWO RUN TIME
    p.add_argument("--phase-two-solve-seconds", type=float, default=10.0,
                   metavar="SECONDS",
                   help="how long one phase-two run may search. Worth setting higher than "
                        "--phase-one-solve-seconds: a transit that beat phase one is "
                        "usually one where restarting wastes the tree built so far, so a "
                        "longer single search helps where another short one does not "
                        "(default: 10)")
    #endregion
    #region ###OPTIMISATION###

    #DISABLE SHORTCUT PASS
    p.add_argument("--no-shortcut", dest="shortcut", action="store_false",
                   help="skip the shortcutting pass and emit the sampling planner's own "
                        "route, which is typically much longer")
    #SHORTCUT PASS TIME
    p.add_argument("--shortcut-seconds", type=float, default=20.0, metavar="SECONDS",
                   help="time budget for shortcutting each freespace transit; longer "
                        "budgets keep shortening with diminishing returns (default: 20)")
    #POLISH PASS TIME
    p.add_argument("--polish-seconds", type=float, default=20.0, metavar="SECONDS",
                   help="time budget for the final pass over each transit's emitted "
                        "waypoints, which removes and relocates them under the full "
                        "stop-to-stop time the robot really pays for each one. The "
                        "earlier passes work on a densified path where a waypoint is a "
                        "sampling artefact rather than a stop; this one does not "
                        "(default: 20)")
    #endregion
    #region ###LINEAR MOTION NEAR THE PARTS###
    #DISABLE LINEAR MOTION NEAR THE PARTS
    p.add_argument("--no-near-panel-linear", dest="near_panel_linear",
                   action="store_false",
                   help="plan every transit as joint motion throughout, rather than "
                        "re-planning the stretches that run close to the parts as straight "
                        "moves")
    #WHAT COUNTS AS NEAR A PART
    p.add_argument("--near-panel-mm", type=float, default=50.0, metavar="MM",
                   help="clearance from a panel or from tooling at or under which a "
                        "transit counts as working near the parts, and is re-planned as "
                        "linear motion. Larger means more of the route comes out linear, "
                        "which is more predictable and slower to execute (default: 120)")
    #SHORTEST STRETCH WORTH MAKING LINEAR
    p.add_argument("--near-panel-min-mm", type=float, default=100.0, metavar="MM",
                   help="shortest near-panel stretch worth converting, measured as tool "
                        "travel. A sweeping transit that clips the proximity band for a "
                        "moment is not working near the panel, and cutting it in three to "
                        "say so costs a stop at each end for nothing. A stretch under "
                        "this length still qualifies on --near-panel-min-pct "
                        "(default: 100)")
    #...OR THIS MUCH OF THE MOVE, HOWEVER SHORT
    p.add_argument("--near-panel-min-pct", type=float, default=60.0, metavar="PCT",
                   help="if this much of a move is within --near-panel-mm of the parts, "
                        "every near-panel stretch of it is made linear however short each "
                        "one is. Measured over the whole move, not over each stretch: time "
                        "spent near the panel does not have to be continuous, so a retract "
                        "whose apex leaves the band for an instant no longer disqualifies "
                        "the move either side of it. Without this no short move could come "
                        "out linear however completely it runs alongside the panel -- a hop "
                        "from one weld to the next being the case that matters. 0 leaves "
                        "--near-panel-min-mm as the only test (default: 50)")
    #endregion
    #region ###COLLISION HULLS###
    #DROP TINY SHELLS
    p.add_argument("--min-shell-mm", type=float, default=20.0,
                   help="drop collision shells smaller than this across their bounding "
                        "box diagonal (default: 20)")
    #SHELL COUNT CAP PER LINK
    p.add_argument("--max-shells", type=int, default=1500,
                   help="keep at most this many convex shells per link; fewer is faster "
                        "but coarser (default: 1500)")
    #HULL REFINEMENT CELL SIZE
    p.add_argument("--hull-cell-mm", type=float, default=50.0, metavar="MM",
                   help="split shells that a single convex hull fits badly into cells of "
                        "roughly this size, hulling each one, so the collision geometry "
                        "follows recesses instead of bridging them. Smaller is more "
                        "accurate and slower; 0 disables (default: 50)")
    #WHEN A SHELL NEEDS REFINING AT ALL
    p.add_argument("--hull-fill", type=float, default=0.85, metavar="F",
                   help="how much of its own bounding box a shell must fill before a "
                        "single hull is accepted for it; below this it is split by "
                        "--hull-cell-mm. Raise it towards 1 for geometry whose recesses "
                        "matter, such as panelling full of shallow bowls that a hull "
                        "would skin over (default: 0.85)")
    #CELL SIZE: ARM
    p.add_argument("--robot-cell-mm", type=float, default=0, metavar="MM",
                   help="--hull-cell-mm for the arm's own links. The arm never comes close "
                        "enough to the parts for hull error to decide anything, so this is "
                        "the first thing to switch off (default: 0, i.e. off)")
    #CELL SIZE: GUN
    p.add_argument("--gun-cell-mm", type=float, default=100, metavar="MM",
                   help="--hull-cell-mm for the gun body and moving tip. The gun is convex "
                        "where it makes contact, so refining it buys accuracy nowhere and "
                        "costs shells everywhere (default: 100)")
    #CELL SIZE: TOOLING
    p.add_argument("--tooling-cell-mm", type=float, default=30, metavar="MM",
                   help="--hull-cell-mm for static objects the manifest calls tooling. "
                        "These are the largest meshes in the cell and refinement runs "
                        "after --max-shells, so a small cell here dominates both "
                        "preparation and every later collision check (default: 30)")
    #CELL SIZE: PANELS
    p.add_argument("--panel-cell-mm", type=float, default=40, metavar="MM",
                   help="--hull-cell-mm for static objects the manifest calls panel. This "
                        "is the geometry that is concave exactly where the welds are, so "
                        "it is where a small cell is worth paying for (default: 40)")
    #WELD PROXIMITY FOR SHELL SPLIT
    p.add_argument("--shell-split-weld-prox", type=float, default=5.0, metavar="MM",
                   help="only refine panel and tooling geometry within this many mm of "
                        "the gun, as the gun sits when it is at a weld; beyond it the cell "
                        "is scaled by --far-cell-factor. Measured from the gun's own "
                        "surface, not from the weld point, so the C-frame's throat and "
                        "back are covered rather than a sphere around the electrodes. "
                        "Refinement runs after --max-shells, so confining it is the "
                        "cheapest way to cut shell count without losing accuracy where it "
                        "decides anything. Allow for the approach as well as the weld "
                        "itself. 0 refines everywhere (default: 5)")
    #TCP PROXIMITY FOR SHELL SPLIT
    p.add_argument("--shell-split-tcp-prox", type=float, default=20.0, metavar="MM",
                   help="only refine the gun body and moving tip within this many mm of "
                        "the tool centre point. The gun and the TCP are rigid with respect "
                        "to each other, so unlike a weld this stays meaningful wherever the "
                        "arm carries them. Allow for the electrode stroke as well as the "
                        "approach, since the tip travels relative to the body. Needs "
                        "--gun-cell-mm above 0 to do anything at all. 0 refines everywhere "
                        "(default: 20)")
    #HOW COARSE THE GEOMETRY AWAY FROM THE WELDS AND THE TCP GETS
    p.add_argument("--far-cell-factor", type=float, default=6.0, metavar="N",
                   help="multiplier on the cell size beyond --shell-split-weld-prox and "
                        "--shell-split-tcp-prox; 0 "
                        "makes each far shell a single hull, which is coarser still and "
                        "can "
                        "bridge back across the welds (default: 6)")
    #HOW FAR A CELL REACHES PAST ITS OWN BOUNDS
    p.add_argument("--hull-cell-overlap", type=float, default=0.02, metavar="F",
                   help="how far past its own bounds a cell claims triangles, as a "
                        "fraction of the cell. A cell's hull ends up spanning about "
                        "1 + 2F cells, so every increment inflates every hull. It is only "
                        "a numerical margin -- at 0 a triangle still lands in one cell "
                        "and that cell's hull contains it, so the surface stays covered "
                        "-- and it exists so hulls with exactly coplanar faces are not "
                        "decided apart by round-off (default: 0.02)")
    #endregion
    #region ###CLEARANCE FROM THE PARTS###
    #CLEARANCE EVERYWHERE
    p.add_argument("--obstacle-clearance-mm", type=float, default=5.0, metavar="MM",
                   help="how close the robot and gun may come to the panels and tooling "
                        "before it counts as a collision. Positive keeps that much clear "
                        "air, 0 means touching collides, negative tolerates that much "
                        "overlap (default: 5)")
    #CLEARANCE ON A MOVE TO OR FROM A WELD
    p.add_argument("--weld-clearance-mm", type=float, default=-3.0, metavar="MM",
                   help="obstacle clearance used instead of --obstacle-clearance-mm on "
                        "any move starting or ending at a weld locator, where the gun "
                        "has to reach the panel (default: -3)")
    #BACK THE WELD OFF THE PANEL SURFACE
    p.add_argument("--weld-shift-mm", type=float, default=-5.0, metavar="MM",
                   help="move every weld locator this far along its own z axis before "
                        "planning, backing the tool off a pose authored on the panel "
                        "surface. Negative retreats along -z (default: -5)")
    #endregion
    #region ###CLEARANCE PENALTY###
    #DISABLE CLEARANCE PENALTY
    p.add_argument("--no-clearance-penalty", dest="clearance_penalty",
                   action="store_false",
                   help="do not penalise routes that run close to the panels and "
                        "tooling; only hard collisions are avoided")
    #PENALTY CURVE START
    p.add_argument("--clearance-penalty-max-mm", type=float, default=200.0, metavar="MM",
                   help="clearance at and above which there is no penalty; the curve "
                        "starts here (default: 200)")
    #PENALTY CURVE PEAK
    p.add_argument("--clearance-penalty-min-mm", type=float, default=-5.0, metavar="MM",
                   help="clearance at and below which the penalty is at its peak "
                        "(default: -5)")
    #PENALTY PEAK STRENGTH
    p.add_argument("--clearance-penalty-multiplier", type=float, default=8.0,
                   metavar="N",
                   help="peak penalty: a second spent at the minimum clearance costs as "
                        "much as N seconds in open space (default: 8)")
    #PENALTY CURVE SHAPE
    p.add_argument("--clearance-penalty-exponent", type=float, default=5.0, metavar="Y",
                   help="shape of the climb between --clearance-penalty-max-mm and "
                        "--clearance-penalty-min-mm, as x^Y on the span between them. "
                        "Both ends are fixed whatever this is, so the peak penalty does "
                        "not move; Y only decides where along the span the curve can tell "
                        "one clearance from another. Above 1 spends that resolution near "
                        "the minimum, which is what a wide maximum needs if very close is "
                        "not to cost about the same as close; below 1 spends it at the "
                        "open end instead. Must be above 0 (default: 5)")
    #PENALTY QUERY RANGE
    p.add_argument("--clearance-penalty-cutoff-mm", type=float, default=0.0, metavar="MM",
                   help="ignore clearances beyond this, so the proximity query looks no "
                        "further and planning runs faster. Truncates the shallow end of "
                        "the curve without reshaping the rest, so the penalty steps "
                        "abruptly at this distance; 0 uses the maximum (default: 0)")
    #endregion
    #region ###STEPPED CLEARANCE PENALTY###
    #USE THE STEPPED CURVE INSTEAD
    p.add_argument("--stepped-penalty", type=boolean, nargs="?", const=True,
                   default=False, metavar="BOOL",
                   help="true uses the stepped exponential penalty curve instead of the "
                        "power curve above. Every --clearance-penalty-* option is ignored "
                        "when it is on, apart from --no-clearance-penalty, which still "
                        "turns all penalties off. Takes true/false (yes/no, on/off, 1/0); "
                        "passing the flag with no value means true (default: false)")
    #STRENGTH AT TOUCHING
    p.add_argument("--stepped-penalty-multiplier", type=float, default=8.0, metavar="N",
                   help="penalty at zero clearance: a second spent touching costs as much "
                        "as N seconds in open space (default: 8)")
    #WHERE THE PENALTY IS SPENT
    p.add_argument("--stepped-penalty-zero-mm", type=float, default=200.0, metavar="MM",
                   help="clearance at which the penalty reaches 1x and stops mattering. "
                        "Also how far the proximity query has to see, so lowering it "
                        "speeds planning up (default: 200)")
    #STEP LENGTH
    p.add_argument("--stepped-penalty-step-mm", type=float, default=25.0, metavar="MM",
                   help="how much extra clearance counts as one step of falloff "
                        "(default: 25)")
    #STEP FACTOR
    p.add_argument("--stepped-penalty-step-factor", type=float, default=0.7, metavar="F",
                   help="what one step multiplies the penalty by: 0.7 means each "
                        "--stepped-penalty-step-mm of extra clearance costs 70%% of what "
                        "the step before it did. Must be between 0 and 1; smaller falls "
                        "off faster and so concentrates the penalty near the part. Holds "
                        "of the curve's exponential term rather than of the multiplier "
                        "exactly, since a quantity scaled by a constant factor per step "
                        "never reaches zero (default: 0.7)")
    #endregion
    #region ###VELOCITY PROFILE###
    #JOINT VELOCITY LIMITS
    p.add_argument("--joint-max-velocity", metavar="V1,V2,...",
                   help="per-joint velocity limits in rad/s, comma separated in joint "
                        "order; a single value applies to every joint "
                        "(default: 2pi/3 for j1-j5, 11pi/9 for j6)")
    #JOINT ACCELERATION LIMITS
    p.add_argument("--joint-max-acceleration", metavar="A1,A2,...",
                   help="per-joint acceleration limits in rad/s^2, comma separated in "
                        "joint order; a single value applies to every joint "
                        "(default: 2.5 for j1-j5, 11 for j6)")
    #TOOL SPEED CAP ON LINEAR MOVES
    p.add_argument("--linear-speed-mm-s", type=float, default=250.0,
                   help="commanded tool speed cap on linear moves; the joint limits still "
                        "govern whenever they are slower (default: 250)")
    #endregion
    #region ###OUTPUT AND DEBUGGING###
    #WRITE THE PRE-OPTIMISATION PATH TOO
    p.add_argument("--unrefined-output", action="store_true",
                   help="also write waypoints-unrefined.json, holding each transit as the "
                        "sampling planner returned it, before shortcutting and waypoint "
                        "reduction. Same schema as waypoints.json, for comparing what the "
                        "optimisation passes actually changed (default: off)")
    #ASK WHAT OCCUPIES A PARTICULAR POINT
    p.add_argument("--probe-point", metavar="LOCATOR:X,Y,Z",
                   help="name the collision hulls containing a point and report how far "
                        "the nearest real material is, then stop. The point is given in a "
                        "locator's own frame -- 'weld_7:-39.8,,-82.1', with an empty axis "
                        "meaning zero -- or as a bare 'x,y,z' in world coordinates. Use it "
                        "when the planner reports something solid that the input model "
                        "says is empty: a hull containing the point with no material near "
                        "it is a hull bridging a void, and the decomposition settings that "
                        "produced it are printed alongside (default: off)")
    #WRITE THE HULLS THE PLANNER SEES
    p.add_argument("--export-collision-geometry", action="store_true",
                   help="write the convex geometry the planner actually collides against "
                        "to <directory>/collision_geometry, one OBJ per link with each "
                        "convex piece as its own o group, in world coordinates and the "
                        "manifest's units so it overlays the source meshes. The "
                        "convex_cache files are the input to the decomposition and still "
                        "hold the original concave triangles, so they cannot show where a "
                        "hull bridges a recess; these can. Also writes a blocked_*.obj "
                        "for every locator the robot cannot be placed at, holding just "
                        "the two links that blocked it, at the pose that was rejected "
                        "(default: off)")
    #SUPPRESS THE RUN LOG
    p.add_argument("--quiet", action="store_true", help="only print the final summary")
    return p
    #endregion

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    directory = os.path.abspath(args.directory)
    if not os.path.isdir(directory):
        print(f"error: not a directory: {directory}", file=sys.stderr)
        return 2

    log = (lambda *a, **k: None) if args.quiet else print

    # Imported here so that --help works without the Tesseract bindings present.
    from weldpath import cell as cell_mod
    from weldpath import manifest as manifest_mod
    from weldpath import output as output_mod
    from weldpath.penalty import ClearancePenalty, SteppedPenalty
    from weldpath import profile as profile_mod
    from weldpath.planning import OmplBudget
    from weldpath.toolpath import ToolpathPlanner

    t0 = time.time()
    try:
        man = manifest_mod.load(directory)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    log(f"loaded manifest: {len(man.locators)} locators, "
        f"{len(man.static_objects)} static objects, units={man.units}")

    moved = manifest_mod.shift_weld_locators(man, args.weld_shift_mm)
    if moved:
        log(f"shifted {moved} weld locators by {args.weld_shift_mm:+g} mm along their "
            f"own z axis")
    if not man.locators:
        print("error: manifest defines no locators, nothing to plan", file=sys.stderr)
        return 1

    n = len(man.robot_joint_names)
    try:
        dynamics = profile_mod.JointDynamics(
            profile_mod.parse_limits(args.joint_max_velocity, n,
                                     profile_mod.default_velocity(n),
                                     "--joint-max-velocity"),
            profile_mod.parse_limits(args.joint_max_acceleration, n,
                                     profile_mod.default_acceleration(n),
                                     "--joint-max-acceleration"))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        if args.stepped_penalty:
            penalty = SteppedPenalty(
                multiplier=args.stepped_penalty_multiplier,
                zero_mm=args.stepped_penalty_zero_mm,
                step_mm=args.stepped_penalty_step_mm,
                step_factor=args.stepped_penalty_step_factor,
                enabled=args.clearance_penalty)
        else:
            penalty = ClearancePenalty(
                max_mm=args.clearance_penalty_max_mm,
                min_mm=args.clearance_penalty_min_mm,
                multiplier=args.clearance_penalty_multiplier,
                cutoff_mm=args.clearance_penalty_cutoff_mm,
                exponent=args.clearance_penalty_exponent,
                enabled=args.clearance_penalty)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    log(penalty.describe())

    per_category = {"robot": args.robot_cell_mm, "gun": args.gun_cell_mm,
                    "tooling": args.tooling_cell_mm, "panel": args.panel_cell_mm}

    try:
        cell = cell_mod.build(man, log=log, min_shell_mm=args.min_shell_mm,
                              max_shells=args.max_shells,
                              hull_cell_mm=args.hull_cell_mm,
                              hull_fill=args.hull_fill,
                              hull_per_category=per_category,
                              weld_proximity_mm=args.shell_split_weld_prox,
                              tcp_proximity_mm=args.shell_split_tcp_prox,
                              far_cell_factor=args.far_cell_factor,
                              hull_overlap=args.hull_cell_overlap,
                              obstacle_clearance_mm=args.obstacle_clearance_mm,
                              tcp_check_mm=args.check_step_mm,
                              export_dir=(directory if args.export_collision_geometry
                                          else None),
                              penalty=penalty, dynamics=dynamics)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.probe_point:
        # Instead of planning: this is asked when a specific pose is already known to be
        # blocked, and the answer does not depend on the toolpath.  The export, if one was
        # asked for, has already happened inside build().
        from weldpath import probe
        probe.report(cell.env, man, args.probe_point, log=print)
        return 0

    log("planning:")
    planner = ToolpathPlanner(
        cell, man,
        linear_step_mm=args.linear_step_mm,
        ompl=OmplBudget(phase_one_runs=args.phase_one_runs,
                        phase_one_seconds=args.phase_one_solve_seconds,
                        phase_two_max_runs=args.phase_two_max_runs,
                        phase_two_seconds=args.phase_two_solve_seconds),
        segment_length=args.segment_length_rad,
        check_step_deg=args.check_step_deg,
        shortcut_seconds=args.shortcut_seconds if args.shortcut else 0.0,
        polish_seconds=args.polish_seconds if args.shortcut else 0.0,
        near_panel_mm=args.near_panel_mm if args.near_panel_linear else 0.0,
        near_panel_min_mm=args.near_panel_min_mm,
        near_panel_min_pct=args.near_panel_min_pct,
        linear_speed_mm_s=args.linear_speed_mm_s,
        weld_clearance_mm=args.weld_clearance_mm,
        export_dir=directory if args.export_collision_geometry else None,
        keep_unrefined=args.unrefined_output,
        log=log)
    segments = planner.run()

    timing = output_mod.Timing(dynamics, args.linear_speed_mm_s)
    document = output_mod.build_document(cell, man, segments, timing)

    for problem in output_mod.check_endpoints(document, man):
        print(f"warning: {problem}", file=sys.stderr)

    if args.unrefined_output:
        raw = output_mod.build_document(cell, man, segments, timing, unrefined=True)
        log(f"wrote {output_mod.write(directory, raw, output_mod.UNREFINED_NAME)}")

    path = output_mod.write(directory, document)
    print(output_mod.summarise(document, segments))
    print(f"wrote {path} in {time.time() - t0:.1f}s")
    return 0 if all(not s.error for s in segments) else 1


if __name__ == "__main__":
    raise SystemExit(main())
