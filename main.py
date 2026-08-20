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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="weldpath",
        description="Plan a FANUC weld path from a Tesseract study directory.")
    #DIRECTORY PATH
    p.add_argument("directory",
                   help="study directory containing manifest.json and meshes/")

    ###WELD HANDLING###
    #WELD PULL BACK AXIS
    p.add_argument("--approach-axis", default = "-z", choices=["+x", "-x", "+y", "-y", "+z", "-z"],
                   help="direction in a weld locator's own frame that the straight "
                        "lead-in and lead-out travel along (default: -z)")
    #DISABLE LINEAR TUNNEL
    p.add_argument("--no-linear-zone", dest="linear_zone", action="store_false",
                   help="do not lead into or out of a weld in a straight line. Every "
                        "millimetre of the manifest's linear_zone_mm has to be clear "
                        "along one fixed direction, which a weld set deep in panelling "
                        "may have no room for; without it the transit runs weld to weld "
                        "and is free to curve away immediately, at the cost of the gun "
                        "arriving on a curve rather than sliding on straight")
    #LINEAR TUNNEL STEP LENGTH
    p.add_argument("--linear-step-mm", type=float, default=50.0,
                   help="spacing of points along linear approach/depart moves "
                        "(default: 50)")

    ###COLLISIONS###
    #COLLISION JOINT STEP RESOLUTION
    p.add_argument("--check-step-deg", type=float, default=3.0,
                   help="joint-space resolution used when checking a move for collision; "
                        "smaller is safer and slower (default: 3)")
    #COLLISION CHECK RESOLUTION
    p.add_argument("--segment-length-rad", type=float, default=0.02,
                   help="collision checking resolution for the sampling planner "
                        "(default: 0.02)")

    ###PATHFINDING###
    #MAX OMPL (pathfinding) ATTEMPTS
    p.add_argument("--ompl-attempts", type=int, default=20,
                   help="most freespace planning attempts per segment, used to retry a "
                        "transit that keeps failing (default: 3)")
    #MIN OMPL ATTEMPTS
    p.add_argument("--ompl-min-runs", type=int, default=7, metavar="N",
                   help="always run the sampling planner at least this many times per "
                        "transit and keep the lowest-penalty solution, rather than taking "
                        "the first one that works. The planner returns whichever route it "
                        "stumbles on first, and no later pass can move a route to the "
                        "other side of an obstacle, so this is the only stage that can "
                        "choose between them. Costs a full solve per run (default: 3)")
    #OMPL ATTEMPT MAX TIME
    p.add_argument("--ompl-seconds", type=float, default=7.0, metavar="SECONDS",
                   help="how long one sampling-planner run may search before giving up. Raise it for a transit that keeps failing, where restarting the search wastes the tree built so far; every run costs this long in the worst case, so it multiplies with --ompl-min-runs (default: 5)")
    #DISABLE SHORTCUT PASS
    p.add_argument("--no-shortcut", dest="shortcut", action="store_false",
                   help="skip the shortcutting pass and emit the sampling planner's own "
                        "route, which is typically much longer")
    #SHORTCUT PASS TIME
    p.add_argument("--shortcut-seconds", type=float, default=20.0, metavar="SECONDS",
                   help="time budget for shortcutting each freespace transit; longer "
                        "budgets keep shortening with diminishing returns (default: 30)")
    #POLISH PASS TIME
    p.add_argument("--polish-seconds", type=float, default=20.0, metavar="SECONDS",
                   help="time budget for the final pass over each transit's emitted "
                        "waypoints, which removes and relocates them under the full "
                        "stop-to-stop time the robot really pays for each one. The "
                        "earlier passes work on a densified path where a waypoint is a "
                        "sampling artefact rather than a stop; this one does not "
                        "(default: 5)")
    p.add_argument("--no-near-panel-linear", dest="near_panel_linear",
                   action="store_false",
                   help="plan every transit as joint motion throughout, rather than "
                        "re-planning the stretches that run close to the parts as straight "
                        "moves")
    p.add_argument("--near-panel-mm", type=float, default=100.0, metavar="MM",
                   help="clearance from a panel or from tooling at or under which a "
                        "transit counts as working near the parts, and is re-planned as "
                        "linear motion. Larger means more of the route comes out linear, "
                        "which is more predictable and slower to execute (default: 100)")
    p.add_argument("--near-panel-min-mm", type=float, default=150.0, metavar="MM",
                   help="shortest near-panel stretch worth converting, measured as tool "
                        "travel. A sweeping transit that clips the proximity band for a "
                        "moment is not working near the panel, and cutting it in three to "
                        "say so costs a stop at each end for nothing (default: 150)")
    p.add_argument("--min-shell-mm", type=float, default=5.0,
                   help="drop collision shells smaller than this across their bounding "
                        "box diagonal (default: 5)")
    p.add_argument("--max-shells", type=int, default=1500,
                   help="keep at most this many convex shells per link; fewer is faster "
                        "but coarser (default: 500)")
    p.add_argument("--hull-cell-mm", type=float, default=25.0, metavar="MM",
                   help="split shells that a single convex hull fits badly into cells of "
                        "roughly this size, hulling each one, so the collision geometry "
                        "follows recesses instead of bridging them. Smaller is more "
                        "accurate and slower; 0 disables (default: 0)")
    p.add_argument("--hull-fill", type=float, default=0.75, metavar="F",
                   help="how much of its own bounding box a shell must fill before a "
                        "single hull is accepted for it; below this it is split by "
                        "--hull-cell-mm. Raise it towards 1 for geometry whose recesses "
                        "matter, such as panelling full of shallow bowls that a hull "
                        "would skin over (default: 0.75)")
    p.add_argument("--robot-cell-mm", type=float, default=0, metavar="MM",
                   help="--hull-cell-mm for the arm's own links. The arm never comes close "
                        "enough to the parts for hull error to decide anything, so this is "
                        "the first thing to switch off (default: --hull-cell-mm)")
    p.add_argument("--gun-cell-mm", type=float, default=0, metavar="MM",
                   help="--hull-cell-mm for the gun body and moving tip. The gun is convex "
                        "where it makes contact, so refining it buys accuracy nowhere and "
                        "costs shells everywhere (default: --hull-cell-mm)")
    p.add_argument("--tooling-cell-mm", type=float, default=25, metavar="MM",
                   help="--hull-cell-mm for static objects the manifest calls tooling. "
                        "These are the largest meshes in the cell and refinement runs "
                        "after --max-shells, so a small cell here dominates both "
                        "preparation and every later collision check "
                        "(default: --hull-cell-mm)")
    p.add_argument("--panel-cell-mm", type=float, default=25, metavar="MM",
                   help="--hull-cell-mm for static objects the manifest calls panel. This "
                        "is the geometry that is concave exactly where the welds are, so "
                        "it is where a small cell is worth paying for "
                        "(default: --hull-cell-mm)")
    p.add_argument("--obstacle-clearance-mm", type=float, default=0.0, metavar="MM",
                   help="how close the robot and gun may come to the panels and tooling "
                        "before it counts as a collision. Positive keeps that much clear "
                        "air, 0 means touching collides, negative tolerates that much "
                        "overlap (default: 0)")
    p.add_argument("--weld-clearance-mm", type=float, default=-12.0, metavar="MM",
                   help="obstacle clearance used instead of --obstacle-clearance-mm on "
                        "any move starting or ending at a weld locator, where the gun "
                        "has to reach the panel (default: 2)")
    p.add_argument("--weld-shift-mm", type=float, default=-5.0, metavar="MM",
                   help="move every weld locator this far along its own z axis before "
                        "planning, backing the tool off a pose authored on the panel "
                        "surface. Negative retreats along -z (default: -5)")
    p.add_argument("--no-clearance-penalty", dest="clearance_penalty",
                   action="store_false",
                   help="do not penalise routes that run close to the panels and "
                        "tooling; only hard collisions are avoided")
    p.add_argument("--clearance-penalty-max-mm", type=float, default=300.0, metavar="MM",
                   help="clearance at and above which there is no penalty; the curve "
                        "starts here (default: 300)")
    p.add_argument("--clearance-penalty-min-mm", type=float, default=10.0, metavar="MM",
                   help="clearance at and below which the penalty is at its peak "
                        "(default: 10)")
    p.add_argument("--clearance-penalty-multiplier", type=float, default=5.0,
                   metavar="N",
                   help="peak penalty: a second spent at the minimum clearance costs as "
                        "much as N seconds in open space (default: 5)")
    p.add_argument("--clearance-penalty-cutoff-mm", type=float, default=0.0, metavar="MM",
                   help="ignore clearances beyond this, so the proximity query looks no "
                        "further and planning runs faster. Truncates the shallow end of "
                        "the curve without reshaping the rest, so the penalty steps "
                        "abruptly at this distance; 0 uses the maximum (default: 0)")
    p.add_argument("--joint-max-velocity", metavar="V1,V2,...",
                   help="per-joint velocity limits in rad/s, comma separated in joint "
                        "order; a single value applies to every joint "
                        "(default: 2pi/3 for j1-j5, 11pi/9 for j6)")
    p.add_argument("--joint-max-acceleration", metavar="A1,A2,...",
                   help="per-joint acceleration limits in rad/s^2, comma separated in "
                        "joint order; a single value applies to every joint "
                        "(default: 2.5 for j1-j5, 11 for j6)")
    p.add_argument("--linear-speed-mm-s", type=float, default=250.0,
                   help="commanded tool speed cap on linear moves; the joint limits still "
                        "govern whenever they are slower (default: 250)")
    p.add_argument("--unrefined-output", action="store_true",
                   help="also write waypoints-unrefined.json, holding each transit as the "
                        "sampling planner returned it, before shortcutting and waypoint "
                        "reduction. Same schema as waypoints.json, for comparing what the "
                        "optimisation passes actually changed (default: off)")
    p.add_argument("--export-collision-geometry", action="store_true",
                   help="write the convex geometry the planner actually collides against "
                        "to <directory>/collision_geometry, one OBJ per link with each "
                        "convex piece as its own o group, in world coordinates and the "
                        "manifest's units so it overlays the source meshes. The "
                        "convex_cache files are the input to the decomposition and still "
                        "hold the original concave triangles, so they cannot show where a "
                        "hull bridges a recess; these can (default: off)")
    p.add_argument("--quiet", action="store_true", help="only print the final summary")
    return p


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
    from weldpath.penalty import ClearancePenalty
    from weldpath import profile as profile_mod
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
        penalty = ClearancePenalty(
            max_mm=args.clearance_penalty_max_mm,
            min_mm=args.clearance_penalty_min_mm,
            multiplier=args.clearance_penalty_multiplier,
            cutoff_mm=args.clearance_penalty_cutoff_mm,
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
                              obstacle_clearance_mm=args.obstacle_clearance_mm,
                              penalty=penalty, dynamics=dynamics)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.export_collision_geometry:
        from weldpath import hullexport
        hullexport.export(cell.env, man, directory, log=print)

    log("planning:")
    planner = ToolpathPlanner(
        cell, man,
        approach_axis=args.approach_axis,
        linear_step_mm=args.linear_step_mm,
        ompl_attempts=args.ompl_attempts,
        ompl_runs=args.ompl_min_runs,
        ompl_seconds=args.ompl_seconds,
        linear_zone=args.linear_zone,
        segment_length=args.segment_length_rad,
        check_step_deg=args.check_step_deg,
        shortcut_seconds=args.shortcut_seconds if args.shortcut else 0.0,
        polish_seconds=args.polish_seconds if args.shortcut else 0.0,
        near_panel_mm=args.near_panel_mm if args.near_panel_linear else 0.0,
        near_panel_min_mm=args.near_panel_min_mm,
        weld_clearance_mm=args.weld_clearance_mm,
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
