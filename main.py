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
    p.add_argument("directory",
                   help="study directory containing manifest.json and meshes/")
    p.add_argument("--approach-axis", choices=["+x", "-x", "+y", "-y", "+z", "-z"],
                   help="TCP-frame direction to retract along when leaving a weld "
                        "(default: derived from the gun's stroke axis)")
    p.add_argument("--linear-step-mm", type=float, default=50.0,
                   help="spacing of points along linear approach/depart moves "
                        "(default: 50)")
    p.add_argument("--check-step-deg", type=float, default=3.0,
                   help="joint-space resolution used when checking a move for collision; "
                        "smaller is safer and slower (default: 3)")
    p.add_argument("--segment-length-rad", type=float, default=0.02,
                   help="collision checking resolution for the sampling planner "
                        "(default: 0.02)")
    p.add_argument("--ompl-attempts", type=int, default=3,
                   help="freespace planning attempts per segment (default: 3)")
    p.add_argument("--no-shortcut", dest="shortcut", action="store_false",
                   help="skip the shortcutting pass and emit the sampling planner's own "
                        "route, which is typically much longer")
    p.add_argument("--shortcut-seconds", type=float, default=3.0, metavar="SECONDS",
                   help="time budget for shortcutting each freespace transit; longer "
                        "budgets keep shortening with diminishing returns (default: 3)")
    p.add_argument("--min-shell-mm", type=float, default=40.0,
                   help="drop collision shells smaller than this across their bounding "
                        "box diagonal (default: 40)")
    p.add_argument("--max-shells", type=int, default=80,
                   help="keep at most this many convex shells per link; fewer is faster "
                        "but coarser (default: 80)")
    p.add_argument("--joint-speed-deg-s", type=float, default=180.0,
                   help="peak joint speed used to fill in waypoint times (default: 180)")
    p.add_argument("--linear-speed-mm-s", type=float, default=250.0,
                   help="peak tool speed on linear moves, used to fill in waypoint "
                        "times (default: 250)")
    p.add_argument("--accel-blend", type=float, default=0.5, metavar="0..1",
                   help="share of each move spent accelerating or decelerating: 0 is "
                        "flat velocity, 1 is bang-bang (accelerate then decelerate, no "
                        "cruise). Shapes the trapezoidal velocity profile behind the "
                        "waypoint times (default: 0.5)")
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
    from weldpath.profile import MotionProfile
    from weldpath.toolpath import ToolpathPlanner

    t0 = time.time()
    try:
        man = manifest_mod.load(directory)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    log(f"loaded manifest: {len(man.locators)} locators, "
        f"{len(man.static_objects)} static objects, units={man.units}")
    if not man.locators:
        print("error: manifest defines no locators, nothing to plan", file=sys.stderr)
        return 1

    try:
        cell = cell_mod.build(man, log=log, min_shell_mm=args.min_shell_mm,
                              max_shells=args.max_shells)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    log("planning:")
    planner = ToolpathPlanner(
        cell, man,
        approach_axis=args.approach_axis,
        linear_step_mm=args.linear_step_mm,
        ompl_attempts=args.ompl_attempts,
        segment_length=args.segment_length_rad,
        check_step_deg=args.check_step_deg,
        shortcut_seconds=args.shortcut_seconds if args.shortcut else 0.0,
        log=log)
    segments = planner.run()

    try:
        profile = MotionProfile(args.accel_blend)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    log(f"velocity profile: {profile.blend:g} blend "
        f"({100 * profile.ramp:.0f}% of each move ramping up, same again down, "
        f"mean speed {100 * profile.duty:.0f}% of peak)")

    timing = output_mod.Timing(args.joint_speed_deg_s, args.linear_speed_mm_s, profile)
    document = output_mod.build_document(cell, man, segments, timing)

    for problem in output_mod.check_endpoints(document, man):
        print(f"warning: {problem}", file=sys.stderr)

    path = output_mod.write(directory, document)
    print(output_mod.summarise(document, segments))
    print(f"wrote {path} in {time.time() - t0:.1f}s")
    return 0 if all(not s.error for s in segments) else 1


if __name__ == "__main__":
    raise SystemExit(main())
