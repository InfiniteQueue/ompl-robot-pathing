"""Waypoints of each planned route before refinement and after each named stage.

A debugging record for the refinement passes, written beside the run log when that is on.
It is off unless ``start`` has been called, and every call is a no-op while it is, so the
planner can report stages unconditionally without knowing whether anyone is listening.

Only named stages are written, never the individual attempts inside them.  Reading a stage
asks the cell for forward kinematics and clearance at every point, which the passes have
usually cached already but not always, so the work counters in the run log include it,
and the cell's clearance cache is warmed by it.
"""
from __future__ import annotations

import copy

import numpy as np

_fh = None
_route = 0

_LEGEND = """\
One block per route handed to refinement, one table per stage.  Columns:
  idx     position in this stage's list
  src     * where the point is one of the solver's own states (solver and densify
          stages only; blank elsewhere, since the passes do not keep that mapping)
  near    N when the point reads inside the linear band (always . when the gate refused)
  clr     clearance to the static objects in mm, saturating at the probe distance
  pen     clearance penalty factor at the point
  j5      S when joint 5 is inside the stop band
  joints  the joint vector exactly as the planner holds it (radians, metres if prismatic)
  tool    tool centre point x y z, manifest units
  move    the move from this point to the next: profile (LIN or PTP, from the endpoint
          rule), tool travel, largest joint delta in degrees (the --check-step-deg
          measure), and stop-to-stop time in seconds under that profile
Each stage ends with a summary: move counts by profile, total stop-to-stop time, tool
travel, and the furthest the tool gets from the straight line between the route's ends.
"""


def start(fh) -> None:
    global _fh, _route
    _fh, _route = fh, 0
    fh.write(_LEGEND)


def stop() -> None:
    global _fh
    if _fh is not None:
        _fh.flush()
    _fh = None


def section(title: str) -> None:
    """A heading for whatever follows, such as the segment being planned."""
    if _fh is not None:
        _fh.write(f"\n{'=' * 100}\n=== {title}\n")
        _fh.flush()


def route(title: str) -> None:
    """Start a new route; the stages written after this belong to it."""
    global _route
    if _fh is not None:
        _route += 1
        _fh.write(f"\n--- route {_route}: {title}\n")


def note(text: str) -> None:
    if _fh is not None:
        _fh.write(f"    {text}\n")


def stage(name: str, model, states, sources=None) -> None:
    """Write one stage of the current route.  ``model`` is the route's ``MotionModel``."""
    if _fh is None:
        return
    # A copy with its own near cache: bound_time trusts whatever that cache already holds,
    # so filling it here would change which candidates the passes reject early.
    model = copy.copy(model)
    model._near = dict(model._near)
    cell = model.cell
    scale = cell.man.scale
    states = [np.asarray(q, dtype=float) for q in states]
    marked = set(sources or ())
    tcp = np.array([cell.tcp_at(q) / scale for q in states]) if states else np.zeros((0, 3))
    w = _fh.write
    w(f"\nstage {name}: {len(states)} points\n")
    w("  idx src near     clr    pen j5 | "
      + " ".join(f"{n[-10:]:>10s}" for n in cell.joint_names)
      + " |   tool_x   tool_y   tool_z | move  travel  max_dq     time\n")
    counts = {"LIN": 0, "PTP": 0}
    secs = travel = 0.0
    for k, q in enumerate(states):
        line = (f"  {k:3d}  {'*' if k in marked else ' '}   {'N' if model.near(q) else '.'} "
                f"{cell.clearance_mm(q):7.1f} {cell.penalty_factor(q):6.2f}  "
                f"{'S' if cell.in_stop_band(q) else '.'} | "
                + " ".join(f"{float(v):10.6f}" for v in q)
                + " | " + " ".join(f"{float(v):8.2f}" for v in tcp[k]) + " |")
        if k + 1 < len(states):
            r = states[k + 1]
            kind = model.motion(q, r)
            dist = float(np.linalg.norm(tcp[k + 1] - tcp[k]))
            t = model.move_time(q, r)
            counts[kind] = counts.get(kind, 0) + 1
            secs, travel = secs + t, travel + dist
            line += (f" {kind:>4s} {dist:7.1f} {np.rad2deg(np.max(np.abs(r - q))):7.2f} "
                     f"{t:8.3f}")
        w(line + "\n")
    if len(states) >= 2:
        a, b = tcp[0], tcp[-1]
        ab = b - a
        t = np.clip(((tcp - a) @ ab) / max(float(ab @ ab), 1e-9), 0.0, 1.0)
        off = float(np.max(np.linalg.norm(tcp - (a + t[:, None] * ab), axis=1)))
        w(f"  summary: {counts['LIN']} LIN / {counts['PTP']} PTP moves, stop-to-stop "
          f"{secs:.3f} s, tool travel {travel:.1f}, max off-line {off:.1f}, "
          f"{sum(model.near(q) for q in states)} near, "
          f"{sum(cell.in_stop_band(q) for q in states)} in stop band\n")
