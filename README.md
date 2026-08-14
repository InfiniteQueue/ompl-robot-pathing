# weldpath

Generates a FANUC weld path from a Tesseract study directory.

```bash
python main.py "C:/Users/christopherf/Documents/TestFiles/Test Studies/Tesseract"
```

Reads `<directory>/manifest.json`, plans motion through the manifest's locators, and
writes `<directory>/waypoints.json`. Filenames are fixed.

## Layout

| Path | Purpose |
| --- | --- |
| `<dir>/manifest.json` | input: kinematics, geometry, locators, start state |
| `<dir>/meshes/*.obj` | input: geometry referenced by the manifest |
| `<dir>/waypoints.json` | **output** |
| `<dir>/convex_cache/` | generated: convex collision geometry, reused between runs |
| `<dir>/generated/` | generated: the URDF/SRDF and plugin configs actually loaded |

The two generated directories are safe to delete; they are rebuilt on the next run.

## How it works

1. **Scene build** (`scene.py`). The manifest describes everything in world coordinates
   at one captured configuration, so each link's frame origin is the anchor of the joint
   that creates it, each joint origin is a difference of anchors, and each mesh origin is
   the negation of the link's frame origin. Forward kinematics at the captured
   configuration reproduces the manifest's TCP pose to sub-micron accuracy, which is the
   check that this convention is right.
2. **Collision geometry** (`meshprep.py`). Meshes are convex-decomposed and cached.
3. **Collision matrix** (`cell.py`). Adjacent pairs are disabled, plus any pair already
   in contact in the start pose — reported in the run log.
4. **Planning** (`planning.py`, `toolpath.py`). A direct joint move is tried first;
   otherwise OMPL, and failing that a two-leg route through the start pose. Paths are
   then reduced to the fewest waypoints that still traverse collision-free.
5. **Output** (`output.py`).

## Why the geometry is preprocessed

Three findings from the supplied study drive the design, and all three are things that
make the difference between planning working and not working at all:

* **The collision margin has to be negative.** `planning.contact_ok_distance_mm` is a
  tolerance for contact, not a clearance requirement. The gun rests against the robot
  base in the start pose. With Tesseract's default zero margin the start state is invalid
  and every planner fails instantly — which is exactly the `freespace transit failed
  (FreespacePipeline + OMPLPipeline)` in the study's original `waypoints.json`.
* **Raw meshes are far too slow.** Checking the CAD meshes as concave geometry costs
  ~708 ms per discrete collision check, so the sampling planner exhausts its time budget
  having explored almost nothing. Convex geometry costs 1–2 ms.
* **One convex hull per link is wrong.** The weld gun is C-shaped; a single hull fills
  the throat the panel has to sit in, so every weld would report a collision. The OBJs
  are flat triangle soups, but they are assemblies — the gun is 948 disconnected shells,
  each panel over 120. Splitting on connected components and writing each shell as its
  own `o` group makes Tesseract build one convex hull per shell, keeping the throat open.

Shells smaller than `--min-shell-mm` are dropped and only the `--max-shells` largest are
kept, because every shell is a collision pair to test. On the sample cell this takes the
gun from 948 shells to 80 and a check from 2.8 ms to 1.6 ms — the difference between the
freespace transit timing out and solving in a few seconds.

## Output format

The schema is the one `TesseractWaypoints.vb` deserialises. Only members that file
declares are written.

```jsonc
{
 "units": "mm/deg",
 "joint_names": ["robot_j1", "..."],
 "segments": [
  {
   "from": "via9",
   "to": "via10",
   "phases": [
    {
     "motion": "PTP",           // PTP | LIN
     "contact_allowed": false,  // true where the gun is deliberately against a panel
     "gun_opening_mm": 0.0,     // one gun state per phase, not per waypoint
     "waypoints": [
      {
       "joints": {"robot_j1": -1.9547943, "...": 0.0},
       "tcp_world_mm": [[0.1736, 0.0, -0.9848, 1002.711],
                        [0.0, -1.0, 0.0, 2136.612],
                        [-0.9848, 0.0, -0.1736, 1215.386],
                        [0.0, 0.0, 0.0, 1.0]],
       "time": 0.0              // seconds from the start of this phase
      }
     ]
    }
   ]
   // "error": present instead of planned phases if the segment could not be planned
  }
 ]
}
```

Three details of the contract are easy to get wrong and are worth restating:

* **`joints` values are radians**, not degrees, despite `units` reading `"mm/deg"` — the
  comment in the `.vb` is explicit that joint values are in the planner's native units
  (radians for revolute, mm for prismatic). `joint_names` lists the six robot joints,
  matching the study's own sample; the gun is reported through `gun_opening_mm`.
* **`tcp_world_mm` is the full 4×4 row-major world pose**, translation in mm in the
  fourth column, in the same frame as the manifest's `locators[].pose_world`, so it drops
  straight back into the cell with no transformation.
* **`gun_opening_mm` and `contact_allowed` belong to the phase**, since a phase is a run
  of waypoints sharing one motion type and one gun state.

`time` comes from the velocity profile described below, reset to zero at the start of each
phase. It is a plausible schedule for the consumer to read, not a controller-verified one.

Every segment runs from the pose of the locator named in `from` to the pose of the locator
named in `to`, so the path starts at the manifest's first locator. Both ends are checked
after generation, with any deviation over 1 mm reported as a warning.

`start_state` never contributes a waypoint. It is used to seed inverse kinematics and as a
known-clear pose to route a difficult transit through, but getting the robot from wherever
it is onto the first locator is left to the caller.

## Choosing a joint solution for a locator

A TCP pose does not determine the arm configuration, and picking badly is expensive. Two
things decide it:

* **Joint wrapping.** J4 and J6 travel ±360°, so a pose usually has more than one legal
  representation. On the sample cell the solver returned via11 at J6 = +155.5°, forcing a
  325.1° spin; J6 = −204.5° is the identical pose to 3e-16 rad and needs 34.9°. Every
  candidate is shifted onto its nearest legal turn before being judged.
* **A weighted distance.** Solutions are ranked by how far the *tool* moves, using each
  joint's measured TCP travel per radian (J3 ≈ 2239 mm/rad against J4 ≈ 624 mm/rad on this
  cell), so a wide J1 swing is not treated as equivalent to a wrist twist.

KDL's solver is local and returns one solution per seed, so scattered extra seeds are used
to expose the other arm configurations rather than only the branch nearest the previous
locator.

## Shortening the route between locators

Picking good endpoints is not enough. RRTConnect returns the *first* path it finds, and
reducing waypoints cannot change a route — deleting a point that a straight move already
bypasses leaves a wide arc exactly as wide. So a shortcutting pass runs before the
reduction: it densifies the planner's path, then repeatedly picks two states on it and
splices in the direct move between them whenever that move is collision free and cheaper
under the same weighted metric. Every replacement is collision-checked before it is kept,
so the result is traversable by construction.

The weighting is what makes it cut the right thing — an excursion in J1 costs what it
actually costs at the tool, instead of the same as a wrist rotation. On the sample study,
emitted travel per segment:

| Segment | Before | After | Direct move needs |
| --- | --- | --- | --- |
| via9 → via10 | 590° | 341° | 182° |
| via10 → via11 | 860° | 212° | 190° |

Repeating one segment four times each way, the medians were 1083° → 351° and 749° → 336°,
with the worst single joint dropping from 426° to 124°. It costs 2–4 s per transit and
usually adds a couple of waypoints, since a tighter route has more corners worth keeping.
Turn it off with `--no-shortcut`.

Point counts are kept low on purpose: a transit is reduced to the fewest waypoints that
still traverse it collision-free, so a large sweeping motion costs a handful of points
rather than dozens.

A segment that cannot be planned gets an `error` and empty `phases`; the rest of the file
is still written and the process exits non-zero.

## Velocity profile

Waypoints are scheduled along a trapezoidal velocity profile, so displacement follows an S
curve rather than a straight line. `--accel-blend` sets the share of a move spent
accelerating or decelerating:

| `--accel-blend` | Shape | Each ramp | Cruise | Mean speed | Duration |
| --- | --- | --- | --- | --- | --- |
| 0 | flat velocity | — | 100% | peak | `D / v` |
| 0.5 (default) | trapezoid | 25% | 50% | 75% of peak | `1.33 × D / v` |
| 1 | bang-bang | 50% | — | 50% of peak | `2 × D / v` |

`--joint-speed-deg-s` and `--linear-speed-mm-s` are **peak** speeds, not averages, so
adding ramps makes a move take longer: mean speed is `1 - blend/2` of peak and duration
scales as `1 / (1 - blend/2)`. `--accel-blend 0` reproduces the old constant-velocity
timing exactly.

The profile spans a **phase**, starting and ending at rest. That follows from the schema
resetting `time` to zero per phase — each phase is its own motion block, and phase
boundaries are where the gun state or motion type changes, which is where the robot would
stop anyway. All joints share one time base, so a coordinated move stays coordinated: each
joint's displacement follows the same S curve scaled by its own excursion.

Distance is measured in whatever governs the move — tool travel in mm for `LIN`, and for
`PTP` the largest joint excursion of each step, so each step is timed by whichever joint
has furthest to go.

Only the times change; **waypoint positions are untouched**. With `--accel-blend 0.5` on a
7-point 300 mm linear move at 250 mm/s the deltas come out `0.4, 0.2, 0.2, 0.2, 0.2, 0.4` —
the cruise steps run at peak speed and only the ramp steps stretch.

One consequence worth knowing: a phase reduced to just two waypoints carries the profile
only through its total duration, since there are no intermediate points for the shape to
show up in. Multi-point phases (a linear approach, or a transit that needed a detour) show
it directly. Adding points purely to express the curve would work against keeping the
program short, so it is not done.

## Welds

At a weld locator the robot runs a straight `linear_zone_mm` approach, stops, and departs
along the same line. Per the brief the gun is treated as stationary through the weld, and
because the gun state defines a phase this comes out as a phase boundary:

| Phase | motion | contact_allowed | gun_opening_mm |
| --- | --- | --- | --- |
| approach onto the joint | `LIN` | true | `gun_opening_arrive` |
| the weld itself — one waypoint, robot stationary | `LIN` | true | `gun_opening_leave` |
| depart along the same line | `LIN` | true | `gun_opening_leave` |
| transit to the next locator | `PTP` | false | `gun_opening_leave` |

No motion is planned for the closing itself. The approach ends and the depart begins on
the locator pose, so the single-waypoint weld phase is where the opening changes.

`contact_allowed` marks the phases where the gun is deliberately up against a panel. Note
that the collision margin itself is global (`-contact_ok_distance_mm`), because that value
is a tolerance for mesh approximation error rather than a per-phase permission — so this
flag records intent, not a margin that varied during planning.

**The retract direction is a derived default.** It is taken from the gun's prismatic
stroke axis expressed in the TCP frame, which for the supplied cell comes out as tool −X.
Every locator in the supplied manifest has `is_weld: false`, so this has never been
exercised against real weld data — check it against a manifest that contains welds, and
override with `--approach-axis` if a cell's tool frame is set up differently.

## Options

| Flag | Default | Effect |
| --- | --- | --- |
| `--approach-axis` | derived | TCP-frame retract direction at welds |
| `--linear-step-mm` | 50 | point spacing on linear approach/depart |
| `--check-step-deg` | 3 | collision checking resolution along a move |
| `--segment-length-rad` | 0.02 | collision resolution inside the sampling planner |
| `--ompl-attempts` | 3 | freespace attempts before the fallback route |
| `--no-shortcut` | off | emit the sampling planner's own route, unshortened |
| `--shortcut-seconds` | 2 | time budget for shortcutting each transit |
| `--min-shell-mm` | 40 | drop collision shells smaller than this |
| `--max-shells` | 80 | cap convex shells per link |
| `--joint-speed-deg-s` | 180 | peak joint speed behind the `time` field |
| `--linear-speed-mm-s` | 250 | peak tool speed on `LIN` moves |
| `--accel-blend` | 0.5 | 0 flat velocity … 1 bang-bang; see above |
| `--quiet` | off | print only the summary |

Exit code is 0 when every segment planned, 1 otherwise.

## Notes and limitations

* Freespace planning is randomised, so a marginal segment can take a different number of
  attempts between runs. The fallback route through the start pose makes this much less
  likely, but a cell with tighter clearances may need `--ompl-attempts` raised.
* The Python OMPL bindings do not expose the planner's time budget, so difficulty is
  managed by making collision checks cheaper rather than by planning for longer.
* Convex decomposition overstates penetration where a hull is a poor fit — on the sample
  cell the gun/base pair reads 62 mm against ~13 mm on exact meshes. Those pairs are
  disabled from the start pose and listed in the log rather than silently tolerated.
* Requires `tesseract-robotics` and `numpy`; both are already in `.venv`.
