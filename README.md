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
| `<dir>/waypoints-unrefined.json` | output, only with `--unrefined-output`: the same motion before shortcutting and reduction |
| `<dir>/weldpath-log.txt` | output, unless `--write-log false`: everything the run printed, replaced each run |
| `<dir>/convex_cache/` | generated: shells for the convex decomposition, reused between runs |
| `<dir>/generated/` | generated: the URDF/SRDF and plugin configs actually loaded |
| `<dir>/collision_geometry/` | generated, only with `--export-collision-geometry`: the convex geometry actually collided against, plus a `blocked_*.obj` per unplaceable locator |

The generated directories are safe to delete; they are rebuilt on the next run.

## How it works

1. **Scene build** (`scene.py`). The manifest describes everything in world coordinates
   at one captured configuration, so each link's frame origin is the anchor of the joint
   that creates it, each joint origin is a difference of anchors, and each mesh origin is
   the negation of the link's frame origin. Forward kinematics at the captured
   configuration reproduces the manifest's TCP pose to sub-micron accuracy, which is the
   check that this convention is right.
2. **Collision geometry** (`meshprep.py`). Meshes are convex-decomposed and cached.
3. **Collision matrix** (`cell.py`). Adjacent pairs are disabled, as are two groups that
   can never be informative: the arm against itself, and the gun against the wrist it is
   bolted to — the attachment link, the link before it, and the link driven by joint 3,
   which the arm folds the gun back against in ordinary poses. Any other pair that reads as in contact in the start pose is re-measured
   against the raw concave meshes: a pair that merely grazes keeps its collision check
   under a corrected margin, and only a genuine overlap is disabled. Both outcomes are
   reported in the run log. Contact between the robot or gun and the panels or tooling is
   never waived this way -- a study that starts inside the parts is rejected with the
   offending pairs and their overlaps.
4. **Planning** (`planning.py`, `cartesian.py`, `toolpath.py`, `fallback/`). Each transit
   is solved whole, by the first of these that works: a direct joint move; OMPL, keeping
   the cheapest of several runs; a Cartesian-space tree whose edges are straight tool moves
   (only while the near-panel band is on); longer OMPL runs; the same again through a
   fallback pose searched for that transit; and finally two legs with the gun changing
   between them. Every gun opening is tried directly before any fallback pose is. Each
   attempt runs with the gun tip where that phase will hold it. The route is then improved
   under a **time** metric — the same bang-bang joint dynamics the output is scheduled with,
   penalised for running close to the parts — by cutting detours, dropping waypoints that
   are not worth stopping at and relocating the rest. Whether each move is flown `LIN` or
   `PTP` is read from where its two ends sit, not from which solver found it; see
   [Linear motion near the parts](#linear-motion-near-the-parts).
5. **Output** (`output.py`).

## Why the geometry is preprocessed

Three findings from the supplied study drive the design, and all three are things that
make the difference between planning working and not working at all:

* **The collision margin has to be negative — but only between the robot's own links.**
  `planning.contact_ok_distance_mm` is a tolerance for contact, not a clearance
  requirement. The gun rests against the robot base in the start pose, so with
  Tesseract's default zero margin the start state is invalid and every planner fails
  instantly — exactly the `freespace transit failed (FreespacePipeline + OMPLPipeline)` in
  the study's original `waypoints.json`. That tolerance exists to absorb error in
  approximating a link's *own* geometry, which is a self-collision concern, so it is
  applied only to link-against-link pairs. Against the panels and tooling the margin is
  `--obstacle-clearance-mm` instead, which defaults to 0 — the robot either clears the
  part or it hits it. Both figures are printed at load.
* **Raw meshes are far too slow.** Checking the CAD meshes as concave geometry costs
  ~708 ms per discrete collision check, so the sampling planner exhausts its time budget
  having explored almost nothing. Convex geometry costs 1–2 ms.
* **One convex hull per link is wrong.** The weld gun is C-shaped; a single hull fills
  the throat the panel has to sit in, so every weld would report a collision. The OBJs
  are flat triangle soups, but they are assemblies — the gun is 948 disconnected shells,
  each panel over 120. Splitting on connected components and writing each shell as its
  own `o` group makes Tesseract build one convex hull per shell, keeping the throat open.

`convex_cache` holds the *input* to that step, not its output. Each file groups the
shells as `o shell_NNNNN`, and each group still carries the original concave triangles;
Tesseract computes the hulls itself at load, because the URDF tags every prepared mesh
`tesseract:make_convex="true"`. Opening a cache file therefore shows faithful CAD and says
nothing about how coarse the collision model is. To see that, run with
`--export-collision-geometry`, which reads the hull vertices back out of the loaded
environment and writes them to `<dir>/collision_geometry/`, one OBJ per link with each
convex piece as its own `o` group, in world coordinates and the manifest's units so they
drop straight on top of the source meshes in a viewer. That is the geometry Bullet tests,
so a bridged throat or a filled recess is visible there and nowhere else.

The same flag also turns the "cannot place the robot at locator" failure into something
you can look at. When a locator's pose is reachable but rejected by the collision check,
the two links that blocked it are written to `blocked_<locator>_at_<opening>mm.obj` — just
those two, at the joint state that was rejected, each link's pieces grouped under its own
name. The log already names the pair, its depth and where the closest approach sits in the
TCP's frame; the file is the same event with the geometry attached, which is usually the
faster way to tell a real interference from a hull bridging a recess.

### Asking what occupies a point

Opening the exported geometry tells you a shape is there. It does not tell you *which*
shape, which shell it came from, or whether the CAD agrees. `--probe-point` answers all
three and then stops, without planning:

```bash
python main.py <dir> --probe-point "73050-01-R_2t:-39.8,,-82.1"
```

The point is given in a locator's own frame, because that is how the discrepancy is seen —
the robot is jumped to a weld and something is in the way a fixed distance along the tool's
axes. An empty axis means zero, so a two-axis offset does not have to be padded by hand and
land in the wrong slot. A bare `x,y,z` is world coordinates.

It reports the convex pieces containing the point and how deep inside each it sits, the
distance from the point to the nearest triangle of the **source** mesh, and what the
decomposition did to the links implicated — shells kept, shells refined, the cell and fill
gate used. The comparison is the whole point: a hull containing a point that has no real
material within tens of millimetres is a hull bridging a void, and that is called out
explicitly. So is a link where nothing was refined at all, which is the usual reason.

Distances to the source mesh are exact point-to-triangle, not point-to-vertex: a coarsely
tessellated fixture has triangles far larger than the gap being measured, and the nearest
vertex of one can be a long way from its nearest point. Every triangle gets a cheap
bounding-box lower bound first and only the closest few hundred are measured properly.

The surface is rebuilt by hulling those vertices rather than by reading the face lists that
come with them, which are not usable: on a 39-vertex hull the faces name 36 vertices apiece
and those vertices sit up to a metre off the plane of their own face. Since the points are
already a hull's vertices, hulling them again reproduces the collision shape exactly — on
`robot_base` every piece comes out closed with exactly 2V−4 triangles. The export costs
about 90 s on the sample cell and is off by default.

Shells smaller than `--min-shell-mm` are dropped and only the `--max-shells` largest are
kept, because every shell is a collision pair to test. Measured on an earlier study whose
gun was one link, this took it from 948 shells to 80 and a check from 2.8 ms to 1.6 ms — the
difference between the freespace transit timing out and solving in a few seconds. The
current cell splits the gun into a body (46 shells) and a moving tip (159), and a discrete
check costs about 1.0 ms.

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
     "gun_opening_mm": 0.0,     // one gun state per phase
     "waypoints": [
      {
       "joints": {"robot_j1": -1.9547943, "...": 0.0},
       "tcp_world_mm": [[0.1736, 0.0, -0.9848, 1002.711],
                        [0.0, -1.0, 0.0, 2136.612],
                        [-0.9848, 0.0, -0.1736, 1215.386],
                        [0.0, 0.0, 0.0, 1.0]],
       "time": 0.0,             // seconds from the start of this phase
       "motion": "PTP",         // repeated from the phase, for flat waypoint lists
       "gun_opening_mm": 0.0    // likewise
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

* **`joints` values are radians** for revolute joints and mm for prismatic, despite
  `units` reading `"mm/deg"` — the comment in the `.vb` is explicit that joint values are in
  the planner's native units. They are the **robot's register values, not the planner's**,
  which differ for joint 3, whose linkage holds it against the floor; see
  [Joint 3 is coupled to joint 2](#joint-3-is-coupled-to-joint-2). `joint_names` lists the
  six robot joints, matching the study's own sample; the gun is reported through
  `gun_opening_mm`.
* **`tcp_world_mm` is the full 4×4 row-major world pose**, translation in mm in the
  fourth column, in the same frame as the manifest's `locators[].pose_world`, so it drops
  straight back into the cell with no transformation.
* **`gun_opening_mm` and `contact_allowed` belong to the phase**, since a phase is a run
  of waypoints sharing one motion type and one gun state. `motion` and `gun_opening_mm`
  are repeated on every waypoint as a convenience for a consumer walking a flat list, so
  that how a point is reached and where the gun must be are both readable without carrying
  the enclosing phase along. They are copies of the phase's values and cannot disagree with
  them; `contact_allowed` is not repeated. On the first waypoint of a phase, `motion`
  describes the move that *leaves* it — there is no move into it.

A segment usually contains **several phases**. Consecutive moves under the same profile form
one phase, so a transit that works along a panel and then swings clear alternates between
`LIN` and `PTP` phases, and each phase's first waypoint repeats the last waypoint of the one
before it. A segment can also change gun opening partway: when no single opening gets the
robot through the transit, the move is split and the gun changes at the join, where the
robot is stationary. The consumer has to drive the gun to each phase's `gun_opening_mm`
before executing that phase, which was already implied by the schema carrying one opening
per phase.

`time` comes from the velocity profile described below, reset to zero at the start of each
phase. It is a plausible schedule for the consumer to read, not a controller-verified one.

Every segment runs from the pose of the locator named in `from` to the pose of the locator
named in `to`, so the path starts at the manifest's first locator. Both ends are checked
after generation, with any deviation over 1 mm reported as a warning.

`start_state`'s gun entry is an **opening in millimetres**, like every other opening in the
manifest — the key is named `..._mm` — and is converted to a joint angle on load. Reading it
as a raw joint value put the start state hundreds of radians out and left the tip wherever
that landed. The opening it resolves to is printed at load, along with a note if the manifest
asked for more than the gun has.

`start_state` never contributes a waypoint. It is used to seed inverse kinematics, and
getting the robot from wherever it is onto the first locator is left to the caller. It is
not what a difficult transit detours through: those poses are searched for per transit, by
`weldpath.fallback`.

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

### Choosing which solution to optimise

RRTConnect returns whichever route it stumbles on first, and which side of a fixture that
lands on is luck. Nothing downstream can correct it: cutting and relocation reshape a route
but cannot move it into a different homotopy class, so whichever one arrives is the one that
ships. `--ompl-min-runs` therefore samples several solutions per transit and keeps the one
with the lowest penalised cost before handing it to the optimiser.

The spread is large enough to matter. On one `via10 -> via11` transit, five runs came back at
3.48 s, 2.88 s, 2.53 s, 3.33 s and 2.62 s of penalised cost — a 38% spread between best and
worst, all of them valid, all found within a second of each other. Ranking on the penalty
rather than raw time genuinely reorders them: the 2.88 s run is quicker unpenalised than the
2.62 s one (1.88 s against 2.15 s), and loses because it spends that time closer to the
panels.

### How long each run may search

`--ompl-seconds` caps a single solve. It matters independently of the run count: restarting
throws away the search tree, so a transit through a genuinely tight gap is better served by
one long run than by several short ones. On the sample study the `via9 -> via10` direct
transit never once solved at the 5 s default and always fell back to a two-leg route through
the start pose; at 20 s it solves directly.

| `--ompl-seconds` | wall clock | solved? |
| --- | --- | --- |
| 5 | 6.25 s | no |
| 12 | 13.80 s | no |
| 20 | 20.40 s | **yes** |

Budget for it multiplying: worst case is `--ompl-seconds x --ompl-min-runs` per transit.

> **This one reaches past the bindings.** Tesseract's `OMPLSolverConfig` is not wrapped by
> `tesseract_robotics` — `profile.solver_config` comes back as a bare `SwigPyObject` with no
> members exposed, and the module defines no constructor, accessor or factory for the type,
> so `planning_time` cannot be set through any supported call. (`OMPLMotionPlanner.terminate()`
> exists, but only ever *shortens* a solve, and the binding warns it is unimplemented.) The
> value is therefore written straight into the C++ struct. To earn that, the code does not
> trust a hard-coded offset: it scans for the documented default layout — `planning_time`
> immediately followed by `max_solutions=10`, `simplify=false`, `optimize=true` — refuses to
> write unless exactly one candidate matches, and reads the value back afterwards. A build
> that reorders the struct or changes its defaults produces no match, so it declines and
> warns rather than corrupting a neighbouring field. Leaving the flag at its 5 s default
> touches nothing at all.

The minimum is a floor rather than a cap. Once it is met and at least one solution exists,
planning moves on; if nothing has solved, runs continue up to `--ompl-attempts`. Each run is
a full solve of several seconds, so this is the most expensive knob in the tool — it is also
the only one that can change the route's basic shape. With `--no-clearance-penalty` the score
degrades to plain cruise time, so it still picks the quickest of the sampled solutions.

### Seeing what the refinement changed

`--unrefined-output` writes a second file, `waypoints-unrefined.json`, holding each transit
exactly as the planner returned the solution that was chosen — before shortcutting and
before waypoint reduction. It uses the same schema as `waypoints.json`, and the weld phases
are identical in both, since nothing optimises those. Each transit is written there as a
single `PTP` phase whatever profiles it ships with in `waypoints.json`, including a route
the Cartesian tree built out of straight tool moves. Only the route that was actually kept
is recorded; solutions the run discarded leave nothing behind.

The two files line up waypoint-for-waypoint at their ends, so a segment can be replayed
either way and compared. On a two-segment sample the point counts came out the same either
way — 4 and 3 — while the emitted schedule fell from 3.65 s to 3.30 s and from 2.99 s to
2.50 s. Refinement is not mainly a waypoint-count story: the sampling planner returns few
points to begin with, and what the passes buy is a better route between them.

Picking good endpoints is not enough. RRTConnect returns the *first* path it finds, and
reducing waypoints cannot change a route — deleting a point that a straight move already
bypasses leaves a wide arc exactly as wide. So a shortcutting pass runs before the
reduction: it densifies the planner's path, then repeatedly picks two states on it and
splices in the direct move between them whenever that move is collision free and quicker.
Every replacement is collision-checked before it is kept, so the result is traversable by
construction.

Timing it under the real joint limits is what makes it cut the right thing — a J1 excursion
costs what it actually costs, instead of the same as a wrist rotation covering the same
angle several times faster. On the sample study, emitted travel per segment:

| Segment | Before | After | Direct move needs |
| --- | --- | --- | --- |
| via9 → via10 | 590° | 341° | 182° |
| via10 → via11 | 860° | 212° | 190° |

Repeating one segment four times each way, the medians were 1083° → 351° and 749° → 336°,
with the worst single joint dropping from 426° to 124°. It costs 2–4 s per transit and
usually adds a couple of waypoints, since a tighter route has more corners worth keeping.
Turn it off with `--no-shortcut`.

Cutting alone can only ever *remove* path, so it cannot move away from an obstacle. A
second move runs in the same loop and same budget: **relocation** displaces a single
waypoint and keeps it when both neighbouring moves stay clear and the pair gets cheaper.
Displacements are drawn in joint space but scaled by the joint weights, so an attempt moves
the tool about as far whichever joints it uses — otherwise nearly every sample would be a
wrist twiddle that changes nothing. Every waypoint is eligible except the two endpoints,
which are the states handed in and belong to the locators either side. Relocation is what
gives the clearance penalty below any teeth: standing a waypoint off a panel costs a little
travel and saves a lot of penalty, so it wins.

### What the cost actually is

Cost is **time**, under the same bang-bang joint limits the output is scheduled with, and
time near a panel is multiplied by the clearance penalty. That is the currency the penalty
was specified in — *a second at the minimum clearance costs as much as N seconds in open
space* — so the two halves finally compose.

The reason this matters more than swapping one distance for another is that **time is not
additive and distance is**. Splitting a move in two costs an extra pair of ramps, so a
waypoint that buys nothing is now visibly expensive; under a distance metric it was exactly
free, which is why paths came back with points clustered a few degrees apart. The measure is
applied at two scopes, and the distinction is load-bearing:

| Pass | Works on | Measure |
| --- | --- | --- |
| shortcut | densified path | **cruise** time — ramps excluded |
| simplify | dense → emitted | full stop-to-stop time |
| polish | emitted waypoints | full stop-to-stop time |

A point on the densified path is a sampling artefact, not a stop the robot will make.
Charging it a full ramp would score a route by how finely it happened to be sampled and make
every cut look good regardless of where it went. Cruise time (`max |Δq| / v` per joint) is
unchanged by subdivision, so shortcutting judges the route's *shape* while the later passes
judge how many stops it needs. The two agree to one constant `v/a` per move in the
trapezoidal regime, so cruise time is a genuine lower bound rather than a different
currency.

### Polishing the emitted waypoints

`--polish-seconds` runs a final pass over the waypoints that will actually be written. Both
its moves are judged on full stop-to-stop penalised time:

* **remove** a waypoint when going straight past it beats stopping at it. Ignoring the
  penalty this is always true — the direct move is no further on any joint and pays one pair
  of ramps instead of two — so what it really tests is whether the corner was bought for
  clearance or is just left over.
* **relocate** a waypoint when the pair of moves through it gets quicker. This is the move
  `simplify` cannot make, and it is what unpicks a cluster: shifting a point away from a
  panel is often what makes its neighbour droppable, so a removal sweep follows every
  accepted relocation.

Measured against the previous distance-based optimiser on **identical raw solutions**, same
random seed and same budgets:

| Segment | Waypoints | Emitted time | Closest approach |
| --- | --- | --- | --- |
| via9 → via10 | 12 → **4** | 6.13 s → **3.27 s** | 188 mm → 103 mm |
| via10 → via11 | 11 → **4** | 5.44 s → **3.05 s** | 76 mm → 41 mm |

Roughly a third of the waypoints and a 45% shorter cycle. Note the third column: the routes
run **closer to the parts** than they used to. That is the trade being made on purpose —
standoff now competes against time instead of being free — and both figures stay far above
the 10 mm the penalty peaks at. Raise `--clearance-penalty-multiplier` to buy the distance
back at the cost of cycle time.

Point counts are kept low on purpose: a transit is reduced to the fewest waypoints that
still traverse it collision-free, so a large sweeping motion costs a handful of points
rather than dozens.

A segment that cannot be planned gets an `error` and empty `phases`; the rest of the file
is still written and the process exits non-zero.

### Linear motion near the parts

Close to a panel, joint motion is hard to reason about: the tool sweeps an arc whose shape
depends on the arm's configuration rather than on anything you can see in the cell. A
straight-line move is predictable, which is what you want where the margin for error is
small. Far from the parts none of that matters and joint motion is both quicker to execute
and easier to plan.

Nothing is *planned* as linear motion. Every solver hands back joint states, and the profile
each move is flown under is read off the route afterwards: a move is `LIN` when either of
its two ends is within `--near-panel-mm` of a panel or tooling, and `PTP` otherwise. Where a
stretch came from plays no part. An OMPL edge with both ends in the band ships linear, and
an edge of the Cartesian tree with both ends outside it ships as a joint move. Deciding by
measurement is the point: where a near-panel stretch begins and ends is itself an output of
pathfinding, so it cannot be picked in advance.

From start to finish:

1. **Band on or off, for the run.** `--near-panel-mm 0` or `--no-near-panel-linear` turns
   it off: every transit move is `PTP`, and the Cartesian tree is never tried either.
2. **Solve.** Whichever solver succeeds (see [How it works](#how-it-works)) returns an
   unlabelled route.
3. **Gate, per leg.** The route is sampled at `--check-step-deg` and each sample is marked
   near or not. The leg earns linear motion if at least `--near-panel-min-pct` of the
   samples are near, or if any unbroken near stretch covers `--near-panel-min-mm` of tool
   travel. A leg that meets neither is `PTP` throughout, however close it runs.
4. **Label, per move.** On a leg that passed, `LIN` if either end is near, else `PTP`. So a
   waypoint is reached by a linear move when it is in the band or the waypoint before it
   was.
5. **Refine.** Shortcut, simplify and polish move and delete waypoints, and each move they
   propose takes its profile from its new ends. A `LIN` candidate is checked along the
   tool's straight line and costed under `--linear-speed-mm-s`; a `PTP` one is checked
   along the joint chord. A replacement that would swallow near waypoints into a joint
   move with both ends outside the band is refused, so the passes cannot dissolve a
   near-panel stretch into one long joint arc. `--linear-crossing-penalty-s` adds a flat
   cost each time a move enters the band from outside.
6. **Split.** Consecutive moves under the same profile become one phase.
7. **Verify.** Every move is swept along the path its profile implies. A failure discards
   the whole route, and the planner moves on to its next gun opening or fallback pose;
   nothing is relabelled or re-planned to rescue it.

A route through a fallback pose is gated and labelled once, over the joined route. A transit
split for a gun change is gated and labelled per leg, under that leg's own opening. The
final per-phase validation checks each phase under its label, and `output.Timing` holds `LIN`
moves to the tool speed cap.

**Brief contact with the band is ignored, but "brief" is measured two ways.** A sweeping
transit that clips the band for a moment is not working near the panel, and cutting it into
three phases to say so would cost a stop at each end for nothing, so `--near-panel-min-mm`
asks for a stretch of real length. On its own that fails a short move: a 60 mm hop from one
weld to the next is near the panel for its whole length, yet no threshold worth setting for
transits would admit it. `--near-panel-min-pct` covers that case. It is measured over the
whole leg rather than per stretch, so a retract whose apex leaves the band for an instant
does not disqualify the leg either side of it.

**Routes from the Cartesian tree.** The tree is tried only when OMPL's first phase found
nothing and the band is on, and never for the two halves of a route through a fallback pose.
Its route is dense: states about `--check-step-mm` of tool travel apart, each gap a straight
tool move, plus one stationary joint move where its two trees meet.

With `--cartesian-recut` on (the default), the route is then cut wherever it leaves the
band. Each stretch outside the band is bounded by its own first and last state, which
become waypoints just outside the band, and every state between them is dropped, linear or
not. The near-panel stretches keep the states the tree validated. The gap between each pair
of cuts is replanned as joint motion: if the straight joint move between the cuts is clear,
that move is used with no search; otherwise the gap gets the regular phase one and phase
two. A gap they cannot join keeps its Cartesian states. Two cases are left alone: a stretch
with nothing between its cuts, and a route with no state in the band at all, since
replanning that would repeat the search phase one has just failed. Each gap that needs a
search can cost up to the full phase-one and phase-two budget.

The route then goes through exactly the same gate and labelling as an OMPL route. With
`--cartesian-recut false`, nothing turns the out-of-band stretches into joint motion
beforehand: those moves label `PTP` by the endpoint rule, and refinement turns them into
long joint moves by cutting across them once both ends of a cut lie outside the band. If
the gate refuses the leg, the whole route ships `PTP`.

**How "near" is measured.** The clearance query only looks as far as its probe, and when it
finds nothing within that it returns the probe distance itself. So a state counts as near
only when its reading is at or under `--near-panel-mm` *and* short of the probe; a reading at
the probe means nothing was found, not that something is exactly that far away. The probe
is sized to `--near-panel-mm` plus 25 mm, so every state inside the band is measured
properly. It can grow during a run — a fallback-pose search widens it to
`--fallback-distance-mm` — but it never drops below the band, so that changes no label.

Until this was fixed the probe was sized to exactly `--near-panel-mm`, and a saturated
reading passed the "at or under" test. Every state read as near, legs reported 100% in
range, and nearly every move came out `LIN`, except on legs planned after the first
fallback search had widened the probe.

Welds get no straight lead-in or lead-out of their own; linear motion near a weld comes from
this rule like anywhere else.

## Velocity profile

Every waypoint is a full stop, and each move runs **bang-bang**: each joint accelerates at
its limit until it either reaches its velocity limit or has to start braking. So the profile
is a triangle for a short move and becomes a trapezoid only once the velocity limit is
actually reached, at `d = v²/a`:

```
t = 2·√(d/a)      d ≤ v²/a     triangular, never reaches v
    d/v + v/a     d > v²/a     trapezoidal, cruises at v
```

Both give `2v/a` at the crossover, so the two branches meet continuously. A move is
coordinated — all joints start and stop together — so its duration is the slowest joint's.

Limits are set with `--joint-max-velocity` and `--joint-max-acceleration`, each taking
comma-separated values in joint order, or a single value applied to every joint. Defaults
are 2π/3 rad/s and 2.5 rad/s² for J1–J5, and 11π/9 rad/s and 11 rad/s² for J6.

**Time is no longer linear in distance**, which is the point. The previous model spread one
trapezoid across a whole phase with the ramps as a fixed *fraction* of each move, which
implied acceleration scaling with move length and made splitting a move free. Now splitting
costs real time — for a triangular move, √2 per doubling:

| 1.0 rad split into | 1 move | 2 | 4 | 10 |
| --- | --- | --- | --- | --- |
| Duration | 1.265 s | 1.789 s | 2.530 s | 4.000 s |

Expect emitted times to roughly double against the old model. Scheduling one sample study's
two transits both ways, they went from 1.940 s and 1.907 s to 4.604 s and 3.773 s — 2.18×
overall. The old figures were optimistic because they charged nothing for stopping.

`--linear-speed-mm-s` remains as a commanded tool-speed cap on `LIN` moves; the joint limits
still govern whenever they are slower. No Cartesian acceleration is modelled, because the
manifest supplies none.

### Joint 3 is coupled to joint 2

A linkage holds link 3 at a fixed angle to the floor as joint 2 moves, so the number the
robot's J3 register carries is not the relative rotation a URDF models. Measured against
this cell's own kinematics, link 3's elevation is a function of `q3 − q2` alone: it reads
the same at `(q2, q3)` of `(−0.3, −0.3)`, `(0, 0)` and `(0.3, 0.3)`, and rises from −22.9°
to +35.2° as `q3 − q2` runs from −0.6 to +0.6 rad. So the register reads `q3 − q2`,
increasing as the arm points up, and that is the value written to `waypoints.json` — in
radians, like every other joint there.

**Timing is a separate question.** The drive still moves joint 3 through the URDF's own
relative rotation, so its motion profile runs on `q3` unmodified, exactly like every other
joint. Only the output value is converted — planning, collision checking and timing all work
in the kinematic values throughout.

### Limits in the environment

The same figures are pushed into the Tesseract environment with
`ChangeJointVelocityLimitsCommand` and `ChangeJointAccelerationLimitsCommand`, and the
per-joint velocity is written into the URDF. This matters because Tesseract **synthesises
limits it was not given** — this cell previously came up with ±1.5 rad/s² on every joint,
which nobody specified and which is nowhere near the real machine. Anything reading limits
from the environment (a time parameteriser, say) would otherwise be working from invented
numbers. The gun joint keeps its placeholder velocity, since the manifest describes no
dynamics for it.

## Welds

At a weld locator the robot stops. The transit runs weld to weld and is free to curve away
from the panel immediately; there is no straight lead-in or lead-out. Per the brief the gun
is treated as stationary through the weld, and because the gun state defines a phase this
comes out as a phase boundary:

| Phase | motion | contact_allowed | gun_opening_mm |
| --- | --- | --- | --- |
| the weld itself — one waypoint, robot stationary | `LIN` | true | `gun_opening_leave` |
| transit to the next locator | `LIN` and `PTP` phases, per [Linear motion near the parts](#linear-motion-near-the-parts) | false | chosen; `gun_opening_leave` tried first |

No motion is planned for the closing itself. The weld phase has a single waypoint, so its
`LIN` describes no move; it is where the opening changes. That waypoint is put back onto the
weld's imported pose after validation, while the transits either side end on the pose
shifted by `--weld-shift-mm`. A tour that ends on a weld gets one more single-waypoint weld
phase at the end of its last segment.

If the robot cannot stand collision free at the full shift, the shorter stand-offs between
it and the imported pose are searched, along the same line, and the clear one with the most
clearance is used instead (ties go to the larger stand-off). That pose then replaces the
shifted one everywhere: the transits end on it and the endpoint check reads it. Clear is not
assumed to be monotonic — backing off frees the tip but can put the throat into tooling — so
this is not a bisection:

1. scan the range every `--weld-shift-scan-mm` (0.5);
2. split every gap between two blocked samples in half, and again, until the spacing is at or
   below `--weld-shift-resolution-mm` (0.05). This carries on after something is clear,
   because the first window found need not be the best. A clear window narrower than the
   final spacing can still fall between samples and be missed;
3. in each clear window, climb from its best sample towards more clearance, halving the step
   until it is at or below the resolution. This finds the best point near that sample, not
   necessarily a narrower peak elsewhere in a wide window.

The log names every clear window and the stand-off chosen. If nothing is clear the weld fails
as before, diagnosed at the full shift. `--no-weld-shift-search` turns it off.

`contact_allowed` marks the phases where the gun is deliberately up against a panel. It
records intent: no margin is relaxed for those phases, so a weld whose gun tip genuinely
overlaps the panel geometry will fail to plan rather than be waved through. On the
re-exported sample study every weld locator does exactly that — the tip sits 24.9 to
76.6 mm inside `Assy_ST240_RH` — so those locators need either a negative
`--obstacle-clearance-mm` or a per-phase relaxation that does not yet exist.

## The moving gun tip

The gun's joint is **angular**, not a linear stroke: 0.4005 rad about an axis 515 mm from
the TCP, giving 205 mm of electrode opening. Openings in the manifest are quoted in
millimetres, so every opening is converted through that lever arm, which is measured from
the manifest rather than hardcoded — the TCP sits 3.3 mm from both the fixed and the moving
electrode mesh, confirming it is the electrode gap and the right point to measure from.

Two approximations are accepted. A quoted opening is really the TCP-to-tip distance measured
along the TCP's z axis, down to the tip's lowest point — which the rounding of the tip puts
below its end — whereas this treats it as the chord swept by the TCP. Measured against the
tip mesh the two disagree by 0.5%: 0.26 mm at a 50 mm opening, 1.1 mm at full stroke, which
is far inside the tens of millimetres of error already present in the convex geometry.
Openings are clamped to the joint's travel, since an opening at full stroke converts to a
hair past the limit.

**The tip is a state variable, not an IK variable.** The gun joint branches off the chain at
the wrist, so it is not part of the kinematic group and inverse kinematics cannot touch it —
but it is set alongside the robot joints on every state, and each phase is planned with the
tip where that phase will actually hold it. Before this it sat permanently closed regardless
of what any phase asked for.

Where the tip sits genuinely decides what the robot can do. Sampling 600 random poses, 16
changed collision state with the gun — **in both directions**: some are blocked closed and
clear wide open, others the reverse. So a destination can be unreachable at the opening the
robot arrives with, and the transit planner searches openings for one that works: the
opening the robot departs with first, then the one it has to arrive with, then closed,
widest and half open, plus `--extra-gun-openings` more. Changing the gun is a real
operation on the machine, so a single opening for the whole transit is always preferred;
only when none works is the move split into two legs with the gun changing at the
intermediate pose, where the robot is stationary anyway.

**The departure opening leads because it is already in force.** Using it costs no gun
change at the start of the move, and it is the state the robot was proved to stand at the
departing locator in. The arrival opening is still tried second. The order used to put a
weld's arrival opening first, on the grounds that the weld pose was only ever checked in
it; that holds for the destination, but a transit has to leave its start as well as reach
its end, and the same argument applies there.

**Both endpoints are screened before the planner starts**, for the same reason. Endpoint
validity at a given opening is two contact queries; letting OMPL find out costs a whole
solve, and every retry at that opening rediscovers it just as slowly. A rejected opening
is logged with its reason, since a microsecond screen would otherwise pass in silence
where a failed OMPL run announces itself at length.

A weld's openings are process data and are never overridden — if the robot cannot reach the
pose at the opening the weld schedule states, that is a failure, not a wider gun. An
ordinary via carries no such requirement, so opening or closing the tip to get there is
legitimate and is tried.

The tip's contacts with the rest of the gun and with J5 and J6 are excluded from collision
checking. Swinging through 200 mm of travel inevitably brings it against its own machinery,
those readings say nothing about whether a move is safe, and leaving them in makes the gun
uncloseable in most poses. The wrist links are found by walking up from the link the gun is
bolted to rather than by matching names.

The gun's motion **during** a weld is deliberately not simulated. The robot is stationary
while the opening steps from arrive to leave, the tip's own travel is clear by inspection,
and not simulating it is what keeps the weld from needing the panel collisions switched off.

## Clearance from the parts

`--obstacle-clearance-mm` sets how close the robot and gun may come to the static objects
before it counts as a collision. It is a Tesseract pair margin, so one number covers all
three intents: **positive** keeps that much clear air, **0** means touching collides, and
**negative** tolerates that much overlap. Verified on the weld study by measuring the true
closest approach along the emitted path — at 0 the route skims to 0.5 mm, at 25 it holds
25.6 mm.

Tightening it costs planning time and can make a transit impossible, since it shrinks the
free space the sampler has to work with. It applies to every moving-link/static-object
pair; self-collision continues to use `contact_ok_distance_mm`.

If a pair is already closer than the requested clearance in the start pose, that pair is
held to the distance actually available there and the run says so, rather than declaring
the start state invalid and refusing to plan at all.

The re-measurement is done one pair at a time, in a scene holding only the two links
involved, and the probe distance backs off from 50 mm to 10 mm to 2 mm if Bullet cannot
allocate for it — raw concave meshes run to millions of triangles here, and a mesh-against-mesh
test at a wide probe builds a contact manifold per candidate triangle pair. Narrowing the
probe only ever makes the answer more conservative, since a pair nothing is found near is
treated as clear at the probe used. If no probe succeeds the pair keeps its default margin
and the start-state check has the final say, rather than a distance being invented for it.

**There is no longer a linear zone at a weld.** A weld used to lead in and out along a
fixed direction in the locator's own frame, for a fixed distance. Every millimetre of it had
to be clear along that one direction with no freedom to curve, which a weld set deep in
panelling may simply have no room for — the symptom was a segment failing partway along a
300 mm linear move. The transit now runs weld to weld and can curve away from the panel
immediately; straight running near the panel is found by measuring the route, in
[Linear motion near the parts](#linear-motion-near-the-parts).

Note that `--weld-shift-mm` still backs the tool off along the locator's **z**. If z is not
the panel normal for these welds then the shift is sliding the tool along the surface rather
than off it, and wants revisiting.

### Penalising low clearance

A collision check is a hard yes/no, so a move clearing a panel by half a millimetre is as
legal as one clearing it by half a metre — and the shortest route almost always hugs the
obstacle. The clearance penalty turns proximity into cost instead, applied to *time*: at
`--clearance-penalty-max-mm` and beyond there is no penalty, and at
`--clearance-penalty-min-mm` and below a second costs as much as
`--clearance-penalty-multiplier` seconds in open air. Between them it follows `x²` on
`[0, 1]`, stretched to fit, so cost climbs slowly at first and sharply as the gun closes in:

| Clearance | 50 mm | 40 mm | 26.5 mm | 20 mm | 10 mm | ≤ 3 mm |
| --- | --- | --- | --- | --- | --- | --- |
| Factor | 1.0× | 3.2× | 13.3× | 21.0× | 36.5× | 50× |

It saturates at the minimum rather than continuing to climb; without that the pass would
spend its whole budget fighting over the last millimetre of a clearance already as bad as
it is allowed to get. The penalty is a *preference*, not a safety mechanism — what the robot
is forbidden to do is still set by `--obstacle-clearance-mm`. The two are complementary: a
hard clearance makes poses illegal and quickly makes a cell unplannable, whereas the penalty
makes them expensive, which is the better tool for "prefer standoff".

Measured on the sample study at a 5 s shortcut budget, sampling the executed path every 1°:

| | Closest approach | Samples inside 50 mm | Travel |
| --- | --- | --- | --- |
| `--no-clearance-penalty` | 0.02 mm / −7.58 mm | 116/406 and 68/115 | 7.45 m / 3.22 m |
| default penalty | 42.9 mm / 48.8 mm | 5/413 and 2/172 | 6.91 m / 4.74 m |

The cost is throughput: a clearance query is cheap in isolation (0.24 ms against 1.04 ms for
a collision check, because only the 18 robot-against-cell pairs are ever considered) but it
is paid per sample, and scoring candidates roughly halves the number of shortcut attempts
that fit in a budget. Raise `--shortcut-seconds` to compensate.

`--clearance-penalty-cutoff-mm` truncates the shallow end of the curve so the query looks no
further, leaving the rest of the shape unchanged. That necessarily introduces a
discontinuity — with the defaults a 20 mm cutoff steps the factor straight from 1× to 21×
at 20 mm — which is inherent to the request rather than a defect. On this cell it saved
nothing measurable, since the query is already dominated by fixed overhead rather than by
how many pairs fall inside the margin.

Implementation note: the clearance query runs on a **clone** of the planning contact
manager, whose default margin is driven to −1 m and whose robot-against-static pairs are
widened to the probe distance. Widening the planning manager's own margins instead would
make every near miss read as a collision. The clone was checked against an independent
measurement — a hard 60 mm obstacle clearance, which makes the planning manager itself
report distances — and the two agree to 0.01 mm from 5.6 mm out to 29.6 mm.

## How precise the collision geometry can be made

The false gun/base contact is a hull artifact, confirmed rather than assumed: the deepest
reported contact point sits at world (−1551, 1897, 26), and a ray-parity test against the
raw `robot_base` triangles puts that point **outside** the solid on every axis tried, with
the nearest real material 7.8 mm away. The hull of `robot_base` shell 0 (5792 vertices
collapsing to 39) bridges a void the gun tip is sitting in.

Splitting on connected components cannot fix that — it separates *parts*, and this is one
part that is not convex. `--hull-cell-mm` therefore adds a second stage: a shell whose mesh
fills less than 75% of its bounding box is cut into cells of roughly that size and each
cell hulled separately, with triangles padded into neighbouring cells so the hulls overlap
instead of leaving a seam. Measured against the 7.1 mm ground truth:

| `--hull-cell-mm` | shells | gun/base reads | check |
| --- | --- | --- | --- |
| 0 (off) | 306 | −62.6 mm | 3.8 ms |
| 300 | 1749 | −62.6 mm | 17.7 ms |
| 200 | 3094 | −62.6 mm | 20.6 ms |
| 120 | 5816 | −62.0 mm | 59.8 ms |

**It does not fix this cell, and the cost is severe.** The void being bridged is about
8 mm across, so resolving it needs cells of that order — which would mean hundreds of
thousands of hulls. At 300 mm the weld study already fails to plan at all, because the
slower checks starve the sampling planner of its time budget. Leave it off unless a
particular cell has a large open feature (a gun throat, a deep fixture pocket) that a
single hull is closing off, which is the case it was built for.

### The cell can only be as fine as the triangles

Cutting a shell into cells sorts whole triangles into buckets, so a shell tessellated more
coarsely than the grid cannot honour the cell size it was given: a flat fixture face
exported as two triangles metres across produced a metres-wide hull no matter how small
`--tooling-cell-mm` was set. That is the same defect as measuring focus distance from a
centroid, on the other side of the fence — and it silently capped how much good any of the
refinement controls could do.

Triangles longer than a cell are therefore bisected down to it first, splitting the longest
edge at its midpoint and replacing the triangle with the two halves, repeatedly. The new
vertex sits on an existing edge, so the subdivided mesh occupies **exactly** the same
space as the original — the hulls built from it can only get tighter, never inflate. On a
2 m plate at a 100 mm cell the largest resulting hull spans 265 mm instead of 2828 mm, and
the surface area is unchanged to the last digit.

Bisecting one edge rather than all three costs two triangles per pass instead of four, and
keeps CAD's long thin triangles from being shattered: a 1000 x 1 mm sliver becomes 81
triangles rather than 256. It leaves hanging nodes where a split triangle meets an unsplit
neighbour, which does not matter here — the pieces only ever reach a hull builder, which
reads points and cares nothing for topology, and the watertightness tests all run earlier
on the whole shell.

Two consequences worth knowing:

* **Shell counts on coarse geometry can rise sharply**, because the cell now means what it
  says. A part that used to be capped by its own tessellation is not any more. The fill
  gate below still keeps solid, well-hulling parts out of it entirely, and outside the
  focus radius the coarser far cell applies, so the growth lands where the accuracy was
  actually wanted.
* **The cached geometry is invalidated** for every link being refined, since the same cell
  now yields different shells. Links with no cell set keep their existing cache entries.

Subdivision stops at 400 000 triangles per shell, which degrades to a coarser result rather
than a wrong one.

### Spending the refinement where it matters

A shell is only refined at all if it fills less than `--hull-fill` of its bounding box.
That gate was inert: `hull_fill` judged watertightness on raw triangle indices, and CAD
tessellation duplicates vertices along shared edges, so **every** shell in the sample cell
read as an open surface and scored 0.0 — 0 of 6 panel shells and 0 of the first 200 fixture
shells were watertight, against 3 and 185 once the vertices are welded. The threshold was
therefore never comparing anything, and `--hull-cell-mm` was on or off for a whole link.
Watertightness is now judged on welded indices.

That fix does not change which shells get refined here, and it is worth being clear that
raising `--hull-fill` will not either: with real fills measured, the panel's shells sit at
0.000 and the fixture's at a median of 0.005, so they are far below any threshold worth
setting. Every shell in both is refined at 0.75 already. The threshold is the right control
for solid, well-filled parts — 14 of the gun body's shells and 12 of `robot_link3`'s score
above 0.9 and are correctly left alone — but for panelling the control that matters is the
cell size.

Cell size is set per *category* — `--robot-cell-mm`, `--gun-cell-mm`,
`--tooling-cell-mm`, `--panel-cell-mm`, each falling back to `--hull-cell-mm` when it is
not given. The categories come from the manifest, not from the CAD names: the first device
is the robot, any later one is the gun, and every static object already carries a
`category` of `panel` or `tooling`. That matters because the links do not all deserve the
same treatment:

* **Panelling** is concave precisely where the gun reaches, and wants the smallest cell
  the check time will bear. On the sample panel, `--hull-cell-mm 25` takes it from 6 shells
  to 1958 and `8` takes it to 10052.
* **Tooling** is large and mostly open, and wants refinement near the panels but not the
  triangle budget of the whole frame. Refinement runs *after* `--max-shells`, so the cap
  stops bounding anything once it is on: at 25 mm the sample fixture goes from 1136 shells
  to 45222, and that is paid on every collision check for the rest of the run.
* **The gun body and moving tip** are convex where they come close to anything, so
  refining them buys accuracy nowhere and costs shells everywhere.
* **The rest of the arm** never approaches the parts closely enough for hull error to
  decide anything.

So a sensible split is aggressive on the panels, moderate on the fixture, and off on the
gun and arm:

```
--hull-cell-mm 0 --panel-cell-mm 8 --tooling-cell-mm 60
```

### Splitting panel cells at bends

A cell limits how far a hull can bridge, not whether it does. A cell that happens to hold
the flat stack around a weld and the slope rising out of it hulls to a wedge filling the
corner between them — which is where the electrode goes. On one ST240 panel a 20 mm cell
did exactly that: the gun sat 2 mm inside the hull and 5.3 mm clear of the real sheet.

`--split-panel-bends` (on by default) re-splits the panel pieces inside the weld radius
wherever they bend:

1. A piece's **thickness** is the spread of its vertices along the direction it mostly
   faces (the principal axis of its face normals, sign ignored, so both faces of a sheet
   agree). A hull stands off the sheet by no more than that. A flat 2t stack reads
   1.5–1.9 mm.
2. A piece thicker than `--panel-bend-mm` (3) is halved by which way its faces point, so
   the flat part and the slope separate and the cut lands on the corner, fillets included.
   Where both halves face the same way — a channel's two walls — it is cut across instead,
   at the widest gap. Each half is measured again, at most five levels deep.
3. The fragments are rejoined wherever the union is still flat to `--panel-bend-mm`.

Both faces of the sheet count, because the electrodes close on it from either side. What it
does not catch is bridging within the plane of a flat piece, such as over a hole. Keep
`--panel-bend-mm` above the thickest stack of sheet, or every flat piece reads as bent.

The extra hulls only appear where the sheet bends, so a coarser `--panel-cell-mm` with
bend splitting can cost fewer hulls than a fine cell without it. On the whole ST240 panel,
split on an even grid: 20 mm gave 546 hulls and overlapped the gun; with bend splitting it
gave 892 and cleared it by 5.2 mm; 40 mm with bend splitting gave 505 and cleared it by
3.0 mm.

### Refining only where the geometry gets close

Cell size is a blunt control: it refines a whole link, and the far side of a fixture costs
exactly as much per shell as the face the gun reaches into. Close proximity is a local
business, so refinement can be confined to a radius around the features that matter, with
the cell multiplied by `--far-cell-factor` beyond it — the geometry out there is still
followed, just coarsely.

Both settings rest on the same fact: a source mesh is stored in the coordinates it was
captured in, and the link's own origin is applied in the URDF rather than to the file. So a
world position out of the manifest is directly comparable with a mesh's vertices, and no
transform is involved.

* `--shell-split-weld-prox` focuses **the panels and the tooling** on *the gun, as it sits
  at each weld*. A weld locator is a TCP pose and the gun is rigid with respect to the TCP,
  so the gun's own surface can be placed at every weld and used as the focus. The weld
  point alone is the wrong thing to measure from: the C-frame reaches a long way back past
  the electrodes, so tooling the throat has to swallow sits outside any radius drawn around
  the weld, and that radius refines the wrong side of the part. The radius is therefore a
  distance from the gun's surface and can be much smaller than one drawn from the weld.
  Static objects never move, so those placements stay where they are put.
* `--shell-split-tcp-prox` focuses **the gun body and the moving tip** on the tool centre
  point. The gun and the TCP are rigid with respect to each other, so a radius in the
  capture frame stays meaningful wherever the arm carries them — which is what makes this
  work for a moving link, where a weld position would not. Allow for the electrode stroke
  as well as the approach: the tip travels up to 200 mm relative to the body, so a radius
  measured from the captured TCP has to cover the opening range too. Note also that
  `--gun-cell-mm` defaults to 0, so this does nothing until the gun is being refined at all.

**The arm has no equivalent.** It has no fixed feature worth focusing on, and it is the one
category that never comes near anything, so it refines everywhere — which, at
`--robot-cell-mm 0`, means not at all.

Two details are load-bearing:

* The band one cell wide either side of the radius belongs to *both* sets. Without it the
  near and far hulls meet edge to edge with nothing spanning the seam, which is a hole a
  planner will drive the gun through — the same reason the grid cells overlap.
* `--far-cell-factor 0` makes each far region a single hull. That is coarser still, and a
  hull of everything far can bridge back *across* the near region: safe for collision, but
  it can wall off a legitimate approach to the weld. The default of 4 grades the cell
  instead.

Two things about how the distance itself is measured:

* **It is measured to a triangle's bounding box, not its centroid.** Tessellation produces
  triangles of wildly different sizes — a flat face on a fixture can be two triangles
  metres across — and a centroid says nothing about where such a triangle reaches. On a
  2 m triangle with a corner sitting on the weld, the centroid reads 943 mm and the box
  reads 0. The box distance is a lower bound on the true one, so the error only ever runs
  towards refining something that did not need it.
* **The gun is sampled, and sampling understates closeness.** The cloud is thinned to one
  point per cube of a quarter the radius, and that spacing is added back to the radius, so
  the test stays conservative. The log reports the spacing and the resulting point count.

The radius still has to cover the approach and not just the weld — the linear run into the
weld is exactly where hull error decides something — but since it is measured from the
gun rather than from the weld point, it needs to cover only the clearance either side of
the gun, not the gun's own reach.

The focus points are part of the cache key along with the radius, so moving a weld
re-prepares the geometry refined around the old one rather than silently reusing it. The
key is per link, so widening the gun's radius leaves the panels' cached geometry alone.

The shell counts refinement produces are large, and they are paid for on every collision
check;
`--export-collision-geometry` is the way to see whether the extra shells actually followed
the recess you were after before committing to the runtime.

Each link's cell size and fill threshold are part of its cache key, so changing one link's
settings re-prepares only that link.

Genuinely tighter geometry needs a decomposition that cuts along concavity rather than on
a grid — VHACD or CoACD. Neither ships with `tesseract-robotics`, and this venv has only
numpy, so that would mean taking on a new dependency. Until then the pair-margin
correction is what keeps the false contact from disabling a real collision check.

## Options

A selection; `python main.py --help` lists every flag with its current default.

| Flag | Default | Effect |
| --- | --- | --- |
| `--check-step-deg` | 3 | collision checking resolution along a move, in joint space |
| `--check-step-mm` | 7 | tool-space companion to it; also the station spacing along a linear move |
| `--segment-length-rad` | 0.01 | collision resolution inside the sampling planner |
| `--phase-one-runs` | 25 | OMPL runs per transit, keeping the cheapest that solves |
| `--phase-one-solve-seconds` | 25 | how long one of those runs may search |
| `--cartesian-seconds` | 120 | Cartesian tree budget, tried when phase one finds nothing; 0 disables |
| `--cartesian-recut` | true | cut a Cartesian-tree route where it leaves the band and replan the parts outside it with phases one and two |
| `--phase-two-max-runs` | 8 | further OMPL runs, stopping at the first solution |
| `--phase-two-solve-seconds` | 45 | how long one of those runs may search |
| `--fallback-distance-mm` | 100 | room a fallback pose must leave around the robot and gun |
| `--no-shortcut` | off | skip shortcutting and polishing |
| `--shortcut-seconds` | 20 | time budget for shortcutting each transit |
| `--polish-seconds` | 50 | time budget for the final pass over the emitted waypoints |
| `--no-near-panel-linear` | off | every transit move `PTP`; also disables the Cartesian tree |
| `--near-panel-mm` | 50 | clearance at or under which a waypoint is in the band |
| `--near-panel-min-mm` | 100 | a leg earns `LIN` moves if an unbroken near stretch covers this much tool travel |
| `--near-panel-min-pct` | 60 | ...or if this share of the leg is near |
| `--linear-crossing-penalty-s` | 100 | costing-only surcharge on each move entering the band |
| `--min-shell-mm` | 10 | drop collision shells smaller than this |
| `--max-shells` | 1500 | cap convex shells per link |
| `--hull-cell-mm` | 50 | refine badly-hulled shells into cells this size; 0 disables |
| `--hull-fill` | 0.85 | bounding-box fill below which a shell is refined |
| `--robot-cell-mm` | 0 | cell size for the arm's own links |
| `--gun-cell-mm` | 60 | cell size for the gun body and moving tip |
| `--tooling-cell-mm` | 30 | cell size for static objects the manifest calls tooling |
| `--panel-cell-mm` | 20 | cell size for static objects the manifest calls panel |
| `--split-panel-bends` | true | split panel cells near a weld again wherever the sheet bends; `false` keeps the grid pieces as they are |
| `--panel-bend-mm` | 3 | thickness past which a panel cell counts as bent, and the most its hull can then stand off the sheet |
| `--shell-split-weld-prox` | 60 | refine panels and tooling only this far from the gun at a weld; 0 refines everywhere |
| `--shell-split-tcp-prox` | 175 | refine the gun only this far from the TCP; 0 refines everywhere |
| `--obstacle-clearance-mm` | 5 | clear air to hold from panels and tooling; may be negative |
| `--weld-clearance-mm` | 0 | clearance used instead on moves to or from a weld |
| `--weld-shift-mm` | −5 | shift weld locators along their own z before planning |
| `--no-weld-shift-search` | off | fail a blocked shifted weld rather than search shorter stand-offs |
| `--weld-shift-scan-mm` | 0.5 | first scan spacing of that search; halved while nothing is clear |
| `--weld-shift-resolution-mm` | 0.05 | finest spacing the search refines to |
| `--no-clearance-penalty` | off | avoid only hard collisions, ignoring proximity |
| `--clearance-penalty-max-mm` | 100 | clearance at and above which there is no penalty |
| `--clearance-penalty-min-mm` | 0 | clearance at and below which the penalty peaks |
| `--clearance-penalty-multiplier` | 8 | peak penalty factor |
| `--clearance-penalty-cutoff-mm` | 0 | ignore clearances beyond this; 0 uses the maximum |
| `--stepped-penalty` | on | use the stepped clearance penalty |
| `--stepped-penalty-multiplier` | 7 | stepped penalty at zero clearance |
| `--stepped-penalty-zero-mm` | 35 | clearance at which the stepped penalty reaches 1× |
| `--joint-max-velocity` | 2π/3, J6 11π/9 | per-joint velocity limits, rad/s, comma separated |
| `--joint-max-acceleration` | 2.5, J6 11 | per-joint acceleration limits, rad/s², comma separated |
| `--linear-speed-mm-s` | 250 | tool speed cap on `LIN` moves |
| `--unrefined-output` | off | also write `waypoints-unrefined.json`, pre-optimisation |
| `--probe-point` | off | name the collision hulls containing `LOCATOR:X,Y,Z` and how far the nearest real material is, then stop |
| `--export-collision-geometry` | off | write the hulls actually collided against, and the blocking pair at each unplaceable locator, to `<dir>/collision_geometry/` |
| `--quiet` | off | print only the summary |
| `--write-log` | true | also write everything printed to `weldpath-log.txt` in the study directory |

Exit code is 0 when every segment planned, 1 otherwise.

## Notes and limitations

* Freespace planning is randomised, so a marginal segment can take a different number of
  attempts between runs. Fallback poses searched per transit make this much less likely,
  but a cell with tighter clearances may need `--phase-two-max-runs` raised.
* The Python OMPL bindings do not expose the planner's time budget, so difficulty is
  managed by making collision checks cheaper rather than by planning for longer.
* Convex decomposition overstates penetration where a hull is a poor fit — on the sample
  cell the gun/base pair reads 62.6 mm against 7.1 mm on the exact meshes. Such pairs get
  a corrected pair margin rather than being disabled, so they are still checked, but the
  correction is measured at the start pose only and hull inflation varies with
  configuration. See below for why the geometry itself cannot currently be made tighter.
* Shortcutting is randomised and time-boxed, so the emitted path still varies between
  runs — just over a much lower and tighter range. It shortens; it does not find the
  optimum, and a longer `--shortcut-seconds` keeps helping with diminishing returns.
* **Discrete collision checking can miss thin penetrations.** Moves are checked by sampling
  the straight joint-space segment every `--check-step-deg`, so anything thinner than the
  sample spacing goes unseen. At the default 3° a path on the sample study passed validation
  while actually reaching 7.6 mm *inside* a panel; the same path fails at 2° and finer. A
  positive `--obstacle-clearance-mm` or the clearance penalty both hide this by keeping the
  route away from surfaces, but neither fixes it. The real fix is a swept check — the scene
  already configures `BulletCastBVHManager` as its continuous plugin and nothing uses it.
* The gun opening search covers a handful of candidate openings rather than treating the
  opening as a continuous dimension, so a transit that needs some specific intermediate
  opening will not be found. The manifest supplies no gun speed, so an opening change is
  costed as "avoid unless necessary" rather than in seconds.
* Requires `tesseract-robotics` and `numpy`; both are already in `.venv`.
