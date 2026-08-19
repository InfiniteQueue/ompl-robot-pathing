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
3. **Collision matrix** (`cell.py`). Adjacent pairs are disabled. Any pair that reads as
   in contact in the start pose is then re-measured against the raw concave meshes: a
   pair that merely grazes keeps its collision check under a corrected margin, and only a
   genuine overlap is disabled. Both outcomes are reported in the run log.
4. **Planning** (`planning.py`, `toolpath.py`). A direct joint move is tried first;
   otherwise OMPL, and failing that a two-leg route through the start pose. Each stretch of
   motion is planned with the gun tip where that phase will hold it, and a transit that no
   single opening gets through is split so the gun can change partway. The result is then
   improved under a **time** metric — the same bang-bang joint dynamics the output is
   scheduled with, penalised for running close to the parts — by cutting detours,
   relocating waypoints and dropping the ones that are not worth stopping at.
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
  of waypoints sharing one motion type and one gun state.

A segment may contain **more than one `PTP` phase**. That happens when no single gun opening
gets the robot through the transit, so the move is split and the gun changes at the join —
where the robot is stationary. The consumer has to drive the gun to each phase's
`gun_opening_mm` before executing that phase, which was already implied by the schema
carrying one opening per phase.

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
exactly as the sampling planner returned the solution that was chosen — before shortcutting
and before waypoint reduction. It uses the same schema as `waypoints.json`, and the linear
approach and depart phases are identical in both, since nothing optimises those. Only the
route that was actually kept is recorded; solutions the run discarded leave nothing behind.

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

At a weld locator the robot runs a straight `linear_zone_mm` approach, stops, and departs
along the same line (unless `--no-linear-zone` is given, which removes both). Per the brief the gun is treated as stationary through the weld, and
because the gun state defines a phase this comes out as a phase boundary:

| Phase | motion | contact_allowed | gun_opening_mm |
| --- | --- | --- | --- |
| approach onto the joint | `LIN` | true | `gun_opening_arrive` |
| the weld itself — one waypoint, robot stationary | `LIN` | true | `gun_opening_leave` |
| depart along the same line | `LIN` | true | `gun_opening_leave` |
| transit to the next locator | `PTP` | false | chosen; `gun_opening_leave` preferred |

No motion is planned for the closing itself. The approach ends and the depart begins on
the locator pose, so the single-waypoint weld phase is where the opening changes.

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
opening carried over from the previous locator first, then closed, then wide, then half.
Changing the gun is a real operation on the machine, so a single opening for the whole
transit is always preferred; only when none works is the move split into two legs with the
gun changing at the intermediate pose, where the robot is stationary anyway.

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

**The retract direction is relative to the weld locator, not the tool.** A weld leads in
and out along the locator's own **−x**. This used to be derived from the gun's prismatic
stroke axis, which made the direction a property of the machine rather than of the weld —
and stopped yielding any answer at all once the gun became angular, silently falling back to
tool −z. Override with `--approach-axis`, interpreted in the locator's frame.

Note that `--weld-shift-mm` still backs the tool off along the locator's **z**, so the shift
and the retract are no longer the same axis. If z is not the panel normal for these welds
then the shift is sliding the tool along the surface rather than off it, and wants revisiting
too.

**The linear zone can be turned off** with `--no-linear-zone`. Every millimetre of
`linear_zone_mm` has to be clear along one fixed direction with no freedom to curve, which a
weld set deep in panelling may simply have no room for — the symptom is a segment failing
partway along a 300 mm linear move. Without it the transit runs weld to weld and can curve
away from the panel immediately, at the cost of the gun arriving on a curve rather than
sliding on straight.

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

Genuinely tighter geometry needs a decomposition that cuts along concavity rather than on
a grid — VHACD or CoACD. Neither ships with `tesseract-robotics`, and this venv has only
numpy, so that would mean taking on a new dependency. Until then the pair-margin
correction is what keeps the false contact from disabling a real collision check.

## Options

| Flag | Default | Effect |
| --- | --- | --- |
| `--approach-axis` | `-x` | lead-in/lead-out direction at welds, in the locator's own frame |
| `--no-linear-zone` | off | skip the straight lead-in and lead-out at welds entirely |
| `--linear-step-mm` | 50 | point spacing on linear approach/depart |
| `--check-step-deg` | 3 | collision checking resolution along a move |
| `--segment-length-rad` | 0.02 | collision resolution inside the sampling planner |
| `--ompl-attempts` | 3 | most freespace attempts before the fallback route |
| `--ompl-min-runs` | 3 | sample this many solutions and keep the cheapest |
| `--ompl-seconds` | 5 | how long one sampling-planner run may search |
| `--no-shortcut` | off | emit the sampling planner's own route, unshortened |
| `--shortcut-seconds` | 10 | time budget for shortcutting each transit |
| `--polish-seconds` | 5 | time budget for the final pass over the emitted waypoints |
| `--min-shell-mm` | 5 | drop collision shells smaller than this |
| `--max-shells` | 500 | cap convex shells per link |
| `--hull-cell-mm` | 0 | refine badly-hulled shells into cells this size; 0 disables |
| `--obstacle-clearance-mm` | 0 | clear air to hold from panels and tooling; may be negative |
| `--weld-clearance-mm` | 2 | clearance used instead on moves to or from a weld |
| `--weld-shift-mm` | −5 | shift weld locators along their own z before planning |
| `--no-clearance-penalty` | off | avoid only hard collisions, ignoring proximity |
| `--clearance-penalty-max-mm` | 50 | clearance at and above which there is no penalty |
| `--clearance-penalty-min-mm` | 3 | clearance at and below which the penalty peaks |
| `--clearance-penalty-multiplier` | 50 | peak penalty factor |
| `--clearance-penalty-cutoff-mm` | 0 | ignore clearances beyond this; 0 uses the maximum |
| `--joint-max-velocity` | 2π/3, J6 11π/9 | per-joint velocity limits, rad/s, comma separated |
| `--joint-max-acceleration` | 2.5, J6 11 | per-joint acceleration limits, rad/s², comma separated |
| `--linear-speed-mm-s` | 250 | tool speed cap on `LIN` moves |
| `--unrefined-output` | off | also write `waypoints-unrefined.json`, pre-optimisation |
| `--quiet` | off | print only the summary |

Exit code is 0 when every segment planned, 1 otherwise.

## Notes and limitations

* Freespace planning is randomised, so a marginal segment can take a different number of
  attempts between runs. The fallback route through the start pose makes this much less
  likely, but a cell with tighter clearances may need `--ompl-attempts` raised.
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
