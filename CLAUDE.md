# Tesseract weldpath — working notes

## Terminology: motion profiles

These two terms describe **how the robot moves between two waypoints**, not how a path was
found. They are properties of a segment, and the planner is free to use different ones on
different parts of the same route.

### Joint profile (`PTP`)

Each joint is driven from its start value to its end value on its own ramp, all finishing
together. Nothing constrains where the tool goes in between: the TCP sweeps an arc whose
shape falls out of the arm's configuration and is not visible anywhere in the cell.

- Timing comes from the joint limits in `weldpath.profile` — `Cell.move_time` /
  `Cell.cruise_time`.
- Collision checking interpolates **in joint space**: `Cell.segment_collides` steps
  `a + t*(b - a)` over the joint vector, which is exactly the path the robot takes.
- This is what OMPL produces and what every unconverted stretch keeps.

### Linear profile (`LIN`)

The TCP is driven in a **straight line in Cartesian space** from the start pose to the end
pose, orientation interpolated alongside it. The controller solves inverse kinematics along
the way, so the joint values in between are whatever that line demands.

- Timing is additionally capped by the commanded tool speed: `output.Timing.phase_times`
  takes `max(joint_move_time, travel / linear_speed)`, with `--linear-speed-mm-s`.
- Collision checking **must follow the Cartesian line**, not the joint chord between the
  same two states. Those are different curves through the cell. For waypoints a
  collision-check step apart the two nearly coincide; for a long chord — anything
  `shortcut` produces — they do not, and a joint-space check on a LIN segment is checking
  a path the robot will not take.
- Inverse kinematics has to exist and be collision free everywhere along the line, or the
  robot faults mid-move. A LIN segment is not merely a labelled joint move: emitting one
  is a claim about the whole line between its endpoints.

## Consequences worth remembering

- `MoveInstructionType_LINEAR` in `tesseract_command_language` is an annotation consumed by
  the simple planner's LVS interpolation, TrajOpt, or Descartes. **OMPL cannot plan linear
  motion** — its only move profile is `OMPLRealVectorMoveProfile`, whose state space is the
  joint vector, so every edge it produces is a joint-space straight line. There is no
  configuration that changes this.
- `plan_linear` is not a planner. It interpolates the pose and runs IK at each step,
  seeding from the previous solution, and takes whatever comes back nearest. It is greedy
  and never backtracks, so it can report "no collision-free IK at 45%" for a line that is
  perfectly reachable from a different starting configuration.
- `_densify` fills the OMPL path in at `--check-step-deg` resolution, measured as the
  **largest joint delta** — not time, not tool distance. OMPL itself returns very few
  points (four is typical); everything downstream runs on the dense list.
- The profile is **not stored** against a waypoint. `MotionModel.motion(a, b)` derives it
  from the two states a move runs between: linear when either end is inside the band. That
  is what lets `shortcut` and `polish` move points around without invalidating anything.
- `shortcut`, `simplify` and `polish` all go through `MotionModel`, so a move destined for
  a linear profile is costed under the tool speed cap and collision checked along the
  straight line the tool will take.
- `MotionModel.demotes` is why the linear stretches survive those passes. Deleting a
  waypoint is nearly always quicker — every retained hop pays its own ramps — so left
  alone the passes collapse a near-panel polyline into one chord whose ends sit outside
  the band, which the endpoint rule then reads as joint motion straight through the region
  the linear profile was chosen for. Replacements that take points out of the band without
  inheriting it are refused.
- `--linear-crossing-penalty` is charged on **crossing** the edge of the band, and on
  **travel** rather than on the whole move. Neither choice is arbitrary, and both were
  arrived at by watching cheaper versions fail.
  - A tax on every linear move cannot decide the question at all. `simplify` weighs a
    chord against the polyline it replaces, whose own near-panel hops are linear too, and
    since each retained hop pays its own ramps the polyline's linear part outweighs the
    chord's — so the chord's share of the tax is the smaller one and the collapse survives
    *any* multiplier. Measured: 10^6 on every linear move left the sweep untouched.
  - Scoped to the crossing but applied to the whole cost, it is still a ratio both sides
    carry, so it converges: ×10 shortened the sweep, ×50 shortened it no further, and the
    entry never reached the band edge. Applied to cruise time only, the ramps stay out of
    it, the two sides no longer travel the same distance outside the band, and the
    preference stops cancelling — ×20 and ×10^4 settle on the same route.
  - It is deliberately unitless and computed without reference to `--linear-speed-mm-s`.
    The two still combine, by `max` in `_time_floor`: a cap slow enough to charge more
    than the surcharge would simply wins. That is the higher of two independent claims,
    not a coupling.
- Order in `_finish` is densify → gate → refine → split. The split is a reading of the
  finished path, not a decision imposed before it.
