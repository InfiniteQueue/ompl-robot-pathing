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
- `--linear-crossing-penalty-s` is a **flat** surcharge on each move that reaches into
  the band from outside it — the same figure however long the move. It therefore prices
  how many times the route enters the band and nothing else. This replaced
  `--linear-crossing-speed-mm-s`, which charged the crossing move by its length.
  - What the flat charge does *not* do is the point of it. Deleting a waypoint from
    inside the band crosses the edge once either way, so the charge appears on both sides
    of `simplify`'s comparison and cancels exactly, leaving the decision to ordinary time
    — where one fewer stop is one fewer pair of ramps. Those deletions are wanted: they
    do not move where linear motion begins, which stays at the last waypoint outside the
    band regardless, and the pairs of near-coincident vias straddling the edge cost a
    stop apiece for nothing.
  - What holds a linear sweep back from growing outward is `--linear-speed-mm-s`, and it
    does so structurally rather than by being tuned to. Growing the sweep converts a
    *joint* hop into a linear one, so the cap lands on one side of the comparison and not
    the other and does not cancel. A rearrangement wholly inside the band converts
    nothing, and a straight line is never longer than the polyline it replaces, so the
    cap correctly stays out of it. The length-based charge could not tell the two apart,
    because it read the crossing move's total length, which grows under both.
  - The earlier note here said a multiplier "cannot decide the question at all" and that
    the collapse survives *any* multiplier, measured at 10^6. That holds only where both
    sides of the comparison are wholly linear, which is the in-band case — there the
    factor is exact and cancels at every magnitude. Where the polyline still contains a
    joint hop, which is the sweep-growing case, it lands on one side only: measured on a
    stub fixture, ×1 deletes the edge waypoint and ×10 already refuses to. So the
    original measurement was sound and the generalisation drawn from it was not.
  - It is not scaled by the clearance penalty. That penalty is a multiplier on time spent
    near the parts; a fixed preference is not time, so every caller adds this outside its
    own factor arithmetic — `MotionModel.cost`, `simplify`'s `polyline_cost`, and both
    sides of `shortcut`'s comparison. Charged on the same footing everywhere or it would
    not cancel where it is supposed to.
- Order in `_finish` is densify → gate → refine → split. The split is a reading of the
  finished path, not a decision imposed before it.
