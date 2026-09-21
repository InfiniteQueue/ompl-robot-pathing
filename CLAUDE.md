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
  `a + t*(b - a)` over the joint vector, bisecting further wherever the tool moves more
  than `--check-step-mm` between samples, which is exactly the path the robot takes.
- This is what OMPL produces, and what every move keeps that has neither end in the band
  or sits on a leg the gate refused.

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
  - Stations come from `line_stations`: no step travels further than `--check-step-mm`
    **or turns the tool further than `--check-step-deg`**. Travel alone used to decide,
    so a move that mostly reoriented the tool was checked at its two ends only.
  - A station more than `LINE_JUMP_RAD` from the previous one is a branch flip and fails
    the line, as `cartesian._steer` already required of its edges.
  - `linear_chain` pins the chain's ends onto `a` and `b`, but only after
    `_line_end_fault` confirms the tracked line actually arrives there. Pinning blind
    let a line that ends a whole joint 6 turn away, or on another wrist branch, pass
    with the flip hidden in its last joint gap. Measured on Path 1, real lines end
    within 1.1e-4 rad of their target and step at most 0.031 rad per station.
- `_densify` fills the OMPL path in at `--check-step-deg` resolution, measured as the
  **largest joint delta** — not time, not tool distance. OMPL itself returns very few
  points (four is typical); everything downstream runs on the dense list.
  - The fill follows **the curve each move will really be flown along**: the joint
    chord for a joint move, the tool's straight line for a linear one, which `_fill`
    takes from `MotionModel.clear_path`. This used to be the chord in both cases,
    which left the optimisation passes drawing their candidate cut endpoints from
    states the robot never visits. Measured on a transit that withdrew 921 mm to
    cross between two welds 59 mm apart: cuts taken off the real line were clear and
    46% cheaper than the route that shipped, and not one of their endpoints existed
    in the chord fill the pass was given.
  - The linear fill is **thinned back to the same spacing**, by `_resample`'s own
    largest-joint-delta rule walked along the real curve. `plan_linear` stations
    every `--check-step-mm` of tool travel, which is far finer — over a hundred
    points against seven on a metre-long move. Only the curve was wrong, not the
    density, and changing both at once would have made the fix impossible to
    attribute.
  - A linear move that will not validate falls back to the chord. Densifying is not
    where a route is rejected; `_verify_runs` sweeps the finished path and reports
    there.
  - A stretch already inside the spacing is handed straight through, before the model
    is consulted at all. That is the whole of a cartesian-tree route — its stations
    arrive one `--check-step-mm` of tool travel apart, far inside one joint step, and
    are the states it validated. Asking for a chain there would re-derive them, which
    is exactly what that planner keeps its own edge states to avoid, and would build
    hundreds of chains only to be told there is nothing to insert.
  - Which profile applies is asked of the model, not taken from where the path came
    from. Every solver's edges do have a known type, but that is how the edge was
    *planned*; `_split_runs` decides how it will be *flown*, and an OMPL edge with
    both ends in the band ships linear.
- The profile is **not stored** against a waypoint. `MotionModel.motion(a, b)` derives it
  from the two states a move runs between: linear when either end is inside the band. That
  is what lets `shortcut` and `polish` move points around without invalidating anything.
  - Only on a leg `_linear_allowed` admits (`--near-panel-min-pct` of the samples near,
    or one unbroken near stretch of `--near-panel-min-mm` tool travel). A refused leg gets
    a model with no zone, and every move on it is joint motion however close it runs.
  - "Near" is `_reads_near`: `clearance_mm(q) <= near_mm` **and** `< cell.probe_mm`.
    `clearance_mm` returns the probe distance when nothing is in range, so a saturated
    reading is "nothing seen", never a distance. `ToolpathPlanner` asks for a probe of
    `near_mm + NEAR_PANEL_HEADROOM_MM` so the band is always measurable. Both halves are
    needed. Before, the probe was `max(near_mm, obstacle clearance + 25)`, exactly
    `near_mm` at the defaults, and the plain `<=` test put every state in range: logs
    said 100% of every leg, and nearly every move shipped linear.
    `fallback.validity.Validator` widening the shared probe mid-run used to change labels
    too; with the probe already past the band it no longer can.
  - A failed `_verify_runs` discards the route. Nothing re-plans or relabels a stretch
    whose profile cannot be flown.
- `shortcut`, `simplify` and `polish` all go through `MotionModel`, so a move destined for
  a linear profile is costed under the tool speed cap and collision checked along the
  straight line the tool will take.
  - Its clearance penalty is read along that line too: `MotionModel.cost` prices a linear
    move off the factors at the stations `linear_chain` places, through `Cell.price`.
    Until this was fixed it sampled the **joint chord** for both profiles, which nothing
    collision checks and which can pass through the parts. Measured on Path 1's
    73075-05 -> 73075-03 transit: a 707 mm linear move clear by 3.6 mm at worst was priced
    at 54.4 s off a chord reaching 64.5 mm into the panel, against 3.6 s off its line, so
    simplify and polish kept a joint detour up and over rather than take it. Shortcut's
    cuts were never affected; they priced the chain they installed.
  - Still read off the joint chord, before any profile is known: `_path_cost` ranking raw
    solver routes, and `_plan_direct`'s check of whether a clear direct move runs close
    enough to the parts to be worth refining.
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
- `plan_cartesian` is the only search that builds straight tool moves. `_plan_direct`
  reaches it only when OMPL phase one returned nothing and the band is on, and never for
  the halves of a fallback-pose route, which are planned with no zone.
  - `_cartesian_solutions` runs it from successive seeds: until one solves or
    `--cartesian-solve-seconds` passes, then on until `--cartesian-min-seconds` has, both
    timed from the first run. Routes are ranked by `_path_cost`, as phase one's are, and
    only the cheapest is recut, since the recut runs the sampling phases per stretch.
  - It returns every station it validated, about `--check-step-mm` of tool travel apart,
    each gap swept as a joint chord — close enough that chord and line coincide — plus
    one stationary joint move where the two trees meet at the same pose.
  - With `--cartesian-recut` (default on), `_recut_outside_band` then cuts the route
    where it leaves the band, before it becomes a candidate. The first and last state of
    each far stretch stay as waypoints just outside the band; every state between them is
    dropped, and the gap is re-planned by `_Sampler` -- the same phase one and phase two
    the transit gets -- or taken as the joint chord when that is already clear. A gap
    that cannot be joined keeps its tree states. A route with no near state is left
    alone, since that query is the one phase one just failed.
  - Afterwards it is treated exactly like an OMPL route. With the recut off, joint motion
    enters it only through the endpoint rule, and the refinement passes then replace
    out-of-band stretches with long joint chords. A leg the gate refuses ships with no
    linear moves at all, even though it was found by searching linear space.
  - `_fill` hands stations straight through only while consecutive ones are within
    `--check-step-deg`. `jump_rad` allows 0.5 rad per station, and a linear gap wider
    than the step is re-derived by `plan_linear`, which need not reproduce the chain the
    tree validated.
  - `--unrefined-output` writes every transit as one `PTP` phase regardless.
- Order in `_finish` is sample → gate → densify → refine → split → verify. The gate and the
  fill each need the other's answer, so the route is sampled twice: whether the leg
  earns linear motion is a proportion over the route, which a chord sample answers
  perfectly well, and only once that is settled is there a model to say which
  stretches are linear. The second pass is free where it changes nothing — gate
  refused means no zone, every move a joint move, and the profile-aware fill is the
  chord again. The split is a reading of the finished path, not a decision imposed
  before it.
