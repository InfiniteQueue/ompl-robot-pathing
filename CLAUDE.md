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
- This is what OMPL produces, and what every move keeps that has neither end in the band.

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
- `_densify_marked` fills the OMPL path in at `--check-step-deg` resolution, measured as the
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
  - **There is no per-leg gate any more.** `_linear_allowed` used to ask, before any move
    was labelled, whether the leg earned linear motion at all -- `--near-panel-min-pct` of
    its samples near, or one unbroken near stretch of `--near-panel-min-mm` tool travel --
    and a leg meeting neither was given a model with no zone, so every move on it was
    joint motion however close it ran. Both flags and the function are gone. The band is
    now the whole of the decision and it is asked per move, so a route that touches the
    band once comes out with the two moves either side of that state linear, where before
    it came out entirely `PTP`. Expect more phases per transit and more short `LIN` runs;
    `--near-panel-mm 0` or `--no-near-panel-linear` is what turns linear motion off.
  - Dropping it took the second sampling pass in `_finish` with it. The fill has to follow
    the curve each move will really be flown along, which needs a model, and the model
    could not exist until the gate had ruled -- so the route was densified once along the
    chord to answer the gate and again along the real curves afterwards. One pass now.
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
- The passes ask one question before checking or pricing anything: `MotionModel.refuses`,
  which is `demotes` or `overreaches`. The two guard the same thing from opposite sides --
  one stops a near-panel stretch being dissolved, the other stops one being grown outward.
- `MotionModel.overreaches` is the `--linear-introduce-mm` limit, default 120 mm. The
  endpoint rule makes a move `LIN` when *either* end is in the band and says nothing about
  where the other end is, so a cut from a state a metre clear straight to one against the
  panel ships as a single straight move whose whole line needs collision-free IK and which
  flies in under the tool speed cap. A replacement that is `LIN` with an end beyond the
  limit is refused **unless the stretch it replaces already had such a move**, which is
  what keeps this a limit on *introducing* linear motion rather than on having it: a
  Cartesian route recut with a long reach still gets shortened, thinned and relocated.
  - `LinearZone.reach_mm` never reads under `near_mm`. A move with both ends in the band is
    linear wherever it runs, so a tighter limit would refuse the reshaping `demotes`
    deliberately allows and freeze every near-panel stretch. 0 switches the rule off, and
    `refuses` is then `demotes` exactly.
  - Measured by `_reads_near` against the limit, so the probe has to see past it:
    `ToolpathPlanner` asks for `max(near_mm, reach_mm) + NEAR_PANEL_HEADROOM_MM`, 145 mm at
    the defaults against 75 mm before. Beyond the probe every state reads the same, so this
    is deliberately a yes-or-no rather than a ranking of one overreach against another --
    there is no measurement to rank with.
  - It gates the passes only. The initial split is untouched: the dense route's states sit
    one check step apart, so a move out of the band into it is short by construction, and
    the reach only appears once a pass lengthens a move.
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
- `Deadline` is the wall clock for one segment, `--segment-minutes` (120). Every other
  budget bounds a part of the search and they multiply -- runs x seconds x openings x
  fallback poses x pairs of openings at each pose -- so this is the only figure that bounds
  a segment with no route. `ran_out` is asked where work is about to start rather than
  interrupting it, since a run stopped halfway leaves nothing usable, and `clamp` cuts each
  per-run limit to the time left. What is already found is still ranked, refined and
  shipped; a segment with nothing raises, and `ToolpathPlanner.run` records the reason and
  goes on, which is what it already did with a segment that failed outright.
- **The gun opening is a search variable, not a setting picked before the search.**
  `_opening_lists` builds two lists per leg: `main`, the rotation phase one and the
  Cartesian tree deal their runs round, and `extra`, held back for phase two. Both are
  drawn in preference order -- departure, arrival, closed, widest, half open, then
  bisections of the widest range nothing has been tried in yet -- and screened as they are
  built, so an opening that repeats one already offered, or that leaves a pose of the leg
  in collision, is skipped and counts against neither total. `--min-gun-openings` and
  `--extra-gun-openings` are counts of openings that can be planned at.
  - Phase one deals its runs round `main`, and `_cheapest` keeps the best of the whole set
    whichever opening it came from, ties going to the earlier one, which is the opening
    already in force. This is why `_Solution` carries an opening: the caller can no longer
    infer it from the order it asked in, and everything downstream -- `_finish`, the recut,
    the per-phase validation -- has to run at the opening its route was found at, the tip
    being part of the machine that has to fit through the gap.
  - Phase two walks `extra`, one opening per run, and never repeats one phase one had runs
    at: a longer search where the short ones just failed is the narrower of the two bets.
    `CartesianBudget.phase_two_seconds` and `phase_two_runs` add short Cartesian searches,
    spread through those runs by `_interleave`, which takes whichever kind's next turn falls
    earliest as a fraction of its own count. The phase stops at the first solution either
    kind finds, so queueing one kind behind the other would let the ordering decide which
    kind ever ran.
  - The rotation belongs to the direct transit alone. A route through a fallback pose is one
    leg whose two halves must agree on one gun state, and so is either half of a two-leg
    split, so those walk `GunOpenings.pinned` one opening at a time -- the old order, kept
    where it is still the only one available. A cell with no gun joint gets `pinned(None)`,
    which is exactly what it did before any of this.
  - `_Sampler._run` loads a pose through `cell.set_state` inside its own `gun_opening`
    block. OMPL reads the gun from the environment's current state and not from the program
    it is handed, and before this each run inherited whatever the last collision query had
    left there -- right in practice only because the endpoint screen always ran first.
- **A weld's two gun openings describe the transits either side of it, not a squeeze.**
  The squeeze is not modelled -- the robot is stationary through the weld -- so a study
  chains them, one weld's `gun_opening_leave` being the next weld's `gun_opening_arrive`,
  which is the same transit read from its two ends. Checked across Paths 2 and 3: every
  pair agrees. Anything that reasons about `leave` as "the gun closing on the part" is
  wrong, and a weld declaring 0/0 is not a weld that does nothing, it is one planned with
  the gun shut throughout.
  - `manifest._opening` parses an absent field as `None`, not `0.0`. The two were the same
    value, so a manifest missing the field planned the weld closed with nothing said, which
    either failed to place or placed somewhere nobody chose. `None` means "nobody said" and
    is what `toolpath._search_opening` fills in.
  - `_search_opening` sweeps the gun's travel for the reachable opening with the most
    clearance, for a weld that declares none and for one whose declared opening places the
    robot nowhere. **Arrival only.** An independent sweep of the departure opening would be
    the same function of the same pose and the same ranking, so it would return the same
    value every time -- and the departure opening is not a geometric question here anyway.
  - The two axes are searched in order, never as one grid: a stand-off is restored before
    export and costs nothing visible, an opening is process data that ships. So the
    declared opening at the declared stand-off, then the declared opening across the
    stand-offs, then other openings at the chosen stand-off, and only then both together --
    one stand-off sweep per opening, coarse grid, since each sample is already a whole
    sweep of the other axis. Searched as one grid, an opening reading a millimetre more
    clearance could displace the study's own choice.
  - `_scan_for_clear` is that sweep, shared by both axes: scan, split every gap between two
    *blocked* samples down to the resolution, then climb from each window's best. Not a
    bisection, because clear is not monotonic in either parameter -- backing a weld off
    frees the tip but can put the throat into tooling, and opening the gun frees the throat
    but swings the electrode into what is beside it.
  - `_plan_pair` writes the departure opening back over the weld phase from what the
    transit out of it was actually solved at, the declared value having only seeded that
    search. Without it the gun changes as the robot starts moving instead of while it
    stands still at the weld. `_arrive_opening` reads `self.openings` for the same reason:
    the declared arrival is a pose the robot may never have been shown to reach.
- `plan_cartesian` is the only search that builds straight tool moves. `_plan_direct`
  runs it on every transit the band is on for, whatever phase one did, and adds its best
  route to the same candidate set -- `_cheapest` then ranks the two searches on one number
  and names the winner's `_Solution.source`. It used to run only where phase one returned
  nothing, which meant a transit phase one solved never had the tree's route costed at all;
  the two differ less in whether they solve than in what they return, uniform joint
  sampling having nothing drawing it into the corridor beside the panel that the tree
  steers along. What it costs is `--cartesian-solve-seconds` plus `--cartesian-min-seconds`
  on every transit rather than only on the ones nothing else reached, bounded by `Deadline`
  alone. Never run for the halves of a fallback-pose route, which are planned with no zone.
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
    out-of-band stretches with long joint chords, so a route found by searching linear
    space can still ship with few linear moves on it.
  - `_fill` hands stations straight through only while consecutive ones are within
    `--check-step-deg`. `jump_rad` allows 0.5 rad per station, and a linear gap wider
    than the step is re-derived by `plan_linear`, which need not reproduce the chain the
    tree validated.
  - `--unrefined-output` writes every transit as one `PTP` phase regardless.
- Order in `_finish` is densify → refine → split → verify, one sampling pass. It was
  sample → gate → densify → … while the gate stood: the fill follows the curve each move
  will really be flown along, which needs a model, and there was no model until the gate
  had said whether the leg was to have a zone, so the route was sampled once along the
  chord for the gate and again along the real curves for the passes. With the band
  answering per move the model exists first and the profile-aware fill is the only fill.
  The split is a reading of the finished path, not a decision imposed before it.
