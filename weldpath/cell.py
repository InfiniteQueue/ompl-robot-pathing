"""Tesseract environment construction, collision configuration and kinematics helpers.

Two things here are load-bearing and were established by measurement rather than taste:

* **The collision margin must be negative.**  ``planning.contact_ok_distance_mm`` in the
  manifest is a tolerance for contact, not a safety buffer.  The gun genuinely rests
  against the robot base in the start pose.  With Tesseract's default zero margin the
  start state is invalid and every planner fails immediately -- which is exactly the
  ``freespace transit failed`` recorded in the sample output.
* **Pairs already in contact at the start state are re-measured before being trusted.**
  Convex decomposition over-estimates penetration badly for non-convex castings: at the
  start state the gun and robot base read ~62 mm of overlap against ~7 mm on the exact
  meshes.  Taking that at face value disables the pair for the whole run, which is both
  wrong and unsafe, so each tripped pair is re-checked against the raw concave geometry.
  A pair that merely grazes keeps its collision check and gets a pair-specific margin
  offsetting the measured hull inflation; only a genuine overlap is disabled outright.
* **A collision against the panels or the tooling is never waived.**  Whatever the arm
  and the gun do among themselves, they may not start inside the parts: disabling such a
  pair would blind the planner to it everywhere, so the build aborts instead and says
  which pair and by how much.
"""
from __future__ import annotations

import contextlib
import os

import numpy as np

from tesseract_robotics.tesseract_collision import (
    ContactRequest, ContactResultMap, ContactResultVector, ContactTestType_ALL,
    ContactTestType_FIRST)
from tesseract_robotics.tesseract_common import (
    CollisionMarginPairData, CollisionMarginPairOverrideType_MODIFY, FilesystemPath,
    GeneralResourceLocator, Isometry3d, ManipulatorInfo)
from tesseract_robotics.tesseract_environment import (
    ChangeCollisionMarginsCommand, ChangeJointAccelerationLimitsCommand,
    ChangeJointVelocityLimitsCommand, Environment)
from tesseract_robotics.tesseract_kinematics import KinGroupIKInput, KinGroupIKInputs

from .manifest import Manifest
from .scene import GROUP, TCP_LINK, SceneBuilder

# How many times an interval may be halved when the tool criterion keeps tripping.  The
# joint step already lays down a grid fine enough on its own, so this refines only where
# the tool outruns that grid, and six levels make the tool gap 64x smaller than it gives.
# It is a guard rather than a working limit: near a singularity the tool moves arbitrarily
# far for arbitrarily little joint travel, and without a cap such a segment would subdivide
# until it ran out of floating point.
MAX_TOOL_SUBDIVISION = 6

# What the planner is asked to do, counted so a slow run can say where it went.  These are
# the primitives everything else is built from: a state load pushes joints into the
# environment, a clearance query adds a contact test over every convex piece within the
# probe, and a collision test adds one at the planning margins.
COUNTER_NAMES = ("state_loads", "clearance", "clearance_hits", "collision_tests", "fk")

# States a cell remembers a clearance for before starting again.  Large enough that the
# waypoints of a run stay resident, small enough that a long polish cannot exhaust memory
# on candidates it drew once and threw away.
CLEARANCE_CACHE_MAX = 250_000


class Cell:
    """A loaded scene plus the collision and kinematics helpers the planner needs."""

    def __init__(self, man: Manifest, builder: SceneBuilder, env: Environment,
                 disabled_pairs: list[tuple[str, str, str]]):
        self.man = man
        self.builder = builder
        self.env = env
        self.disabled_pairs = disabled_pairs
        self.margin_overrides: dict[tuple[str, str], float] = {}
        self.margin = 0.0                       # default (self-collision) margin
        self.obstacle_clearance = 0.0           # margin against the static objects
        self.joint_names = man.robot_joint_names
        self.kin = env.getKinematicGroup(GROUP)
        self.manip_info = ManipulatorInfo(GROUP, man.robot.base_link, TCP_LINK)
        limits = np.array(self.kin.getLimits().joint_limits, dtype=float)
        self.lower, self.upper = limits[:, 0], limits[:, 1]
        self._cm = env.getDiscreteContactManager()
        self._cm.setActiveCollisionObjects(env.getActiveLinkNames())
        # Clearance is the dearest query in the planner -- a contact test over every
        # convex piece within the probe, reduced in Python -- and the optimisation passes
        # ask it of the same states repeatedly.  Keyed by the gun opening as well as the
        # joint vector, since where the moving tip sits changes the answer.
        self._clearance_cache: dict[tuple[float, bytes], float] = {}
        self.counters = dict.fromkeys(COUNTER_NAMES, 0)
        self._state = None                      # keeps the SWIG state object alive
        self._revolute = [j.type == "revolute" for j in man.robot.joints
                          if j.type != "fixed"]

        # The gun joint branches off the kinematic chain at the wrist, so it is not an IK
        # variable -- but it is very much a state variable, and leaving it out is what made
        # the tip sit permanently closed regardless of what any phase asked for.
        self.gun_joint_name = man.gun_joint_name
        self.gun_value = 0.0                    # environment units (radians here)
        self._state_names = list(self.joint_names)
        if self.gun_joint_name:
            self._state_names.append(self.gun_joint_name)
            # The manifest states the gun's start position as an opening in millimetres,
            # like every other opening in the file -- the key is even named ..._mm -- while
            # the joint itself is now an angle.  Reading it as a raw joint value put the
            # start state 500-odd radians out and left the tip wherever that landed.
            self.gun_value = man.gun_joint_value(
                float(man.start_state.get(self.gun_joint_name, 0.0)))

        self.penalty = None                     # set by attach_penalty
        self._pm = None                         # proximity manager, only if measured
        self._probe_mm = 0.0                    # how far that manager can actually see
        self._extra_probe_mm = 0.0              # asked for by something other than the penalty
        self.tcp_check_mm = 0.0                 # tool-space check resolution; 0 disables
        self.dynamics = None                    # set by attach_dynamics
        self.weights = self._joint_weights()

    # -- joint metric --------------------------------------------------------
    def _joint_weights(self, delta: float = 1e-4) -> np.ndarray:
        """Millimetres of TCP travel per radian of each joint, at the start pose.

        A radian of J1 swings the whole arm through metres while a radian of J6 barely
        moves the tool, but a planner working in raw joint space treats them as equal --
        which is how a transit ends up circling the base to save a wrist rotation.
        Weighting distances by this makes "short" mean short in the cell, not in the
        joint vector.
        """
        q = np.array([self.man.start_state[n] for n in self.joint_names], dtype=float)
        base = self.fk(q)[:3, 3]
        weights = np.ones(len(self.joint_names))
        for i in range(len(self.joint_names)):
            probe = q.copy()
            probe[i] += delta
            weights[i] = np.linalg.norm(self.fk(probe)[:3, 3] - base) / delta
        # keep the wrist from collapsing to zero cost
        floor = 0.02 * float(weights.max()) if weights.max() > 0 else 1.0
        return np.maximum(weights, floor)

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Weighted joint distance: how far the tool travels, roughly, from a to b."""
        return float(np.linalg.norm((np.asarray(b) - np.asarray(a)) * self.weights))

    # -- time ----------------------------------------------------------------
    def attach_dynamics(self, dynamics) -> None:
        """Give the cell the joint limits that decide how long a move takes."""
        self.dynamics = dynamics

    def move_time(self, a: np.ndarray, b: np.ndarray) -> float:
        """Seconds to go straight from ``a`` to ``b``, starting and ending at rest.

        This is what the robot really does between two emitted waypoints, so it is the
        measure to judge an emitted waypoint by.  Its defining property is that it is
        **not additive**: splitting a move in two costs an extra ramp, and up to 41% more
        again when neither half reaches cruise.  That is exactly the cost of a surplus
        waypoint, and it is invisible to any distance metric.
        """
        if self.dynamics is None:
            return self.distance(a, b)
        return self.dynamics.move_time(a, b)

    def cruise_time(self, a: np.ndarray, b: np.ndarray) -> float:
        """Seconds for ``a`` to ``b`` counting cruise only, ignoring the ramps.

        The ramp-free part of :meth:`move_time`, and the useful thing about it is that it
        **is** additive: subdividing a move leaves it unchanged.  That is what makes it
        the right measure on a densified path, where the intermediate points are an
        artefact of the sampling rather than stops the robot will really make -- scoring
        those with :meth:`move_time` would charge a full ramp every few degrees and value
        the path by how finely it happened to be sampled.

        The two agree in the trapezoidal regime up to one constant ``v/a`` per move, so
        this is a genuine lower bound on the real time rather than a different currency.
        """
        if self.dynamics is None:
            return self.distance(a, b)
        d = np.abs(np.asarray(b, dtype=float) - np.asarray(a, dtype=float))
        return float(np.max(d / self.dynamics.velocity)) if d.size else 0.0

    def wrap_towards(self, q: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Shift each revolute joint by whole turns to sit as close to ``reference``.

        A revolute joint at q and at q + 2*pi put the tool in exactly the same place, and
        J4/J6 here have +/-360 deg of travel, so both representations are usually legal.
        Picking the wrong one costs a full extra revolution of that joint for no gain.
        Joints are independent under this transform, so choosing each one's nearest turn
        is optimal rather than merely greedy.
        """
        out = np.array(q, dtype=float).copy()
        for i in range(len(out)):
            if not self._revolute[i]:
                continue
            best = out[i]
            turns = int(np.ceil((self.upper[i] - self.lower[i]) / (2 * np.pi))) + 1
            for k in range(-turns, turns + 1):
                cand = out[i] + k * 2 * np.pi
                if self.lower[i] - 1e-9 <= cand <= self.upper[i] + 1e-9:
                    if abs(cand - reference[i]) < abs(best - reference[i]):
                        best = cand
            out[i] = best
        return out

    # -- clearance -----------------------------------------------------------
    def _set_obstacle_clearance(self, value: float) -> None:
        _apply_margins(self.env, self.man, self.margin, self.margin_overrides,
                       obstacle_clearance=value)
        self.obstacle_clearance = value
        # The cached manager was built under the old margins, so take a fresh one -- and
        # the proximity manager is a clone of it, so that has to be rebuilt too.
        self._cm = self.env.getDiscreteContactManager()
        self._cm.setActiveCollisionObjects(self.env.getActiveLinkNames())
        self.attach_penalty(self.penalty)
        self._state = None

    @contextlib.contextmanager
    def clearance(self, value: float):
        """Temporarily plan against a different clearance from the static objects.

        Welding is the case that needs this: the gun is meant to close on the panel, so
        the clearance that keeps transits honest would reject the weld itself.  Planners
        read their margins from the environment, so this swaps them there and puts the
        previous value back afterwards.
        """
        previous = self.obstacle_clearance
        if value == previous:
            yield
            return
        self._set_obstacle_clearance(value)
        try:
            yield
        finally:
            self._set_obstacle_clearance(previous)

    # -- gun opening ---------------------------------------------------------
    def set_gun_opening(self, opening_mm: float) -> None:
        """Place the moving electrode for the opening the current phase will hold."""
        if not self.gun_joint_name:
            return
        self.gun_value = self.man.gun_joint_value(opening_mm)
        self._state = None

    @contextlib.contextmanager
    def gun_opening(self, opening_mm: float | None):
        """Plan a stretch of motion with the gun held at a given opening.

        The tip is 145k triangles of geometry swinging through 200 mm, so where it sits
        decides what the robot can fit through -- a transit that is blocked with the gun
        closed can be clear with it open, and the reverse.  ``None`` leaves it alone.
        """
        if opening_mm is None or not self.gun_joint_name:
            yield
            return
        previous = self.gun_value
        self.set_gun_opening(opening_mm)
        try:
            yield
        finally:
            self.gun_value = previous
            self._state = None

    def _state_values(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if not self.gun_joint_name:
            return q
        return np.concatenate([q, [self.gun_value]])

    # -- proximity -----------------------------------------------------------
    def attach_penalty(self, penalty, log=None) -> None:
        """Give the cell a second contact manager dedicated to measuring clearance.

        Margins decide what a manager will even report, and the planning manager's are set
        so that *contact* is the event of interest -- it cannot see a panel 40 mm away.
        Rather than widen those and have every near miss read as a collision, this clones
        the manager and widens only the clone, and only for the robot-against-cell pairs.
        The clone's default margin is driven hugely negative so self-collision pairs stay
        silent and the query cost is confined to the pairs that matter.
        """
        self.penalty = penalty
        wanted = self._extra_probe_mm
        if penalty is not None and penalty.enabled:
            wanted = max(wanted, penalty.probe_mm)
        if log and wanted > 0.0:
            log(f"  cloning the collision scene into a clearance manager that can see "
                f"{wanted:g} mm")
        self._build_proximity(wanted)

    def require_proximity(self, probe_mm: float, log=None) -> None:
        """Make sure clearance can be measured out to ``probe_mm``, penalty or no penalty.

        The penalty owns how far it needs to see, but it is not the only caller that wants
        a distance -- and it can be switched off entirely, which must not take the
        measurement with it.  This raises the probe if it has to and leaves it alone
        otherwise, so asking twice costs nothing.
        """
        self._extra_probe_mm = max(self._extra_probe_mm, float(probe_mm))
        if self._extra_probe_mm > self._probe_mm:
            if log:
                log(f"  building a clearance manager that can see {self._extra_probe_mm:g} "
                    f"mm; it clones the collision scene and sets a margin on every "
                    f"robot-to-part pair")
            self._build_proximity(self._extra_probe_mm)

    def _build_proximity(self, probe_mm: float) -> None:
        self._pm = None
        self._probe_mm = 0.0
        if probe_mm <= 0.0:
            return
        probe = probe_mm * self.man.scale
        pm = self._cm.clone()
        pm.setDefaultCollisionMargin(-1.0)
        for a, b in _obstacle_margins(self.man):
            pm.setCollisionMarginPair(a, b, probe)
        pm.setActiveCollisionObjects(self.env.getActiveLinkNames())
        self._pm = pm
        self._probe_mm = float(probe_mm)
        # Readings taken against the old manager, at whatever probe it had, do not carry.
        self._clearance_cache.clear()

    @property
    def probe_mm(self) -> float:
        """How far :meth:`clearance_mm` can actually see, in mm.

        A clearance at or beyond this is the query running out of range, not a measurement,
        and anything reporting a distance has to be able to say which of the two it has.
        """
        return self._probe_mm

    def clearance_mm(self, q: np.ndarray) -> float:
        """Closest approach between the robot or gun and the static objects, in mm.

        Returns the probe distance when nothing is within it, which is all the penalty
        needs: beyond that the cost is flat, so the exact figure does not matter.
        """
        if self._pm is None:
            return float("inf")
        key = (self.gun_value, np.asarray(q, dtype=float).tobytes())
        hit = self._clearance_cache.get(key)
        if hit is not None:
            self.counters["clearance_hits"] += 1
            return hit
        self.counters["clearance"] += 1
        # Deliberately not set_state: refreshing the planning manager's transforms costs as
        # much as the query itself and nothing here is going to ask it anything.
        self._load_state(q)
        got = self._clearance_now()
        if len(self._clearance_cache) >= CLEARANCE_CACHE_MAX:
            # Relocation candidates are drawn at random and never revisited, so the cache
            # grows with states that will never be asked for again.  Dropping the lot is
            # fine: what matters is the waypoints, and they are re-measured on demand.
            self._clearance_cache.clear()
        self._clearance_cache[key] = got
        return got

    def _clearance_now(self) -> float:
        """Clearance at the state already loaded.  Assumes ``set_state`` has just run."""
        self._pm.setCollisionObjectsTransform(self._state.link_transforms)
        res = ContactResultMap()
        self._pm.contactTest(res, ContactRequest(ContactTestType_ALL))
        if res.size() == 0:
            return self._probe_mm
        vec = ContactResultVector()
        res.flattenCopyResults(vec)
        worst = min((float(c.distance) for c in vec), default=None)
        if worst is None:
            return self._probe_mm
        return worst / self.man.scale

    def penalty_factor(self, q: np.ndarray) -> float:
        """Cost multiplier for standing where ``q`` puts the robot."""
        if self._pm is None or self.penalty is None:
            return 1.0
        return self.penalty.factor(self.clearance_mm(q))

    # -- state / collision ---------------------------------------------------
    def _load_state(self, q: np.ndarray) -> None:
        self.counters["state_loads"] += 1
        self.env.setState(self._state_names, self._state_values(q))
        # Must hold a reference: passing env.getState().link_transforms inline lets the
        # temporary die and the binding then reads freed memory.
        self._state = self.env.getState()

    def set_state(self, q: np.ndarray) -> None:
        self._load_state(q)
        self._cm.setCollisionObjectsTransform(self._state.link_transforms)

    def in_collision(self, q: np.ndarray) -> bool:
        self.counters["collision_tests"] += 1
        self.set_state(q)
        res = ContactResultMap()
        self._cm.contactTest(res, ContactRequest(ContactTestType_FIRST))
        return res.size() > 0

    def contact_pairs(self, q: np.ndarray) -> dict[tuple[str, str], float]:
        return {pair: d for pair, (d, _) in self.contacts(q).items()}

    def contacts(self, q: np.ndarray) -> dict[tuple[str, str], tuple[float, np.ndarray]]:
        """Worst contact per pair, with the world point where it was measured.

        The point is the midpoint of the contact's two nearest points, which for anything
        close enough to be worth reporting are within a millimetre or two of each other.
        It answers "where on the gun is this happening", which the distance alone does not.
        """
        self.set_state(q)
        res = ContactResultMap()
        self._cm.contactTest(res, ContactRequest(ContactTestType_ALL))
        vec = ContactResultVector()
        res.flattenCopyResults(vec)
        worst: dict[tuple[str, str], tuple[float, np.ndarray]] = {}
        for c in vec:
            key = tuple(sorted((c.link_names[0], c.link_names[1])))
            d = float(c.distance)
            if key not in worst or d < worst[key][0]:
                worst[key] = (d, _contact_point(c))
        return worst

    def in_tcp_frame(self, q: np.ndarray, point_world: np.ndarray) -> np.ndarray:
        """A world point expressed in the TCP's own frame, in manifest units.

        The TCP frame is the one worth quoting a contact in: it is where the operator is
        looking, and its axes are the ones the approach and the lead-in are defined along.
        """
        T = self.fk(q)
        local = T[:3, :3].T @ (np.asarray(point_world, dtype=float) - T[:3, 3])
        return local / self.man.scale

    def segment_collides(self, a: np.ndarray, b: np.ndarray, max_step: float = 0.05) -> bool:
        """Discretely check the straight joint-space segment a->b.

        ``max_step`` sets the base resolution in joint space, which is the whole of the
        test when :attr:`tcp_check_mm` is 0.  A joint step is a poor proxy for how far the
        gun actually goes, though: the same few degrees are millimetres at the wrist and a
        hand's breadth at the base, so a step fine enough for the one over-checks the
        other by an order of magnitude and a step sized for the base can carry the gun
        clean through a fixture between two samples.

        With :attr:`tcp_check_mm` set, both are asked at once and either may trip.  The
        joint step lays down the grid as before, then any interval whose ends are further
        apart than that at the tool is halved and rechecked, and its halves are judged the
        same way.  So the test is never weaker than the joint step alone, it costs nothing
        on the stretches where the two agree, and the samples land where the gun is really
        moving rather than where the numbers happen to be large.

        The tool position is free: :meth:`in_collision` has already loaded the state, so
        reading the transform off it costs no kinematics.
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        delta = b - a
        span = float(np.max(np.abs(delta)))
        n = max(2, int(np.ceil(span / max_step)) + 1)
        grid = np.linspace(0.0, 1.0, n)

        if self.tcp_check_mm <= 0.0:
            for t in grid:
                if self.in_collision(a + t * delta):
                    return True
            return False

        tool_step = self.tcp_check_mm * self.man.scale
        tools = []
        for t in grid:
            if self.in_collision(a + t * delta):
                return True
            tools.append(self._tcp_now())

        # Right to left, so the halves of a split are popped in the order they are flown.
        stack = [(grid[k], grid[k + 1], tools[k], tools[k + 1], 0)
                 for k in range(n - 2, -1, -1)]
        while stack:
            t0, t1, p0, p1, depth = stack.pop()
            if depth >= MAX_TOOL_SUBDIVISION:
                continue
            if float(np.linalg.norm(p1 - p0)) <= tool_step:
                continue
            tm = 0.5 * (t0 + t1)
            if self.in_collision(a + tm * delta):
                return True
            pm = self._tcp_now()
            stack.append((tm, t1, pm, p1, depth + 1))
            stack.append((t0, tm, p0, pm, depth + 1))
        return False

    def segment_cost(self, a: np.ndarray, b: np.ndarray, max_step: float = 0.05,
                     fa: float | None = None, fb: float | None = None,
                     stops: bool = False) -> float:
        """Penalised time for the segment a->b, assuming it is already known clear.

        Time near a panel is charged at :class:`~weldpath.penalty.ClearancePenalty`'s
        multiplier, which is what the penalty was specified in: a second at the minimum
        clearance costs as much as N seconds in open space.

        ``stops`` picks which time this is.  False measures cruise only, for a move that is
        one step of a densified path the robot will not really stop along; True measures
        the full move, ramps included, for a move between two waypoints that will be
        emitted.  See :meth:`move_time` and :meth:`cruise_time`.

        Loading a joint state and refreshing the collision transforms costs far more than
        the geometry query on top of it, so the interior samples are walked once and the
        clearance is read off the state that is already loaded.  ``fa``/``fb`` let a caller
        that has already measured the endpoints avoid paying for them twice, which is the
        common case when a path is being rescored.
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        raw = self.move_time(a, b) if stops else self.cruise_time(a, b)
        if self._pm is None or raw <= 0.0:
            return raw
        n = max(2, int(np.ceil(np.max(np.abs(b - a)) / max_step)) + 1)
        factors = []
        for k, t in enumerate(np.linspace(0.0, 1.0, n)):
            if k == 0 and fa is not None:
                factors.append(fa)
            elif k == n - 1 and fb is not None:
                factors.append(fb)
            else:
                factors.append(self.penalty_factor(a + t * (b - a)))
        # Each sub-step is charged at the worse of the states it runs between, so a dip
        # towards the panel is never averaged away by the clear air on either side.  The
        # move's time is spread evenly over the sub-steps rather than following the ramps,
        # which slightly under-charges the ends of a move; where the route is close to the
        # parts it is close for a stretch, not at a single instant, so the shape of the
        # penalty across a move matters much less than its total.
        step = raw / (n - 1)
        return sum(step * max(x, y) for x, y in zip(factors, factors[1:]))

    def within_limits(self, q: np.ndarray) -> bool:
        return bool(np.all(q >= self.lower - 1e-9) and np.all(q <= self.upper + 1e-9))

    # -- kinematics ----------------------------------------------------------
    def _tcp_now(self) -> np.ndarray:
        """Where the tool is for the state already loaded, in environment units.

        :meth:`fk` answers the same question but pushes the state first, which is the
        expensive half.  This is for callers that have just collision-checked a state and
        so have already paid for it.
        """
        return np.array(self.env.getLinkTransform(TCP_LINK).matrix(),
                        dtype=float)[:3, 3]

    def fk(self, q: np.ndarray, link: str = TCP_LINK) -> np.ndarray:
        self.counters["fk"] += 1
        self.env.setState(self._state_names, self._state_values(q))
        self._state = self.env.getState()
        return np.array(self.env.getLinkTransform(link).matrix(), dtype=float)

    def ik(self, pose_world_mm: np.ndarray, seed: np.ndarray) -> list[np.ndarray]:
        """Inverse kinematics for a world TCP pose given in manifest units."""
        pose = np.array(pose_world_mm, dtype=float).copy()
        pose[:3, 3] *= self.man.scale
        req = KinGroupIKInput()
        req.pose = Isometry3d(pose)
        req.working_frame = self.man.robot.base_link
        req.tip_link_name = TCP_LINK
        inputs = KinGroupIKInputs()
        inputs.append(req)
        try:
            raw = self.kin.calcInvKin(inputs, np.asarray(seed, dtype=float))
        except Exception:
            return []
        n = len(self.joint_names)
        # Elements come back as 1-element arrays. Flatten explicitly: a (n, 1) column
        # would silently broadcast in the joint-limit comparison and reject every solution.
        def scalar(v) -> float:
            return float(np.asarray(v).reshape(-1)[0])

        return [np.array([scalar(raw[i][j]) for j in range(n)], dtype=float)
                for i in range(len(raw))]

    def solve_pose(self, pose_world_mm: np.ndarray, seeds: list[np.ndarray],
                   require_collision_free: bool = True, branch_seeds: int = 8,
                   rng: np.random.Generator | None = None) -> np.ndarray | None:
        """Best joint solution for a pose: in limits, collision free, nearest the seed.

        KDL's solver is a local method returning one solution per seed, so the branch it
        lands on is whatever the seed was closest to.  Extra scattered seeds expose the
        other arm configurations, every candidate is then wrapped onto its nearest
        equivalent turn, and the winner is chosen by weighted distance so a solution is
        only preferred if it genuinely moves the tool less.
        """
        reference = np.asarray(seeds[0], dtype=float)
        rng = rng or np.random.default_rng(0)
        all_seeds = list(seeds)
        if branch_seeds:
            span = self.upper - self.lower
            for _ in range(branch_seeds):
                all_seeds.append(np.clip(reference + rng.uniform(-0.35, 0.35) * span,
                                         self.lower, self.upper))

        best, best_cost = None, np.inf
        for seed in all_seeds:
            for q in self.ik(pose_world_mm, seed):
                q = self.wrap_towards(q, reference)
                if not self.within_limits(q):
                    continue
                cost = self.distance(reference, q)
                if cost >= best_cost:
                    continue                    # cheaper test before the collision check
                if require_collision_free and self.in_collision(q):
                    continue
                best, best_cost = q, cost
        return best


def hull_categories(man: Manifest) -> dict[str, str]:
    """Link name -> the category whose cell size applies to it.

    The categories -- robot, gun, tooling, panel -- come from the manifest: the first
    device is the robot, any later one is the gun, and every static object already carries
    a category.  So nothing here depends on how the CAD happens to be named.
    """
    out = {link.name: "robot" for link in man.robot.links}
    for device in man.devices[1:]:
        out.update({link.name: "gun" for link in device.links})
    out.update({s.name: s.category for s in man.static_objects})
    return out


def hull_cells(man: Manifest, hull_cell: float,
               per_category: dict[str, float | None] | None = None) -> dict[str, float]:
    """Link name -> the cell size to refine its shells at, in manifest units.

    Refinement is worth paying for where the geometry is concave *at the point of closest
    approach*: the panels the gun reaches into, and the tooling around them.  It is wasted
    on the arm, which never comes near anything, and on the gun, which is convex where it
    matters -- and since refinement runs after ``--max-shells``, spending it on a large
    part is what takes a 1136-shell fixture to 45222 and slows every later check.
    """
    chosen = per_category or {}
    return {name: float(chosen.get(category) if chosen.get(category) is not None
                        else hull_cell)
            for name, category in hull_categories(man).items()}


def weld_points(man: Manifest) -> np.ndarray:
    """Every weld locator's position, in the frame the static meshes are stored in.

    The static objects are placed at the world origin (see ``SceneBuilder._compute_frames``)
    and their meshes are exported in world coordinates, so a locator's world position is
    directly comparable with their vertices.  That is what makes a proximity test possible
    without a transform -- and why it is offered for the statics only, the arm and the gun
    being captured in one pose and then moving away from it.
    """
    return np.array([loc.pose_world[:3, 3] for loc in man.locators if loc.is_weld],
                    dtype=float).reshape(-1, 3)


# The gun cloud is sampled at a fraction of the radius being tested, floored so a tight
# radius cannot ask for an unbounded number of points, and the whole cloud is thinned again
# if it still comes out too large to measure a 2M-triangle fixture against.
GUN_SAMPLE_FRACTION = 0.25
GUN_SAMPLE_FLOOR = 10.0
MAX_FOCUS_POINTS = 4000


def _thin(P: np.ndarray, spacing: float) -> np.ndarray:
    """One point per ``spacing``-sized cube, keeping the first that lands in each."""
    if spacing <= 0.0 or not len(P):
        return P
    _, idx = np.unique(np.floor(P / spacing).astype(np.int64), axis=0, return_index=True)
    return P[np.sort(idx)]


def gun_cloud(man: Manifest, spacing: float) -> np.ndarray:
    """The gun's surface, thinned, expressed in the TCP's frame in manifest units.

    Read from the source meshes rather than the prepared ones: this is a question about
    where the metal is, and the prepared file is a decomposition of it whose hulls may sit
    slightly proud of the surface.
    """
    from . import meshprep

    T = np.array(man.tcp_world_pose, dtype=float)
    R, t = T[:3, :3], T[:3, 3]
    out: list[np.ndarray] = []
    for device in man.devices[1:]:
        for link in device.links:
            if not link.mesh:
                continue
            try:
                V, _ = meshprep.load_obj(man.mesh_path(link.mesh))
            except (OSError, ValueError):
                continue
            if len(V):
                out.append(_thin((V - t) @ R, spacing))
    return np.concatenate(out) if out else np.zeros((0, 3))


def gun_at_welds(man: Manifest, radius: float, log=None) -> tuple[np.ndarray, float]:
    """Where the gun sits when it is at a weld, as points, with the radius to test it at.

    A weld locator is a TCP pose, and the gun is rigid with respect to the TCP, so placing
    the gun's own surface at every weld says where the metal will actually be -- which is
    the question the panels and the tooling need answered.  The weld point alone does not
    answer it: the C-frame reaches a long way back past the electrodes, so tooling that the
    gun's *throat* will pass through sits well outside any radius drawn around the weld,
    and refining by that radius refines the wrong side of the part.

    The returned radius is the caller's plus the sampling spacing.  Sampling a surface can
    only ever understate how close it comes, and that margin covers the understatement, so
    the test stays conservative.
    """
    spacing = max(radius * GUN_SAMPLE_FRACTION, GUN_SAMPLE_FLOOR)
    welds = [loc for loc in man.locators if loc.is_weld]
    local = gun_cloud(man, spacing)
    if not len(local) or not welds:
        if log:
            log("  ! no gun geometry to place at the welds; falling back to the weld "
                "points themselves")
        return weld_points(man), radius
    placed = np.concatenate([np.array(loc.pose_world, dtype=float)[:3, :3] @ local.T
                             + np.array(loc.pose_world, dtype=float)[:3, 3:4]
                             for loc in welds], axis=1).T
    points = _thin(placed, spacing)
    while len(points) > MAX_FOCUS_POINTS:
        spacing *= 1.5
        points = _thin(points, spacing)
    if log:
        log(f"  the gun sampled at {spacing:g} mm and placed at {len(welds)} weld"
            f"{'' if len(welds) == 1 else 's'} gives {len(points)} focus points for the "
            f"panels and tooling")
    return points, radius + spacing


def refinement_focus(man: Manifest, weld_proximity: float, tcp_proximity: float,
                     log=None) -> dict[str, tuple[np.ndarray, float]]:
    """Link -> the points its hull refinement should concentrate around, and the radius.

    Both entries rest on the same fact: a source mesh is stored in the coordinates it was
    captured in, and the link's own origin is applied in the URDF rather than to the file.
    So a world position from the manifest is directly comparable with a mesh's vertices,
    and no transform is involved.

    * **Panels and tooling** focus on the gun as it sits at each weld -- see
      :func:`gun_at_welds`.  They never move, so the capture frame is the world frame and
      those placements stay where they are put.
    * **The gun** focuses on the TCP.  The gun and the tool centre point are rigid with
      respect to each other, so a radius in the capture frame stays meaningful wherever the
      arm carries them -- which is what makes this work for a moving link, where a weld
      position would not.  The one caveat is the electrode stroke: the tip travels relative
      to the body, so a radius around the captured TCP has to be generous enough to cover
      the opening range as well as the approach.

    The arm is deliberately absent.  It has no fixed feature to focus on, and it is the one
    category that never comes close to anything.
    """
    out: dict[str, tuple[np.ndarray, float]] = {}
    if weld_proximity > 0.0:
        points, radius = gun_at_welds(man, weld_proximity, log=log)
        if len(points):
            out.update({s.name: (points, radius) for s in man.static_objects})
    if tcp_proximity > 0.0:
        tcp = np.array(man.tcp_world_pose, dtype=float)[:3, 3].reshape(1, 3)
        for device in man.devices[1:]:
            out.update({link.name: (tcp, tcp_proximity) for link in device.links})
    return out


def _resolve_collision_meshes(man: Manifest, log, min_extent: float, max_shells: int,
                              hull_cell: float, hull_fill: float,
                              hull_per_category, weld_proximity: float = 0.0,
                              tcp_proximity: float = 0.0,
                              far_cell_mm: float = 0.0,
                              far_per_category=None,
                              hull_overlap: float | None = None,
                              merge_cell_mm: float = 0.0, merge_per_category=None,
                              enclosed_per_category=None,
                              enclosed_probe_mm: float = 0.0,
                              enclosed_voxel_mm: float = 0.0,
                              enclosed_keep_mm: float = 0.0,
                              enclosed_dump: bool = False) -> dict[str, str]:
    from . import meshprep

    overlap = meshprep.DEFAULT_OVERLAP if hull_overlap is None else float(hull_overlap)

    rel: dict[str, str] = {}
    for link in man.all_links():
        if link.mesh:
            rel[link.name] = link.mesh
    for s in man.static_objects:
        if s.mesh:
            rel[s.name] = s.mesh
    log("preparing collision geometry (convex decomposition):")
    focus = refinement_focus(man, weld_proximity, tcp_proximity, log=log)
    # Only the categories that actually have geometry, so the line names what is there.
    present = set(hull_categories(man).values())
    far_resolved = {c: float((far_per_category or {}).get(c) if
                             (far_per_category or {}).get(c) is not None
                             else far_cell_mm)
                    for c in present}
    if focus:
        if weld_proximity > 0.0:
            log(f"  panels and tooling refined within {weld_proximity:g} mm of the gun "
                f"at a weld")
        if tcp_proximity > 0.0:
            log(f"  the gun refined within {tcp_proximity:g} mm of the tool centre point")
        log("  elsewhere: " + ", ".join(
            f"{category} {size:g} mm" if size > 0 else f"{category} one hull"
            for category, size in sorted(far_resolved.items())))
    if hull_cell > 0.0:
        log(f"  cells claim triangles {overlap:g} of a cell past their own bounds, so a "
            f"cell's hull spans about {1.0 + 2.0 * overlap:.2f}x the cell")
    # Reuses the cell resolver because the question has the same shape -- a per-category
    # figure over a global one -- but the global default is 0, so a category nobody named
    # is left alone.  This filter deletes geometry, and switching it on for a link is a
    # decision about that link.
    merges = {k: v for k, v in
              hull_cells(man, merge_cell_mm, merge_per_category).items() if v > 0}
    if merges:
        by_cat: dict[str, float] = {}
        for name, value in merges.items():
            by_cat[hull_categories(man)[name]] = value
        log("  beyond the focus, one hull per cell across whatever solids fall in it: "
            + ", ".join(f"{c} {v:g} mm" for c, v in sorted(by_cat.items()))
            + " -- the only step that can bring two shells together, and the only one "
              "that claims space rather than giving it up")
    probes = {k: v for k, v in
              hull_cells(man, enclosed_probe_mm, enclosed_per_category).items() if v > 0}
    if probes:
        by_category: dict[str, float] = {}
        categories = hull_categories(man)
        for name, value in probes.items():
            by_category[categories[name]] = value
        log("  dropping shells no probe can reach from outside the link: "
            + ", ".join(f"{c} {v:g} mm" for c, v in sorted(by_category.items()))
            + f", screened on a {enclosed_voxel_mm:g} mm voxel"
            + (f", never above {enclosed_keep_mm:g} mm across" if enclosed_keep_mm > 0
               else ", with no size backstop"))
    return meshprep.prepare(man.directory, rel, man.scale, log=log,
                            min_extent=min_extent, max_shells=max_shells,
                            hull_cell=hull_cell, fill=hull_fill,
                            cells=hull_cells(man, hull_cell, hull_per_category),
                            focus=focus, far_cell=far_cell_mm,
                            far_cells=hull_cells(man, far_cell_mm, far_per_category),
                            overlap=overlap, merge_cells=merges,
                            merge_cell=merge_cell_mm, enclosed_probes=probes,
                            enclosed_voxel=enclosed_voxel_mm,
                            enclosed_keep=enclosed_keep_mm,
                            enclosed_dump=enclosed_dump)


# Probe distance for the exact re-measurement.  Generous on purpose: a pair the hulls
# call a deep crash must come back with a real number, not "nothing found".
EXACT_PROBE_MARGIN = 0.05
# Probe distances to try, widest first. Narrowing costs accuracy but never safety: a pair
# nothing is found near is treated as clear at the probe used, so a smaller probe claims
# less clearance, not more.
EXACT_PROBE_STEPS = (EXACT_PROBE_MARGIN, 0.01, 0.002)


def _geometry_extent(geom) -> np.ndarray:
    """Bounding-box size, in metres, of a loaded collision geometry."""
    meshes = geom.getMeshes() if hasattr(geom, "getMeshes") else [geom]
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for mesh in meshes:
        verts = mesh.getVertices()
        n = len(verts)
        # Stride large meshes: this is a sanity check on scale, where the error being
        # caught is a factor of 1000, not a few percent.
        step = max(1, n // 2000)
        pts = np.array([np.asarray(verts[i], dtype=float).ravel()
                        for i in range(0, n, step)])
        lo = np.minimum(lo, pts.min(axis=0))
        hi = np.maximum(hi, pts.max(axis=0))
    return hi - lo


def _assert_scale(env: Environment, man: Manifest, name: str, rel_mesh: str) -> None:
    """Confirm a link's loaded geometry matches its source OBJ in size.

    A mesh authored in millimetres but loaded without a scale lands 1000x too large and
    kilometres away, which reports a confident and completely wrong "no contact".  That
    has burned this code once already, so the exact measurement refuses to run unless the
    geometry it is about to trust is the size the OBJ says it should be.
    """
    from . import meshprep

    verts, _ = meshprep.load_obj(man.mesh_path(rel_mesh))
    want = (verts.max(axis=0) - verts.min(axis=0)) * man.scale
    got = _geometry_extent(env.getLink(name).collision[0].geometry)
    if np.linalg.norm(want) <= 0:
        raise RuntimeError(f"source mesh for '{name}' is degenerate")
    err = float(np.linalg.norm(got - want) / np.linalg.norm(want))
    if err > 0.1:
        raise RuntimeError(
            f"exact geometry for '{name}' loaded at the wrong scale: "
            f"extent {got} m against {want} m from the source mesh")


def _exact_contacts(man: Manifest, collision: dict[str, str], out_dir: str,
                    pairs: list[tuple[str, str]], q: np.ndarray,
                    joint_names: list[str], log=print) -> dict[tuple[str, str], float]:
    """Measure ``pairs`` at joint state ``q`` against the raw concave meshes.

    One pair at a time, and each pair's scene holds only the two links involved.  Loading
    every tripped pair's geometry at once is what made this fall over on a cell with a
    1.6M-triangle fixture: raw concave meshes are enormous, and Bullet's mesh-against-mesh
    test at a wide probe generates a contact manifold per candidate triangle pair, so the
    allocation grows with both mesh size and probe distance.

    The probe is therefore also allowed to back off.  Narrowing it only ever makes the
    answer more conservative -- a pair the probe cannot see is treated as clear at the
    probe distance, so a smaller probe claims *less* clearance than a larger one -- which
    makes retrying at 50, 10 and 2 mm a safe escalation rather than a compromise.

    Returns the worst distance per pair, negative for penetration.  A pair absent from the
    result was not measurable: either nothing lay within the probe, or the measurement
    could not be made at all, and the caller decides what to do about that.
    """
    out: dict[tuple[str, str], float] = {}
    for i, (a, b) in enumerate(pairs, 1):
        log(f"    . {i}/{len(pairs)} measuring {a} <-> {b}")
        for probe in EXACT_PROBE_STEPS:
            try:
                worst = _exact_pair(man, collision, out_dir, a, b, q, joint_names, probe)
            except Exception as exc:                # SystemError: bad allocation, etc.
                log(f"  ! exact re-measurement of {a} <-> {b} ran out of room at a "
                    f"{probe * 1000:.0f} mm probe ({type(exc).__name__}); retrying closer")
                continue
            # Measured cleanly but found nothing: the pair is clear by at least the probe
            # used, which is all that can honestly be claimed.
            out[(a, b)] = worst if worst is not None else probe
            break
        else:
            log(f"  ! cannot re-measure {a} <-> {b} against the raw meshes at any probe "
                f"distance; leaving the pair on its default margin")
    return out


def _exact_pair(man: Manifest, collision: dict[str, str], out_dir: str,
                a: str, b: str, q: np.ndarray, joint_names: list[str],
                probe: float) -> float | None:
    """Worst raw-geometry distance between two links, or None if nothing is within probe."""
    links = sorted({a, b})
    builder = SceneBuilder(man, collision, exact_links=set(links))
    ex_dir = os.path.join(out_dir, "exact_check")
    os.makedirs(ex_dir, exist_ok=True)

    paths = {}
    for stem, text in (("cell.urdf", builder.urdf()),
                       ("kinematics_plugins.yaml", builder.kinematics_plugins_yaml()),
                       ("contact_managers_plugins.yaml", builder.contact_managers_yaml())):
        paths[stem] = os.path.join(ex_dir, stem).replace("\\", "/")
        with open(paths[stem], "w", encoding="utf-8") as fh:
            fh.write(text)
    paths["cell.srdf"] = os.path.join(ex_dir, "cell.srdf").replace("\\", "/")
    with open(paths["cell.srdf"], "w", encoding="utf-8") as fh:
        fh.write(builder.srdf(builder.adjacent_pairs(), paths["kinematics_plugins.yaml"],
                              paths["contact_managers_plugins.yaml"]))

    env = Environment()
    if not env.init(FilesystemPath(paths["cell.urdf"]), FilesystemPath(paths["cell.srdf"]),
                    GeneralResourceLocator()):
        raise RuntimeError(f"failed to load the exact-geometry scene in {ex_dir}")
    env.applyCommand(ChangeCollisionMarginsCommand(probe))

    raw = {l.name: l.mesh for l in man.all_links() if l.mesh}
    raw.update({s.name: s.mesh for s in man.static_objects if s.mesh})
    for name in links:
        _assert_scale(env, man, name, raw[name])

    cm = env.getDiscreteContactManager()
    for name in cm.getCollisionObjects():
        if name in (a, b):
            cm.enableCollisionObject(name)
        else:
            cm.disableCollisionObject(name)
    cm.setActiveCollisionObjects([a, b])
    env.setState(joint_names, np.asarray(q, dtype=float))
    state = env.getState()          # must outlive the transform call, see Cell.set_state
    cm.setCollisionObjectsTransform(state.link_transforms)
    res = ContactResultMap()
    cm.contactTest(res, ContactRequest(ContactTestType_ALL))
    vec = ContactResultVector()
    res.flattenCopyResults(vec)
    return min((float(c.distance) for c in vec), default=None)


def _contact_point(c) -> np.ndarray:
    """Midpoint of a contact's two nearest points, in world metres.

    Defensive because these come straight out of SWIG: a contact that cannot report a
    position is still a contact worth reporting a distance for.
    """
    try:
        a = np.asarray(c.nearest_points[0], dtype=float).reshape(3)
        b = np.asarray(c.nearest_points[1], dtype=float).reshape(3)
    except Exception:
        return np.full(3, np.nan)
    return 0.5 * (a + b)


def _obstacle_margins(man: Manifest) -> list[tuple[str, str]]:
    """Every (moving link, static object) pair, i.e. the robot and gun against the cell.

    ``contact_ok_distance_mm`` exists to absorb the error in approximating a link's own
    geometry, which is a self-collision concern.  Against the panels and tooling there is
    nothing to absorb, so these pairs are governed by ``obstacle_clearance`` instead.
    """
    statics = [s.name for s in man.static_objects]
    return [(link.name, s) for link in man.all_links() if link.mesh for s in statics]


def _apply_margins(env: Environment, man: Manifest, default: float,
                   overrides: dict[tuple[str, str], float] | None = None,
                   obstacle_clearance: float = 0.0) -> None:
    """Set the default margin, then apply the obstacle clearance and any overrides.

    A Tesseract margin is the distance at which a pair counts as colliding, so this one
    number spans both intents: 0 means "touching is a collision", a positive value keeps
    the robot that far clear of the parts, and a negative one tolerates that much overlap.
    """
    env.applyCommand(ChangeCollisionMarginsCommand(default))
    pair_data = CollisionMarginPairData()
    for a, b in _obstacle_margins(man):
        pair_data.setCollisionMargin(a, b, obstacle_clearance)
    for (a, b), m in (overrides or {}).items():
        pair_data.setCollisionMargin(a, b, m)
    env.applyCommand(ChangeCollisionMarginsCommand(
        pair_data, CollisionMarginPairOverrideType_MODIFY))


def _log_gun(man: Manifest, log) -> None:
    """Report the gun's travel in the units the manifest states openings in.

    The joint is an angle and every opening in the manifest is a length, so the conversion
    between them is worth printing: if the stroke shown here does not match the real gun,
    the lever arm has been measured off the wrong point and every opening will be wrong.
    """
    j = man.gun_joint
    if j is None:
        log("no gun joint in the manifest; the tool is treated as rigid")
        return
    if j.is_prismatic:
        log(f"gun joint '{j.name}' is prismatic, opening maps directly to travel "
            f"({man.gun_opening_max:.1f} mm)")
        return
    lo, hi = j.limits
    log(f"gun joint '{j.name}' is angular: {abs(lo if abs(lo) > abs(hi) else hi):.4f} rad "
        f"about a {man.gun_lever_arm:.1f} mm arm, giving {man.gun_opening_max:.1f} mm of "
        f"electrode opening")
    # The start opening is worth printing next to the maximum: a manifest asking for more
    # than the gun has is clamped rather than refused, and that is easy to miss.
    asked = float(man.start_state.get(j.name, 0.0))
    got = man.gun_opening(man.gun_joint_value(asked))
    note = "" if abs(abs(asked) - got) < 0.05 else f"  (clamped from {abs(asked):.1f} mm)"
    log(f"start state opens the gun to {got:.1f} mm{note}")


def _apply_joint_limits(env: Environment, names: list[str], dynamics, log) -> None:
    """Push the real velocity and acceleration limits into the environment.

    Tesseract synthesises limits it was not given -- this cell came up with +/-1.5 rad/s^2
    on every joint, which nobody specified and which is far off the real machine.  Anything
    reading limits from the environment would otherwise be working from invented numbers.
    """
    if dynamics is None:
        return
    for i, name in enumerate(names):
        if i >= len(dynamics.velocity):
            break
        env.applyCommand(ChangeJointVelocityLimitsCommand(name, float(dynamics.velocity[i])))
        env.applyCommand(
            ChangeJointAccelerationLimitsCommand(name, float(dynamics.acceleration[i])))
    log(dynamics.describe(names))


def _export_start_contacts(cell: Cell, man: Manifest, directory: str,
                           start: np.ndarray, pairs: list, log) -> None:
    """Write each pair that reads as touching at the start pose to its own OBJ.

    Called before the exact re-measurement rather than after it, because that step is the
    slow one and this is the geometry it is about to spend minutes on.  The state is pushed
    again here so the file does not depend on what the contact test happened to leave
    loaded: what lands in it is the geometry at the pose that read as contact.
    """
    from . import hullexport
    cell.set_state(start)
    for a, b in pairs:
        try:
            hullexport.export_pair(cell.env, man, directory, (a, b),
                                   f"start_{a}__{b}", log=log)
        except Exception as exc:        # diagnostics must never mask the contact report
            log(f"      could not write the blocking geometry for {a} <-> {b}: "
                f"{type(exc).__name__}: {exc}")


def build(man: Manifest, log=print, out_dir: str | None = None,
          min_shell_mm: float = 40.0, max_shells: int = 80,
          hull_cell_mm: float = 0.0, hull_fill: float = 0.75,
          hull_per_category=None, weld_proximity_mm: float = 0.0,
          tcp_proximity_mm: float = 0.0, far_cell_mm: float = 0.0,
          far_per_category=None,
          hull_overlap: float | None = None,
          merge_cell_mm: float = 0.0, merge_per_category=None,
          enclosed_per_category=None, enclosed_probe_mm: float = 0.0,
          enclosed_voxel_mm: float = 0.0,
          enclosed_keep_mm: float = 0.0, enclosed_dump: bool = False,
          obstacle_clearance_mm: float = 0.0, tcp_check_mm: float = 0.0,
          export_dir: str | None = None,
          penalty=None, dynamics=None) -> Cell:
    """Prepare geometry, emit URDF/SRDF, load the environment and generate the ACM."""
    collision = _resolve_collision_meshes(man, log, min_shell_mm, max_shells, hull_cell_mm,
                                          hull_fill, hull_per_category, weld_proximity_mm,
                                          tcp_proximity_mm, far_cell_mm, far_per_category,
                                          hull_overlap, merge_cell_mm,
                                          merge_per_category, enclosed_per_category,
                                          enclosed_probe_mm,
                                          enclosed_voxel_mm, enclosed_keep_mm,
                                          enclosed_dump)
    velocity = {}
    if dynamics is not None:
        velocity = {n: float(v) for n, v in zip(man.robot_joint_names, dynamics.velocity)}
    builder = SceneBuilder(man, collision, joint_velocity=velocity)

    out_dir = out_dir or os.path.join(man.directory, "generated")
    os.makedirs(out_dir, exist_ok=True)
    urdf_path = os.path.join(out_dir, "cell.urdf").replace("\\", "/")
    srdf_path = os.path.join(out_dir, "cell.srdf").replace("\\", "/")
    kin_path = os.path.join(out_dir, "kinematics_plugins.yaml").replace("\\", "/")
    contact_path = os.path.join(out_dir, "contact_managers_plugins.yaml").replace("\\", "/")
    with open(urdf_path, "w", encoding="utf-8") as fh:
        fh.write(builder.urdf())
    with open(kin_path, "w", encoding="utf-8") as fh:
        fh.write(builder.kinematics_plugins_yaml())
    with open(contact_path, "w", encoding="utf-8") as fh:
        fh.write(builder.contact_managers_yaml())

    def write_srdf(pairs):
        with open(srdf_path, "w", encoding="utf-8") as fh:
            fh.write(builder.srdf(pairs, kin_path, contact_path))

    pairs = builder.adjacent_pairs()
    write_srdf(pairs)

    os.environ.setdefault("TESSERACT_RESOURCE_PATH", man.directory)
    log("loading the scene into Tesseract; builds a collision broadphase over every "
        "shell above")
    env = Environment()
    if not env.init(FilesystemPath(urdf_path), FilesystemPath(srdf_path),
                    GeneralResourceLocator()):
        raise RuntimeError(f"Tesseract failed to load the generated scene in {out_dir}")
    _apply_joint_limits(env, man.robot_joint_names, dynamics, log)
    margin = -man.contact_ok_distance_mm * man.scale
    clearance = obstacle_clearance_mm * man.scale
    _apply_margins(env, man, margin, obstacle_clearance=clearance)
    log(f"scene loaded; collision margin {margin * 1000:.1f} mm between the robot's own "
        f"links (contact_ok_distance_mm={man.contact_ok_distance_mm:g}), "
        f"{clearance * 1000:+.1f} mm against panels and tooling")

    cell = Cell(man, builder, env, pairs)
    cell.margin, cell.obstacle_clearance = margin, clearance
    cell.tcp_check_mm = max(0.0, float(tcp_check_mm))
    if cell.tcp_check_mm > 0.0:
        log(f"collision checks also measure the tool: an interval that carries the gun "
            f"more than {cell.tcp_check_mm:g} mm is split and rechecked, up to "
            f"{MAX_TOOL_SUBDIVISION} times")
    cell.attach_penalty(penalty, log=log)
    cell.attach_dynamics(dynamics)
    _log_gun(man, log)

    if export_dir:
        # Written here rather than after planning.  The start-pose test below re-measures
        # every contacting pair against full-resolution CAD and can run for minutes per
        # pair, and the hulls it is complaining about are exactly what a reader wants open
        # in front of them while it does.  Nothing after this point changes them.
        from . import hullexport
        hullexport.export(env, man, export_dir, log=log)

    # -- pairs in contact at the start pose: re-measure, then loosen or disable ---------
    start = np.array([man.start_state[n] for n in cell.joint_names], dtype=float)
    log("testing the start pose against every collision pair; this is the first full test "
        "of the scene and the slowest one")
    always = cell.contact_pairs(start)
    overrides: dict[tuple[str, str], float] = {}
    obstacle = {tuple(sorted(p)) for p in _obstacle_margins(man)}
    if always:
        if export_dir:
            _export_start_contacts(cell, man, export_dir, start, sorted(always), log)
        log(f"  {len(always)} pairs read as in contact; re-measuring each against the raw "
            f"concave meshes. Every pair loads full-resolution CAD into a scene of its "
            f"own, so expect seconds to minutes per pair")
        exact = _exact_contacts(man, collision, out_dir, sorted(always), start,
                                cell.joint_names, log)
        disable: list[tuple[str, str]] = []
        fatal: list[tuple[str, str, float]] = []
        for (a, b), hull in sorted(always.items(), key=lambda kv: kv[1]):
            if (a, b) not in exact:
                # Unmeasurable, already reported. Inventing a distance here would either
                # disable a real collision or hand the pair a margin nothing justifies, so
                # the pair keeps its default and the start-state check has the final say.
                continue
            true_d = exact[(a, b)]
            base = clearance if (a, b) in obstacle else margin
            if true_d < 0.0:
                # Real geometry genuinely interpenetrates: a modelling problem, not
                # something a margin should paper over.
                if (a, b) in obstacle:
                    # The arm or the gun is inside a panel or a fixture before anything
                    # has moved.  Waiving that would blind the planner to that pair for
                    # the whole run, so the study is wrong and has to be fixed.
                    fatal.append((a, b, true_d))
                    continue
                disable.append((a, b))
                log(f"  ACM: disabling {a} <-> {b} (exact overlap "
                    f"{-true_d * 1000:.1f} mm)")
            elif true_d < base:
                # Clear, but by less than the requested clearance.  The start pose is a
                # given, so the pair keeps its check at the distance actually available
                # rather than making the whole run unplannable.
                overrides[(a, b)] = hull - 1e-9
                log(f"  ! {a} <-> {b} is only {true_d * 1000:.1f} mm apart at the start "
                    f"pose, short of the {base * 1000:.1f} mm clearance; this pair is "
                    f"held to {true_d * 1000:.1f} mm instead")
            else:
                # The pair keeps its collision check; the margin is shifted by the hull
                # inflation measured here, so it still trips at the same *true* overlap.
                overrides[(a, b)] = base + (hull - true_d)
                log(f"  margin: {a} <-> {b} set to {overrides[(a, b)] * 1000:.1f} mm "
                    f"(hulls read {hull * 1000:.1f} mm, exact geometry "
                    f"{true_d * 1000:.1f} mm)")

        if fatal:
            detail = "\n".join(f"  {a} <-> {b}: {-d * 1000:.1f} mm of overlap"
                               for a, b, d in sorted(fatal, key=lambda f: f[2]))
            raise RuntimeError(
                "the robot or the gun starts inside the tooling or the panels:\n"
                f"{detail}\n"
                "These are measured on the exact meshes, not the collision hulls, so "
                "this is a real overlap in the imported study rather than an artefact "
                "of the approximation. Collisions against the parts are never waived -- "
                "fix the start pose or the part placement in the study and re-import.")

        pairs = pairs + [(a, b, "InContactAtStart") for (a, b) in disable]
        write_srdf(pairs)
        log("  reloading the scene with the generated collision matrix; the broadphase is "
            "built a second time")
        env = Environment()
        if not env.init(FilesystemPath(urdf_path), FilesystemPath(srdf_path),
                        GeneralResourceLocator()):
            raise RuntimeError("failed to reload scene with generated collision matrix")
        _apply_joint_limits(env, man.robot_joint_names, dynamics, lambda *a: None)
        _apply_margins(env, man, margin, overrides, obstacle_clearance=clearance)
        cell = Cell(man, builder, env, pairs)
        cell.margin, cell.obstacle_clearance = margin, clearance
        cell.margin_overrides = overrides
        cell.attach_penalty(penalty, log=log)
        cell.attach_dynamics(dynamics)

    if cell.in_collision(start):
        remaining = sorted(cell.contact_pairs(start))
        parts = [p for p in remaining if tuple(sorted(p)) in obstacle]
        if parts:
            # Reached when the exact re-measurement could not put a number on the pair,
            # so it was neither loosened nor waived.  Same verdict as a measured overlap:
            # the parts are not something to plan through.
            raise RuntimeError(
                "the robot or the gun starts in contact with the tooling or the panels: "
                f"{parts}; collisions against the parts are never waived, so fix the "
                "start pose or the part placement in the study and re-import")
        raise RuntimeError(f"start state still in collision: {remaining}")
    log(f"start state is collision free ({len(pairs)} disabled pairs in generated SRDF, "
        f"{len(overrides)} pair margins adjusted)")
    return cell
