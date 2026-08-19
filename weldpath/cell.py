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
        self._pm = None                         # proximity manager, only if penalised
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
    def attach_penalty(self, penalty) -> None:
        """Give the cell a second contact manager dedicated to measuring clearance.

        Margins decide what a manager will even report, and the planning manager's are set
        so that *contact* is the event of interest -- it cannot see a panel 40 mm away.
        Rather than widen those and have every near miss read as a collision, this clones
        the manager and widens only the clone, and only for the robot-against-cell pairs.
        The clone's default margin is driven hugely negative so self-collision pairs stay
        silent and the query cost is confined to the pairs that matter.
        """
        self.penalty = penalty
        self._pm = None
        if penalty is None or not penalty.enabled:
            return
        probe = penalty.probe_mm * self.man.scale
        pm = self._cm.clone()
        pm.setDefaultCollisionMargin(-1.0)
        for a, b in _obstacle_margins(self.man):
            pm.setCollisionMarginPair(a, b, probe)
        pm.setActiveCollisionObjects(self.env.getActiveLinkNames())
        self._pm = pm

    def clearance_mm(self, q: np.ndarray) -> float:
        """Closest approach between the robot or gun and the static objects, in mm.

        Returns the probe distance when nothing is within it, which is all the penalty
        needs: beyond that the cost is flat, so the exact figure does not matter.
        """
        if self._pm is None:
            return float("inf")
        # Deliberately not set_state: refreshing the planning manager's transforms costs as
        # much as the query itself and nothing here is going to ask it anything.
        self._load_state(q)
        return self._clearance_now()

    def _clearance_now(self) -> float:
        """Clearance at the state already loaded.  Assumes ``set_state`` has just run."""
        self._pm.setCollisionObjectsTransform(self._state.link_transforms)
        res = ContactResultMap()
        self._pm.contactTest(res, ContactRequest(ContactTestType_ALL))
        if res.size() == 0:
            return self.penalty.probe_mm
        vec = ContactResultVector()
        res.flattenCopyResults(vec)
        worst = min((float(c.distance) for c in vec), default=None)
        if worst is None:
            return self.penalty.probe_mm
        return worst / self.man.scale

    def penalty_factor(self, q: np.ndarray) -> float:
        """Cost multiplier for standing where ``q`` puts the robot."""
        if self._pm is None:
            return 1.0
        return self.penalty.factor(self.clearance_mm(q))

    # -- state / collision ---------------------------------------------------
    def _load_state(self, q: np.ndarray) -> None:
        self.env.setState(self._state_names, self._state_values(q))
        # Must hold a reference: passing env.getState().link_transforms inline lets the
        # temporary die and the binding then reads freed memory.
        self._state = self.env.getState()

    def set_state(self, q: np.ndarray) -> None:
        self._load_state(q)
        self._cm.setCollisionObjectsTransform(self._state.link_transforms)

    def in_collision(self, q: np.ndarray) -> bool:
        self.set_state(q)
        res = ContactResultMap()
        self._cm.contactTest(res, ContactRequest(ContactTestType_FIRST))
        return res.size() > 0

    def contact_pairs(self, q: np.ndarray) -> dict[tuple[str, str], float]:
        self.set_state(q)
        res = ContactResultMap()
        self._cm.contactTest(res, ContactRequest(ContactTestType_ALL))
        vec = ContactResultVector()
        res.flattenCopyResults(vec)
        worst: dict[tuple[str, str], float] = {}
        for c in vec:
            key = tuple(sorted((c.link_names[0], c.link_names[1])))
            if key not in worst or c.distance < worst[key]:
                worst[key] = float(c.distance)
        return worst

    def segment_collides(self, a: np.ndarray, b: np.ndarray, max_step: float = 0.05) -> bool:
        """Discretely check the straight joint-space segment a->b."""
        n = max(2, int(np.ceil(np.max(np.abs(b - a)) / max_step)) + 1)
        for t in np.linspace(0.0, 1.0, n):
            if self.in_collision(a + t * (b - a)):
                return True
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
    def fk(self, q: np.ndarray, link: str = TCP_LINK) -> np.ndarray:
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


def _resolve_collision_meshes(man: Manifest, log, min_extent: float,
                              max_shells: int, hull_cell: float) -> dict[str, str]:
    from . import meshprep

    rel: dict[str, str] = {}
    for link in man.all_links():
        if link.mesh:
            rel[link.name] = link.mesh
    for s in man.static_objects:
        if s.mesh:
            rel[s.name] = s.mesh
    log("preparing collision geometry (convex decomposition):")
    return meshprep.prepare(man.directory, rel, man.scale, log=log,
                            min_extent=min_extent, max_shells=max_shells,
                            hull_cell=hull_cell)


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
    for a, b in pairs:
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


def build(man: Manifest, log=print, out_dir: str | None = None,
          min_shell_mm: float = 40.0, max_shells: int = 80,
          hull_cell_mm: float = 0.0, obstacle_clearance_mm: float = 0.0,
          penalty=None, dynamics=None) -> Cell:
    """Prepare geometry, emit URDF/SRDF, load the environment and generate the ACM."""
    collision = _resolve_collision_meshes(man, log, min_shell_mm, max_shells, hull_cell_mm)
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
    cell.attach_penalty(penalty)
    cell.attach_dynamics(dynamics)
    _log_gun(man, log)

    # -- pairs in contact at the start pose: re-measure, then loosen or disable ---------
    start = np.array([man.start_state[n] for n in cell.joint_names], dtype=float)
    always = cell.contact_pairs(start)
    overrides: dict[tuple[str, str], float] = {}
    if always:
        exact = _exact_contacts(man, collision, out_dir, sorted(always), start,
                                cell.joint_names, log)
        obstacle = {tuple(sorted(p)) for p in _obstacle_margins(man)}
        disable: list[tuple[str, str]] = []
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

        pairs = pairs + [(a, b, "InContactAtStart") for (a, b) in disable]
        write_srdf(pairs)
        env = Environment()
        if not env.init(FilesystemPath(urdf_path), FilesystemPath(srdf_path),
                        GeneralResourceLocator()):
            raise RuntimeError("failed to reload scene with generated collision matrix")
        _apply_joint_limits(env, man.robot_joint_names, dynamics, lambda *a: None)
        _apply_margins(env, man, margin, overrides, obstacle_clearance=clearance)
        cell = Cell(man, builder, env, pairs)
        cell.margin, cell.obstacle_clearance = margin, clearance
        cell.margin_overrides = overrides
        cell.attach_penalty(penalty)
        cell.attach_dynamics(dynamics)

    if cell.in_collision(start):
        remaining = cell.contact_pairs(start)
        raise RuntimeError(f"start state still in collision: {sorted(remaining)}")
    log(f"start state is collision free ({len(pairs)} disabled pairs in generated SRDF, "
        f"{len(overrides)} pair margins adjusted)")
    return cell
