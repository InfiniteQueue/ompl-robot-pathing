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
    ChangeCollisionMarginsCommand, Environment)
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
        # The cached manager was built under the old margins, so take a fresh one.
        self._cm = self.env.getDiscreteContactManager()
        self._cm.setActiveCollisionObjects(self.env.getActiveLinkNames())
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

    # -- state / collision ---------------------------------------------------
    def set_state(self, q: np.ndarray) -> None:
        self.env.setState(self.joint_names, np.asarray(q, dtype=float))
        # Must hold a reference: passing env.getState().link_transforms inline lets the
        # temporary die and the binding then reads freed memory.
        self._state = self.env.getState()
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

    def within_limits(self, q: np.ndarray) -> bool:
        return bool(np.all(q >= self.lower - 1e-9) and np.all(q <= self.upper + 1e-9))

    # -- kinematics ----------------------------------------------------------
    def fk(self, q: np.ndarray, link: str = TCP_LINK) -> np.ndarray:
        self.env.setState(self.joint_names, np.asarray(q, dtype=float))
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

    Concave geometry is far too slow to plan with (~700 ms a check against ~3 ms), but
    this runs once per tripped pair with every other object switched off, so the cost is
    a few hundred milliseconds in total.  Returns the worst distance per pair, negative
    for penetration; a pair absent from the result had no contact within the probe.
    """
    links = sorted({name for pair in pairs for name in pair})
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
    env.applyCommand(ChangeCollisionMarginsCommand(EXACT_PROBE_MARGIN))

    raw = {l.name: l.mesh for l in man.all_links() if l.mesh}
    raw.update({s.name: s.mesh for s in man.static_objects if s.mesh})
    for name in links:
        _assert_scale(env, man, name, raw[name])

    cm = env.getDiscreteContactManager()
    objects = list(cm.getCollisionObjects())
    env.setState(joint_names, np.asarray(q, dtype=float))
    state = env.getState()          # must outlive the transform call, see Cell.set_state

    out: dict[tuple[str, str], float] = {}
    for a, b in pairs:
        for name in objects:
            if name in (a, b):
                cm.enableCollisionObject(name)
            else:
                cm.disableCollisionObject(name)
        cm.setActiveCollisionObjects([a, b])
        cm.setCollisionObjectsTransform(state.link_transforms)
        res = ContactResultMap()
        cm.contactTest(res, ContactRequest(ContactTestType_ALL))
        vec = ContactResultVector()
        res.flattenCopyResults(vec)
        worst = min((float(c.distance) for c in vec), default=None)
        if worst is not None:
            out[(a, b)] = worst
    return out


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


def build(man: Manifest, log=print, out_dir: str | None = None,
          min_shell_mm: float = 40.0, max_shells: int = 80,
          hull_cell_mm: float = 0.0, obstacle_clearance_mm: float = 0.0) -> Cell:
    """Prepare geometry, emit URDF/SRDF, load the environment and generate the ACM."""
    collision = _resolve_collision_meshes(man, log, min_shell_mm, max_shells, hull_cell_mm)
    builder = SceneBuilder(man, collision)

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
    margin = -man.contact_ok_distance_mm * man.scale
    clearance = obstacle_clearance_mm * man.scale
    _apply_margins(env, man, margin, obstacle_clearance=clearance)
    log(f"scene loaded; collision margin {margin * 1000:.1f} mm between the robot's own "
        f"links (contact_ok_distance_mm={man.contact_ok_distance_mm:g}), "
        f"{clearance * 1000:+.1f} mm against panels and tooling")

    cell = Cell(man, builder, env, pairs)
    cell.margin, cell.obstacle_clearance = margin, clearance

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
            # A pair with no contact within the probe is treated as clear at the probe
            # distance, which is the most conservative reading of "nothing found".
            true_d = exact.get((a, b), EXACT_PROBE_MARGIN)
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
        _apply_margins(env, man, margin, overrides, obstacle_clearance=clearance)
        cell = Cell(man, builder, env, pairs)
        cell.margin, cell.obstacle_clearance = margin, clearance
        cell.margin_overrides = overrides

    if cell.in_collision(start):
        remaining = cell.contact_pairs(start)
        raise RuntimeError(f"start state still in collision: {sorted(remaining)}")
    log(f"start state is collision free ({len(pairs)} disabled pairs in generated SRDF, "
        f"{len(overrides)} pair margins adjusted)")
    return cell
