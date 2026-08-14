"""Tesseract environment construction, collision configuration and kinematics helpers.

Two things here are load-bearing and were established by measurement rather than taste:

* **The collision margin must be negative.**  ``planning.contact_ok_distance_mm`` in the
  manifest is a tolerance for contact, not a safety buffer.  The gun genuinely rests
  against the robot base in the start pose.  With Tesseract's default zero margin the
  start state is invalid and every planner fails immediately -- which is exactly the
  ``freespace transit failed`` recorded in the sample output.
* **Pairs already in contact at the start state are disabled.**  Convex decomposition
  over-estimates penetration for the gun/base pair (~62 mm against ~13 mm on the exact
  meshes).  Rather than silently loosening the margin everywhere, those specific pairs go
  into the generated collision matrix, and the run reports which ones it disabled.
"""
from __future__ import annotations

import os

import numpy as np

from tesseract_robotics.tesseract_collision import (
    ContactRequest, ContactResultMap, ContactResultVector, ContactTestType_ALL,
    ContactTestType_FIRST)
from tesseract_robotics.tesseract_common import (
    FilesystemPath, GeneralResourceLocator, Isometry3d, ManipulatorInfo)
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
                              max_shells: int) -> dict[str, str]:
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
                            min_extent=min_extent, max_shells=max_shells)


def build(man: Manifest, log=print, out_dir: str | None = None,
          min_shell_mm: float = 40.0, max_shells: int = 80) -> Cell:
    """Prepare geometry, emit URDF/SRDF, load the environment and generate the ACM."""
    collision = _resolve_collision_meshes(man, log, min_shell_mm, max_shells)
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
    env.applyCommand(ChangeCollisionMarginsCommand(margin))
    log(f"scene loaded; collision margin {margin * 1000:.1f} mm "
        f"(contact_ok_distance_mm={man.contact_ok_distance_mm:g})")

    cell = Cell(man, builder, env, pairs)

    # -- collision matrix: disable pairs that are in contact in the start pose ----------
    start = np.array([man.start_state[n] for n in cell.joint_names], dtype=float)
    always = cell.contact_pairs(start)
    if always:
        for (a, b), dist in sorted(always.items(), key=lambda kv: kv[1]):
            log(f"  ACM: disabling {a} <-> {b} (in contact at start, {dist * 1000:.1f} mm)")
        pairs = pairs + [(a, b, "InContactAtStart") for (a, b) in always]
        write_srdf(pairs)
        env = Environment()
        if not env.init(FilesystemPath(urdf_path), FilesystemPath(srdf_path),
                        GeneralResourceLocator()):
            raise RuntimeError("failed to reload scene with generated collision matrix")
        env.applyCommand(ChangeCollisionMarginsCommand(margin))
        cell = Cell(man, builder, env, pairs)

    if cell.in_collision(start):
        remaining = cell.contact_pairs(start)
        raise RuntimeError(f"start state still in collision: {sorted(remaining)}")
    log(f"start state is collision free ({len(pairs)} disabled pairs in generated SRDF)")
    return cell
