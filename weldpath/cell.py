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
                   require_collision_free: bool = True) -> np.ndarray | None:
        """Best joint solution for a pose: in limits, collision free, closest to a seed."""
        best, best_cost = None, np.inf
        for seed in seeds:
            for q in self.ik(pose_world_mm, seed):
                if not self.within_limits(q):
                    continue
                if require_collision_free and self.in_collision(q):
                    continue
                cost = float(np.linalg.norm(q - seeds[0]))
                if cost < best_cost:
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
