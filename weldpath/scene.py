"""Build a Tesseract scene (URDF + SRDF) from the manifest and load it.

Coordinate convention
---------------------
The manifest gives every pose in *world* coordinates at the captured configuration
(all joints at ``captured_value``).  A URDF is a tree of relative transforms, so:

* each link gets a frame origin -- the anchor of the joint that creates it, or the world
  origin for a device base link;
* a joint's ``<origin>`` is ``frame_origin(child) - frame_origin(parent)``;
* a link's mesh ``<origin>`` is ``-frame_origin(link)``, which cancels the accumulated
  frame offset and puts the world-coordinate mesh back exactly where it belongs.

A revolute joint's anchor is only defined up to a slide along its own axis, and the
manifest exploits that (anchors sit 1 m off along the axis).  The mesh offset above
cancels regardless, so the arbitrary choice is harmless.  Forward kinematics at the
captured configuration reproduces the manifest's TCP pose to sub-micron accuracy.
"""
from __future__ import annotations

import os

import numpy as np

from .manifest import Manifest

WORLD_LINK = "world"
TCP_LINK = "tcp"
GROUP = "manipulator"


def rpy_from_matrix(R: np.ndarray) -> tuple[float, float, float]:
    """URDF fixed-axis roll-pitch-yaw from a rotation matrix."""
    sy = float(np.hypot(R[0, 0], R[1, 0]))
    if sy < 1e-9:                                   # gimbal lock
        return float(np.arctan2(-R[1, 2], R[1, 1])), float(np.arctan2(-R[2, 0], sy)), 0.0
    return (
        float(np.arctan2(R[2, 1], R[2, 2])),
        float(np.arctan2(-R[2, 0], sy)),
        float(np.arctan2(R[1, 0], R[0, 0])),
    )


def _xyz(v: np.ndarray) -> str:
    return "%.9g %.9g %.9g" % (v[0], v[1], v[2])


# The arm carries the gun back against this link in ordinary poses, so contact between the
# two is a constant of the machine rather than news.  A joint number, in the same
# convention as weldpath.profile's COUPLED_JOINT: joint N drives the link named here.
GUN_CLEAR_JOINT = 3


class SceneBuilder:
    def __init__(self, man: Manifest, collision_meshes: dict[str, str],
                 exact_links: set[str] | None = None,
                 joint_velocity: dict[str, float] | None = None):
        """``exact_links`` take their collision geometry from the raw manifest mesh, kept
        concave, instead of the prepared convex decomposition.  Used to re-measure a
        contact against true geometry; far too slow to plan with."""
        self.man = man
        self.collision = collision_meshes
        self.exact_links = set(exact_links or ())
        # Per-joint velocity limits, so the emitted scene describes the real machine rather
        # than a placeholder. Acceleration has no URDF attribute and is applied to the
        # environment directly instead -- see weldpath.cell.
        self.joint_velocity = dict(joint_velocity or {})
        self.raw_meshes = {l.name: man.mesh_path(l.mesh)
                           for l in man.all_links() if l.mesh}
        self.raw_meshes.update({s.name: man.mesh_path(s.mesh)
                                for s in man.static_objects if s.mesh})
        self.scale = man.scale
        self.frame_origin: dict[str, np.ndarray] = {}
        self._compute_frames()

    # -- frames -------------------------------------------------------------
    def _compute_frames(self) -> None:
        for dev in self.man.devices:
            self.frame_origin[dev.base_link] = np.zeros(3)
            for j in dev.joints:
                self.frame_origin[j.child_link] = np.array(j.anchor_world, dtype=float)
        for s in self.man.static_objects:
            self.frame_origin[s.name] = np.zeros(3)

    # -- URDF ---------------------------------------------------------------
    def _link_xml(self, name: str, mesh_path: str | None, visual_mesh: str | None) -> str:
        if not mesh_path and not visual_mesh:
            return f'  <link name="{name}"/>\n'
        origin = -self.frame_origin.get(name, np.zeros(3)) * self.scale
        parts = [f'  <link name="{name}">\n']
        if visual_mesh:
            parts.append(
                f'    <visual>\n'
                f'      <origin xyz="{_xyz(origin)}" rpy="0 0 0"/>\n'
                f'      <geometry><mesh filename="{visual_mesh}" '
                f'tesseract:make_convex="false" scale="{self.scale} {self.scale} {self.scale}"/>'
                f'</geometry>\n'
                f'    </visual>\n'
            )
        if name in self.exact_links and name in self.raw_meshes:
            # Raw manifest mesh: still in manifest units, and must not be hulled.
            s = self.scale
            parts.append(
                f'    <collision>\n'
                f'      <origin xyz="{_xyz(origin)}" rpy="0 0 0"/>\n'
                f'      <geometry><mesh filename="{self.raw_meshes[name]}" '
                f'tesseract:make_convex="false" scale="{s} {s} {s}"/></geometry>\n'
                f'    </collision>\n'
            )
        elif mesh_path:
            # Prepared meshes are already in metres, hence no scale attribute.
            parts.append(
                f'    <collision>\n'
                f'      <origin xyz="{_xyz(origin)}" rpy="0 0 0"/>\n'
                f'      <geometry><mesh filename="{mesh_path}" '
                f'tesseract:make_convex="true"/></geometry>\n'
                f'    </collision>\n'
            )
        parts.append("  </link>\n")
        return "".join(parts)

    def _joint_xml(self, j, parent: str, child: str) -> str:
        origin = (self.frame_origin[child] - self.frame_origin[parent]) * self.scale
        axis = np.array(j.axis_world, dtype=float)
        n = np.linalg.norm(axis)
        axis = axis / n if n > 0 else np.array([0.0, 0.0, 1.0])
        lo, hi = j.limits
        if j.is_prismatic:                      # prismatic limits are a length
            lo, hi = lo * self.scale, hi * self.scale
        velocity = self.joint_velocity.get(j.name, 3.0)
        return (
            f'  <joint name="{j.name}" type="{j.type}">\n'
            f'    <parent link="{parent}"/>\n'
            f'    <child link="{child}"/>\n'
            f'    <origin xyz="{_xyz(origin)}" rpy="0 0 0"/>\n'
            f'    <axis xyz="{_xyz(axis)}"/>\n'
            f'    <limit lower="{lo:.9g}" upper="{hi:.9g}" effort="0" velocity="{velocity:.9g}"/>\n'
            f'  </joint>\n'
        )

    def urdf(self) -> str:
        man = self.man
        out = ['<?xml version="1.0"?>\n<robot name="cell" tesseract:make_convex="false">\n']
        out.append(f'  <link name="{WORLD_LINK}"/>\n')

        for dev in man.devices:
            for link in dev.links:
                visual = man.mesh_path(link.mesh) if link.mesh else None
                out.append(self._link_xml(link.name, self.collision.get(link.name), visual))
        out.append(f'  <link name="{TCP_LINK}"/>\n')
        for s in man.static_objects:
            visual = man.mesh_path(s.mesh) if s.mesh else None
            out.append(self._link_xml(s.name, self.collision.get(s.name), visual))

        for dev in man.devices:
            for j in dev.joints:
                out.append(self._joint_xml(j, j.parent_link, j.child_link))

        # gun (and any other device) rigidly attached to a robot link
        for parent, child in man.attachments:
            origin = (self.frame_origin[child] - self.frame_origin[parent]) * self.scale
            out.append(
                f'  <joint name="attach_{child}" type="fixed">\n'
                f'    <parent link="{parent}"/>\n    <child link="{child}"/>\n'
                f'    <origin xyz="{_xyz(origin)}" rpy="0 0 0"/>\n  </joint>\n'
            )

        # TCP, expressed relative to its parent link's frame
        T = np.array(man.tcp_world_pose, dtype=float)
        pos = (T[:3, 3] - self.frame_origin[man.tcp_parent_link]) * self.scale
        r, p, y = rpy_from_matrix(T[:3, :3])
        out.append(
            f'  <joint name="tcp_joint" type="fixed">\n'
            f'    <parent link="{man.tcp_parent_link}"/>\n    <child link="{TCP_LINK}"/>\n'
            f'    <origin xyz="{_xyz(pos)}" rpy="{r:.9g} {p:.9g} {y:.9g}"/>\n  </joint>\n'
        )

        roots = [d.base_link for d in man.devices
                 if d.base_link not in {c for _, c in man.attachments}]
        for name in roots + [s.name for s in man.static_objects]:
            out.append(
                f'  <joint name="world_to_{name}" type="fixed">\n'
                f'    <parent link="{WORLD_LINK}"/>\n    <child link="{name}"/>\n'
                f'    <origin xyz="0 0 0" rpy="0 0 0"/>\n  </joint>\n'
            )
        out.append("</robot>\n")
        return "".join(out)

    # -- SRDF ---------------------------------------------------------------
    def srdf(self, disabled_pairs: list[tuple[str, str, str]],
             kinematics_config: str | None = None,
             contact_config: str | None = None) -> str:
        out = ['<?xml version="1.0"?>\n<robot name="cell">\n']
        out.append(f'  <group name="{GROUP}">\n'
                   f'    <chain base_link="{self.man.robot.base_link}" tip_link="{TCP_LINK}"/>\n'
                   f'  </group>\n')
        for a, b, reason in disabled_pairs:
            out.append(f'  <disable_collisions link1="{a}" link2="{b}" reason="{reason}"/>\n')
        if kinematics_config:
            out.append(f'  <kinematics_plugin_config filename="{kinematics_config}"/>\n')
        if contact_config:
            out.append(f'  <contact_managers_plugin_config filename="{contact_config}"/>\n')
        out.append("</robot>\n")
        return "".join(out)

    def kinematics_plugins_yaml(self) -> str:
        base = self.man.robot.base_link
        return (
            "kinematic_plugins:\n"
            "  search_libraries:\n"
            "    - tesseract_kinematics_kdl_factories\n"
            "  fwd_kin_plugins:\n"
            f"    {GROUP}:\n"
            "      default: KDLFwdKinChain\n"
            "      plugins:\n"
            "        KDLFwdKinChain:\n"
            "          class: KDLFwdKinChainFactory\n"
            "          config:\n"
            f"            base_link: {base}\n"
            f"            tip_link: {TCP_LINK}\n"
            "  inv_kin_plugins:\n"
            f"    {GROUP}:\n"
            "      default: KDLInvKinChainLMA\n"
            "      plugins:\n"
            "        KDLInvKinChainLMA:\n"
            "          class: KDLInvKinChainLMAFactory\n"
            "          config:\n"
            f"            base_link: {base}\n"
            f"            tip_link: {TCP_LINK}\n"
        )

    @staticmethod
    def contact_managers_yaml() -> str:
        return (
            "contact_manager_plugins:\n"
            "  search_libraries:\n"
            "    - tesseract_collision_bullet_factories\n"
            "  discrete_plugins:\n"
            "    default: BulletDiscreteBVHManager\n"
            "    plugins:\n"
            "      BulletDiscreteBVHManager:\n"
            "        class: BulletDiscreteBVHManagerFactory\n"
            "  continuous_plugins:\n"
            "    default: BulletCastBVHManager\n"
            "    plugins:\n"
            "      BulletCastBVHManager:\n"
            "        class: BulletCastBVHManagerFactory\n"
        )

    def adjacent_pairs(self) -> list[tuple[str, str, str]]:
        pairs = [(WORLD_LINK, d.base_link, "Adjacent") for d in self.man.devices
                 if d.base_link not in {c for _, c in self.man.attachments}]
        for dev in self.man.devices:
            for j in dev.joints:
                pairs.append((j.parent_link, j.child_link, "Adjacent"))
        pairs += [(p, c, "Adjacent") for p, c in self.man.attachments]
        pairs.append((self.man.tcp_parent_link, TCP_LINK, "Adjacent"))
        for s in self.man.static_objects:
            pairs.append((WORLD_LINK, s.name, "Adjacent"))
        # statics never move relative to each other
        names = [s.name for s in self.man.static_objects]
        for i in range(len(names)):
            for k in range(i + 1, len(names)):
                pairs.append((names[i], names[k], "Never"))
        pairs += self.never_collide_pairs()
        seen: set[tuple[str, str]] = set()
        out: list[tuple[str, str, str]] = []
        for a, b, reason in pairs:
            key = (a, b) if a <= b else (b, a)
            if key not in seen:
                seen.add(key)
                out.append((a, b, reason))
        return out

    def never_collide_pairs(self) -> list[tuple[str, str, str]]:
        """Contacts that can never say anything about whether a move is safe.

        Two groups, both worked out from the manifest's structure rather than by matching
        names, so they hold for any cell:

        * **every pair of robot links.**  The arm's own castings read as touching wherever
          they are merely close, and this machine cannot fold into itself, so a
          self-collision reading is noise in every pose.
        * **every pair among the gun and the wrist it hangs off** -- the whole gun (body
          and moving electrode alike) plus the link the gun is bolted to and the link
          before it.  The gun is *mounted* there, and swinging the tip through 200 mm of
          travel inevitably brings it close to both, so those readings are a constant of
          the assembly; leaving them in makes the gun uncloseable in most poses.  The link
          driven by joint ``GUN_CLEAR_JOINT`` joins that group: the arm folds the gun back
          against it in ordinary poses, and the reading says nothing about whether the
          move is safe.

        Everything else the gun and the arm can hit -- the panels, the tooling, the rest
        of the arm -- keeps its check, and cannot be waived later (see
        :func:`weldpath.cell.build`).
        """
        def combinations(names, reason):
            names = sorted(set(names))
            return [(names[i], names[k], reason)
                    for i in range(len(names)) for k in range(i + 1, len(names))]

        robot = [l.name for l in self.man.robot.links]
        mount = [l.name for dev in self.man.devices[1:] for l in dev.links]
        for parent, _child in self.man.attachments:   # the wrist, and the link before it
            mount.append(parent)
            for dev in self.man.devices:
                for j in dev.joints:
                    if j.child_link == parent:
                        mount.append(j.parent_link)
        joints = self.man.robot.joints
        if len(joints) >= GUN_CLEAR_JOINT:
            mount.append(joints[GUN_CLEAR_JOINT - 1].child_link)
        return (combinations(robot, "RobotSelfCollision")
                + combinations(mount, "GunMounting"))