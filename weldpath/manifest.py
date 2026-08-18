"""Parsing of the input manifest.

The manifest describes the cell in *world* coordinates at a single captured
configuration: every ``anchor_world`` / ``pose_world`` / ``world_pose`` is the pose the
object had when all joints were at their ``captured_value``.  That is the fact the whole
scene build rests on -- see :mod:`weldpath.scene`.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MANIFEST_NAME = "manifest.json"


@dataclass
class Link:
    name: str
    mesh: str | None
    triangles: int = 0


@dataclass
class Joint:
    name: str
    type: str
    parent_link: str
    child_link: str
    anchor_world: np.ndarray
    axis_world: np.ndarray
    limits: tuple[float, float]
    captured_value: float = 0.0
    # ``measured`` marks a joint whose axis and anchor were recovered from the geometry
    # rather than declared.  ``measured_slide`` is the residual slide along the axis from
    # that recovery, which the scene build cancels anyway (see :mod:`weldpath.scene`), so
    # both are carried for traceability and neither changes the kinematics.
    measured: bool = False
    measured_slide: float = 0.0

    @property
    def is_prismatic(self) -> bool:
        return self.type == "prismatic"

    @property
    def unit_axis(self) -> np.ndarray:
        axis = np.array(self.axis_world, dtype=float)
        n = float(np.linalg.norm(axis))
        return axis / n if n > 0 else np.array([0.0, 0.0, 1.0])

    @property
    def open_sign(self) -> float:
        """Which way off the closed position this joint is free to travel.

        The gun's limits sit entirely on one side of zero, so the sign of the reachable
        travel is a property of the manifest rather than something to assume.
        """
        lo, hi = self.limits
        return -1.0 if abs(lo) > abs(hi) else 1.0


@dataclass
class Device:
    name: str
    base_link: str
    links: list[Link]
    joints: list[Joint]


@dataclass
class StaticObject:
    name: str
    mesh: str | None
    category: str
    triangles: int = 0


@dataclass
class Locator:
    name: str
    pose_world: np.ndarray          # 4x4, translation in mm -- the pose planned against
    is_weld: bool
    gun_opening_arrive: float       # mm
    gun_opening_leave: float        # mm
    pose_world_import: np.ndarray | None = None   # set when pose_world has been shifted

    @property
    def export_pose(self) -> np.ndarray:
        """Where this locator belongs in the output: as imported, not as planned.

        A weld may be planned against a pose backed off from the panel, but the exported
        program has to name the weld where the study put it.
        """
        return self.pose_world if self.pose_world_import is None else self.pose_world_import


@dataclass
class Manifest:
    directory: str
    units: str
    contact_ok_distance_mm: float
    linear_zone_mm: float
    devices: list[Device]
    attachments: list[tuple[str, str]]
    gun_moving_links: list[str]
    tcp_parent_link: str
    tcp_world_pose: np.ndarray
    static_objects: list[StaticObject]
    locators: list[Locator]
    start_state: dict[str, float]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    # ---- derived -----------------------------------------------------------
    @property
    def scale(self) -> float:
        """Factor converting manifest length units into metres."""
        return {"mm": 0.001, "m": 1.0, "cm": 0.01}[self.units]

    @property
    def robot(self) -> Device:
        return self.devices[0]

    @property
    def robot_joint_names(self) -> list[str]:
        return [j.name for j in self.robot.joints if j.type != "fixed"]

    @property
    def gun_joint(self) -> Joint | None:
        for d in self.devices[1:]:
            for j in d.joints:
                return j
        return None

    @property
    def gun_joint_name(self) -> str | None:
        j = self.gun_joint
        return j.name if j else None

    @property
    def gun_lever_arm(self) -> float:
        """Distance from the gun's rotation axis to the electrode gap, in manifest units.

        The TCP of a weld gun sits at the electrode faces, so it is the point whose travel
        *is* the opening.  Measuring the arm from the manifest rather than hardcoding it
        means a different gun needs no code change; for the supplied cell it comes out at
        515 mm, giving a 205 mm stroke over the joint's 0.40 rad of travel.

        Two approximations are known and deliberately accepted here.  A quoted opening is
        really the TCP-to-tip distance measured *along the TCP's z axis*, down to the tip's
        lowest point -- which the rounding of the tip puts below its end -- whereas this
        treats it as the chord swept by the TCP itself.  Measured against the tip mesh the
        two disagree by about 0.5%: 0.26 mm at a 50 mm opening, 1.1 mm at full stroke.  That
        is far inside the error already present in the convex collision geometry, which runs
        to tens of millimetres, so it is not worth modelling the tip profile to remove.
        """
        j = self.gun_joint
        if j is None:
            return 0.0
        axis = j.unit_axis
        d = np.array(self.tcp_world_pose, dtype=float)[:3, 3] - np.array(j.anchor_world,
                                                                        dtype=float)
        return float(np.linalg.norm(d - np.dot(d, axis) * axis))

    def gun_joint_value(self, opening: float) -> float:
        """Joint value, in *environment* units, that opens the gun by ``opening``.

        The manifest states openings as a length while the joint is now an angle, so this
        is the conversion between them.  The electrode gap is the straight-line distance
        between the moving electrode and its closed position, i.e. the chord of the arc
        rather than the arc itself -- a 1.3% difference at full travel, but the chord is
        the one that is physically the gap.  A prismatic gun keeps the direct mapping.
        """
        j = self.gun_joint
        if j is None:
            return 0.0
        if j.is_prismatic:
            # Prismatic limits are scaled into metres when the URDF is written.
            lo, hi = sorted((j.limits[0] * self.scale, j.limits[1] * self.scale))
            return min(hi, max(lo, j.open_sign * abs(opening) * self.scale))
        radius = self.gun_lever_arm
        if radius <= 0.0:
            return 0.0
        ratio = min(1.0, max(-1.0, abs(opening) / (2.0 * radius)))
        value = j.open_sign * 2.0 * float(np.arcsin(ratio))
        # An opening at or beyond full travel converts to a hair past the limit, and an
        # over-wide opening from the manifest would land well past it.  Clamped here so the
        # environment is never handed a joint value it would refuse or silently truncate.
        lo, hi = sorted(j.limits)
        return min(hi, max(lo, value))

    def gun_opening(self, joint_value: float) -> float:
        """Inverse of :meth:`gun_joint_value`: the opening a joint value corresponds to."""
        j = self.gun_joint
        if j is None:
            return 0.0
        if j.is_prismatic:
            return abs(joint_value) / self.scale
        return 2.0 * self.gun_lever_arm * float(np.sin(abs(joint_value) / 2.0))

    @property
    def gun_opening_max(self) -> float:
        """Widest opening the gun can reach, in manifest units."""
        j = self.gun_joint
        if j is None:
            return 0.0
        lo, hi = j.limits
        return self.gun_opening(lo if abs(lo) > abs(hi) else hi)

    def all_links(self) -> list[Link]:
        out: list[Link] = []
        for d in self.devices:
            out.extend(d.links)
        return out

    def mesh_path(self, rel: str) -> str:
        return os.path.join(self.directory, rel).replace("\\", "/")

    def obstacle_names(self) -> list[str]:
        """Static objects that the robot must avoid."""
        return [s.name for s in self.static_objects if s.mesh]


def shift_weld_locators(man: Manifest, distance: float) -> int:
    """Move every weld locator along its own z axis by ``distance`` manifest units.

    A weld locator is authored on the panel surface, which puts the gun tip inside the
    sheet once the gun geometry is taken into account -- so the pose as exported can be
    unreachable without allowing the tool to interpenetrate the part.  Backing the
    locator off along its own approach axis restores a pose the robot can actually hold,
    and doing it here means the planner, the collision checks and the emitted
    ``tcp_world_mm`` all agree on where the weld is.

    The pose as imported is kept on the locator, because that is the one the exported
    program must name -- the shift exists to make the pose plannable, not to move the weld.

    Returns the number of locators moved.
    """
    if not distance:
        return 0
    moved = 0
    for loc in man.locators:
        if not loc.is_weld:
            continue
        loc.pose_world_import = loc.pose_world
        shifted = loc.pose_world.copy()
        shifted[:3, 3] += shifted[:3, 2] * distance
        loc.pose_world = shifted
        moved += 1
    return moved


def _mat(v: Any) -> np.ndarray:
    return np.array(v, dtype=np.float64)


def load(directory: str) -> Manifest:
    path = os.path.join(directory, MANIFEST_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no {MANIFEST_NAME} in {directory}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    devices = []
    for d in raw.get("devices", []):
        links = [Link(l["name"], l.get("mesh"), l.get("triangles", 0)) for l in d.get("links", [])]
        joints = [
            Joint(
                name=j["name"],
                type=j["type"],
                parent_link=j["parent_link"],
                child_link=j["child_link"],
                anchor_world=_mat(j["anchor_world"]),
                axis_world=_mat(j["axis_world"]),
                limits=(float(j["limits"][0]), float(j["limits"][1])),
                captured_value=float(j.get("captured_value", 0.0)),
                measured=bool(j.get("measured", False)),
                measured_slide=float(j.get("measured_slide_mm", 0.0)),
            )
            for j in d.get("joints", [])
        ]
        devices.append(Device(d["name"], d["base_link"], links, joints))

    planning = raw.get("planning", {})
    tcp = raw.get("tcp", {})
    return Manifest(
        directory=directory.replace("\\", "/"),
        units=raw.get("units", "mm"),
        contact_ok_distance_mm=float(planning.get("contact_ok_distance_mm", 0.0)),
        linear_zone_mm=float(planning.get("linear_zone_mm", 0.0)),
        devices=devices,
        attachments=[(a["parent_link"], a["child_link"]) for a in raw.get("attachments", [])],
        gun_moving_links=list(raw.get("gun_moving_links", [])),
        tcp_parent_link=tcp.get("parent_link", ""),
        tcp_world_pose=_mat(tcp.get("world_pose", np.eye(4).tolist())),
        static_objects=[
            StaticObject(s["name"], s.get("mesh"), s.get("category", ""), s.get("triangles", 0))
            for s in raw.get("static_objects", [])
        ],
        locators=[
            Locator(
                name=l["name"],
                pose_world=_mat(l["pose_world"]),
                is_weld=bool(l.get("is_weld", False)),
                gun_opening_arrive=float(l.get("gun_opening_arrive", 0.0)),
                gun_opening_leave=float(l.get("gun_opening_leave", 0.0)),
            )
            for l in raw.get("locators", [])
        ],
        start_state=dict(raw.get("start_state", {})),
        raw=raw,
    )
