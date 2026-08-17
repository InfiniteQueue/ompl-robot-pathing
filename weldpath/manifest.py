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

    @property
    def is_prismatic(self) -> bool:
        return self.type == "prismatic"


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
    def gun_joint_name(self) -> str | None:
        for d in self.devices[1:]:
            for j in d.joints:
                return j.name
        return None

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
