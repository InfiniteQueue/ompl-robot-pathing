"""Which way is "back" from a locator.

The retreat direction is the locator axis that points most nearly at the middle of the
gun's own bulk, with the robot standing where that locator puts it.  The reasoning is
that the gun body sits on the opposite side of the tool from whatever the tool is pointed
at, so walking the pose that way walks it out of the pocket it is in rather than deeper
into it -- and it does so in the locator's own frame, which is the frame the approach was
authored in, instead of some world direction that happens to be up.

Only the six axis directions are considered, signs included.  A free direction would be
the vector to the centre itself, but the locator frame is the one the operator reasons in
and a pure axis is the answer they would give; the axis is also stable under small
changes in the gun's pose, where a free vector wanders.
"""
from __future__ import annotations

import numpy as np

# The six signed axes of a frame, as (index, sign).  Written out rather than derived so
# the order of preference on a tie is fixed and readable: +X, -X, +Y, -Y, +Z, -Z.
SIGNED_AXES = tuple((i, s) for i in range(3) for s in (1.0, -1.0))


def link_bounds_world(cell, link_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """World-space axis-aligned bounds of a link's collision geometry, in metres.

    Read from the environment rather than from the source meshes: the question is where
    the geometry the planner is checking against actually sits, and the environment holds
    it already placed.  ``None`` where the link carries no collision geometry, which is
    every frame-only link in the chain.

    Assumes the state is already loaded -- the caller sets it once and asks about several
    links, and reloading per link would cost more than the query.
    """
    link = cell.env.getLink(link_name)
    if link is None or not len(link.collision):
        return None
    T = np.array(cell.env.getLinkTransform(link_name).matrix(), dtype=float)
    R, t = T[:3, :3], T[:3, 3]
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for element in link.collision:
        geom = element.geometry
        local = np.array(element.origin.matrix(), dtype=float)
        meshes = geom.getMeshes() if hasattr(geom, "getMeshes") else [geom]
        for mesh in meshes:
            if not hasattr(mesh, "getVertices"):
                continue                # a primitive: no vertices to read, and the gun is
                                        # meshes throughout, so skipping is not a loss
            verts = mesh.getVertices()
            n = len(verts)
            if not n:
                continue
            # Strided like ``cell._geometry_extent``: this is a centre-of-bulk estimate
            # feeding a choice between six axes, not a measurement, and the axis does not
            # change for want of every vertex of a 200k-triangle casting.
            step = max(1, n // 2000)
            P = np.array([np.asarray(verts[i], dtype=float).ravel()
                          for i in range(0, n, step)])
            P = P @ local[:3, :3].T + local[:3, 3]      # link frame
            P = P @ R.T + t                             # world
            lo = np.minimum(lo, P.min(axis=0))
            hi = np.maximum(hi, P.max(axis=0))
    if not np.all(np.isfinite(lo)):
        return None
    return lo, hi


def bulk_centre_world(cell, link_names: list[str], q: np.ndarray) -> np.ndarray | None:
    """Centre of the bounding box enclosing every named link, in metres.

    One box over the whole group rather than a box each: the gun is several links that
    move together and the question is where its mass sits as a body.  ``None`` when none
    of the links carries geometry.
    """
    cell.set_state(q)
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for name in link_names:
        bounds = link_bounds_world(cell, name)
        if bounds is None:
            continue
        lo = np.minimum(lo, bounds[0])
        hi = np.maximum(hi, bounds[1])
    if not np.all(np.isfinite(lo)):
        return None
    return (lo + hi) / 2.0


def gun_link_names(man) -> list[str]:
    """The links whose bulk defines "back", most appropriate first.

    The gun, where there is one: every link of every device past the robot.  Where there
    is none, the last robot link carrying geometry, which is the wrist casting -- the
    nearest thing to a tool body the cell has, and the part that will foul something if
    the pose is walked the wrong way.
    """
    gun = [link.name for device in man.devices[1:] for link in device.links if link.mesh]
    if gun:
        return gun
    robot = [link.name for link in man.robot.links if link.mesh]
    return robot[-1:]


def retreat_axis(cell, man, pose_world: np.ndarray, q: np.ndarray
                 ) -> tuple[np.ndarray, str] | None:
    """The world direction to retreat along from ``pose_world``, and its name.

    ``pose_world`` is the locator's 4x4 in manifest units; ``q`` is the configuration the
    robot reaches it in.  Returns a unit world vector and a label like ``"-Y"`` for the
    log, or ``None`` where the gun's bulk cannot be located -- which leaves the caller to
    skip the method rather than retreat in an arbitrary direction.
    """
    centre = bulk_centre_world(cell, gun_link_names(man), q)
    if centre is None:
        return None
    T = np.asarray(pose_world, dtype=float)
    origin = T[:3, 3] * man.scale               # locator frames are in manifest units
    to_centre = centre - origin
    norm = float(np.linalg.norm(to_centre))
    if norm < 1e-9:
        # The tool sits at the middle of the gun's own box.  There is no direction in
        # this, and picking one anyway would be picking it out of rounding noise.
        return None
    to_centre = to_centre / norm
    R = T[:3, :3]
    best, best_dot = None, -np.inf
    for i, sign in SIGNED_AXES:
        axis = sign * R[:, i]
        axis = axis / float(np.linalg.norm(axis))
        dot = float(axis @ to_centre)
        if dot > best_dot:
            best, best_dot = (axis, f"{'+' if sign > 0 else '-'}{'XYZ'[i]}"), dot
    return best
