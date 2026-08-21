"""Where a point sits in the collision model, and where it sits in the real one.

The question this answers is the one the exported geometry raises and cannot settle: the
planner says a place is solid, the CAD says it is empty, and neither the manifest nor the
hulls on their own say which shape is responsible.  Given a point it names the convex
pieces that contain it, the link and cache entry they came from, and how far the nearest
real material actually is.  A large gap between the two is a hull bridging a void.

Points are given in a locator's frame, because that is how the discrepancy is seen -- the
robot is jumped to a weld and something is in the way a fixed distance along the tool's own
axes -- and a world position would have to be worked out by hand first.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import meshprep
from .hullexport import _outward, _link_hulls

# How many triangles of a link are measured exactly.  The box bound below is cheap and
# runs over every triangle; only the closest handful can win, and measuring those properly
# costs nothing next to loading the mesh.
EXACT_CANDIDATES = 256


def parse_point(text: str) -> tuple[str | None, np.ndarray]:
    """``"locator:x,y,z"`` or ``"x,y,z"`` -> the frame's name and the offset in it.

    A bare triple is world.  An axis may be left empty as ``x,,z``: a discrepancy spotted
    on the tool's axes is usually quoted on two of them, and writing an explicit zero into
    the third is exactly the sort of thing that ends up in the wrong slot.
    """
    frame, _, coords = text.rpartition(":")
    parts = coords.split(",")
    if len(parts) != 3:
        raise ValueError(f"expected 'locator:x,y,z' or 'x,y,z', got {text!r}")
    xyz = np.array([float(p) if p.strip() else 0.0 for p in parts], dtype=float)
    return (frame or None), xyz


def world_point(man, spec: str) -> tuple[np.ndarray, str]:
    """Resolve a point spec to world coordinates in the manifest's units."""
    frame, xyz = parse_point(spec)
    if frame is None:
        return xyz, "world"
    for loc in man.locators:
        if loc.name == frame:
            pose = np.array(loc.pose_world, dtype=float)
            offset = f"({xyz[0]:g}, {xyz[1]:g}, {xyz[2]:g})"
            return pose[:3, :3] @ xyz + pose[:3, 3], f"{frame} + {offset}"
    known = ", ".join(loc.name for loc in man.locators[:6])
    raise ValueError(f"no locator named {frame!r}; the manifest has {known}, ...")


def _point_triangle_distance(p: np.ndarray, A: np.ndarray, B: np.ndarray,
                             C: np.ndarray) -> np.ndarray:
    """Distance from one point to each of many triangles.

    The closest point on a triangle lies in its interior, on an edge, or at a vertex.  This
    computes the interior projection and each of the three edge projections, clamping the
    edge parameters to their own segments -- which folds the vertex cases in, since a
    clamped parameter lands on one -- and takes the nearest.  Clamping rather than
    branching keeps it a single vectorised pass over every triangle at once.
    """
    def edge(P, Q):
        d = Q - P
        length = np.einsum("ij,ij->i", d, d)
        t = np.clip(np.einsum("ij,ij->i", p - P, d) / np.where(length > 0, length, 1.0),
                    0.0, 1.0)
        return np.linalg.norm(P + t[:, None] * d - p, axis=1)

    normal = np.cross(B - A, C - A)
    twice_area = np.linalg.norm(normal, axis=1)
    safe = np.where(twice_area > 0, twice_area, 1.0)
    unit = normal / safe[:, None]
    foot = p - np.einsum("ij,ij->i", unit, p - A)[:, None] * unit
    # Barycentric coordinates of the projection, as signed area ratios.
    u = np.einsum("ij,ij->i", np.cross(B - foot, C - foot), unit) / safe
    v = np.einsum("ij,ij->i", np.cross(C - foot, A - foot), unit) / safe
    interior = (twice_area > 0) & (u >= 0) & (v >= 0) & (1.0 - u - v >= 0)
    rim = np.minimum(np.minimum(edge(A, B), edge(B, C)), edge(C, A))
    return np.where(interior, np.linalg.norm(foot - p, axis=1), rim)


def nearest_material(man, meshes: dict[str, str], point: np.ndarray,
                     transforms: dict[str, np.ndarray]) -> list[tuple[float, str]]:
    """Distance from ``point`` to the nearest source-mesh triangle of each link.

    The *source* mesh, deliberately: the collision geometry is the thing in doubt, so the
    only comparison worth making is against the CAD the study was built from.
    """
    out: list[tuple[float, str]] = []
    for name, path in sorted(meshes.items()):
        if not os.path.isfile(path):
            continue
        V, F = meshprep.load_obj(path)
        if not len(F):
            continue
        placement = transforms.get(name)
        if placement is not None:
            V = V @ placement[:3, :3].T + placement[:3, 3]
        corners = np.stack((V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]))
        lo, hi = corners.min(axis=0), corners.max(axis=0)
        # A cheap lower bound over every triangle, then the exact measurement on the few
        # that could possibly win.  The bound never overstates, so nothing close is lost.
        bound = np.linalg.norm(np.maximum(np.maximum(lo - point, point - hi), 0.0), axis=1)
        cut = min(EXACT_CANDIDATES, len(bound) - 1)
        keep = F[bound <= float(np.partition(bound, cut)[cut])]
        distance = _point_triangle_distance(point, V[keep[:, 0]], V[keep[:, 1]],
                                            V[keep[:, 2]])
        out.append((float(distance.min()), name))
    return sorted(out)


def _depth_inside(V: np.ndarray, tris, point: np.ndarray) -> float | None:
    """How far inside this convex piece the point is, or ``None`` if it is outside."""
    F = np.asarray(tris, dtype=np.int64)
    if not len(F):
        return None
    normals, offsets = _outward(V, F)
    gap = normals @ point - offsets
    return None if float(gap.max()) > 0.0 else float(-gap.max())


def report(env, man, spec: str, log=print) -> None:
    """Say which collision hulls contain the point and how far the real material is."""
    point, described = world_point(man, spec)
    log(f"probing {point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f} in world coordinates "
        f"-- {described}")

    scene = env.getSceneGraph()
    # The bindings hand back a temporary state; letting it die leaves link_transforms
    # dangling and segfaults the interpreter, so keep a reference to it.
    state = env.getState()
    transforms = state.link_transforms
    unit = 1.0 / man.scale                      # metres back to the manifest's units

    hits: list[tuple[float, str, int]] = []
    for name in list(env.getLinkNames()):
        link = scene.getLink(name)
        if link is None or not len(link.collision):
            continue
        for index, (V, tris) in enumerate(_link_hulls(scene, transforms, name, unit)):
            depth = _depth_inside(V, tris, point)
            if depth is not None:
                hits.append((depth, name, index))

    if not hits:
        log("  the point is inside no collision hull, so whatever blocks the robot here "
            "is a different shape from the one being looked at")
    else:
        log(f"  inside {len(hits)} collision hull{'' if len(hits) == 1 else 's'}:")
        for depth, name, index in sorted(hits, reverse=True)[:12]:
            log(f"    {name:<28} piece {index:<5} {depth:8.1f} mm inside its surface")

    meshes = {s.name: man.mesh_path(s.mesh) for s in man.static_objects if s.mesh}
    placed: dict[str, np.ndarray] = {}
    for link in man.all_links():
        if not link.mesh:
            continue
        meshes[link.name] = man.mesh_path(link.mesh)
        try:                                    # a moving link is compared where it is now
            matrix = np.array(transforms[link.name].matrix(), dtype=float)
            matrix[:3, 3] *= unit
            placed[link.name] = matrix
        except Exception:
            pass

    near = nearest_material(man, meshes, point, placed)
    if near:
        log("  nearest real material in the source CAD:")
        for distance, name in near[:4]:
            log(f"    {name:<28} {distance:8.1f} mm away")
    blocking = {name for _, name, _ in hits}
    for distance, name in near:
        if name in blocking and distance > 1.0:
            log(f"  ! {name} has a hull containing this point but no material within "
                f"{distance:.1f} mm of it -- that hull is bridging a void")

    _provenance(man, blocking, log)


def _provenance(man, links: set[str], log=print) -> None:
    """What the decomposition actually did to the links implicated, from the cache index."""
    path = os.path.join(man.directory, meshprep.CACHE_DIRNAME, "index.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            index = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return
    for name in sorted(links):
        entry = index.get(name)
        if not entry:
            continue
        stats = entry.get("stats", {})
        settings = entry.get("settings", [])
        cell = settings[3] if len(settings) > 3 else "?"
        fill = settings[4] if len(settings) > 4 else "?"
        radius = next((s for s in settings[5:] if isinstance(s, (int, float))), 0)
        log(f"  {name}: {stats.get('shells')} shells -> {stats.get('shells_kept')} kept, "
            f"{stats.get('shells_refined')} refined at a {cell} mm cell, fill gate {fill}"
            + (f", focus radius {radius} mm" if radius else ", no focus radius"))
        if not stats.get("shells_refined"):
            log(f"  ! nothing in {name} was refined at all, so every one of its shells is "
                f"a single hull of its whole shape")
