"""Write out the collision geometry the planner actually sees.

The ``convex_cache`` OBJs are the *input* to the decomposition, not its output: each one
holds the original concave triangles of a shell under an ``o shell_NNNNN`` group, and
Tesseract builds one convex hull per group at load time because the URDF tags the mesh
``tesseract:make_convex="true"``.  Opening a cache file therefore shows the faithful CAD
and tells you nothing about how coarse the collision model really is -- the bridged
throats and filled recesses that make a reachable weld read as a collision are invisible
there.

This module reads the hulls back out of the loaded environment, so what it writes is the
geometry Bullet is testing, not a reconstruction of it.  Vertices come out in world
coordinates at the environment's current state and in the manifest's own units, which
means the exported OBJs drop straight on top of the source meshes in a viewer.
"""
from __future__ import annotations

import os
import time

import numpy as np

DIRNAME = "collision_geometry"


def _ring(P: np.ndarray) -> list[int]:
    """Indices of the 2D convex hull of ``P``, counter-clockwise (monotone chain)."""
    order = np.lexsort((P[:, 1], P[:, 0]))

    def turn(a, b, c) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    def half(seq):
        out: list[int] = []
        for i in seq:
            while len(out) >= 2 and turn(P[out[-2]], P[out[-1]], P[i]) <= 0.0:
                out.pop()
            out.append(int(i))
        return out

    lower, upper = half(order), half(order[::-1])
    return lower[:-1] + upper[:-1]


def _flat(V: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangulate a point set that has no volume, as a two-sided sheet."""
    centred = V - V.mean(axis=0)
    _, _, basis = np.linalg.svd(centred, full_matrices=True)
    loop = _ring(centred @ basis[:2].T)
    if len(loop) < 3:
        return []
    front = [(loop[0], loop[k], loop[k + 1]) for k in range(1, len(loop) - 1)]
    return front + [(a, c, b) for a, b, c in front]


def _outward(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit normals and plane offsets for a triangle array."""
    normal = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    length = np.linalg.norm(normal, axis=1)
    normal = normal / np.where(length > 0, length, 1.0)[:, None]
    return normal, np.einsum("ij,ij->i", normal, V[F[:, 0]])


def _hull(V: np.ndarray) -> list[tuple[int, int, int]]:
    """Convex hull of ``V`` as a triangle list, by incremental insertion.

    The points handed back by Tesseract are already a hull's vertices, so hulling them
    again reproduces the collision shape exactly rather than approximating it.  It is done
    here rather than read from the mesh because the face lists that come with those
    vertices are not usable: on a 39-vertex hull the faces name 36 vertices apiece, and
    those vertices sit up to a metre off the plane of their own face.
    """
    n = len(V)
    if n < 4:
        return _flat(V) if n == 3 else []
    span = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    eps = max(1e-12, 1e-9 * span)

    a = int(np.argmin(V[:, 0]))
    b = int(np.argmax(np.linalg.norm(V - V[a], axis=1)))
    if np.linalg.norm(V[b] - V[a]) <= eps:
        return []
    c = int(np.argmax(np.linalg.norm(np.cross(V - V[a], V[b] - V[a]), axis=1)))
    normal = np.cross(V[b] - V[a], V[c] - V[a])
    if np.linalg.norm(normal) <= eps:
        return []
    normal = normal / np.linalg.norm(normal)
    d = int(np.argmax(np.abs((V - V[a]) @ normal)))
    if abs(float((V[d] - V[a]) @ normal)) <= eps:
        return _flat(V)

    seed = np.array([(a, b, c), (a, c, d), (a, d, b), (b, d, c)], dtype=np.int64)
    inner = V[[a, b, c, d]].mean(axis=0)
    N, O = _outward(V, seed)
    behind = (N @ inner - O) > 0.0                  # normal must face away from the inside
    seed[behind] = seed[behind][:, ::-1]
    F = seed
    N, O = _outward(V, F)

    for p in range(n):
        if p in (a, b, c, d):
            continue
        seen = (N @ V[p] - O) > eps
        if not seen.any():
            continue
        rim = F[seen]
        edges = np.concatenate([rim[:, [0, 1]], rim[:, [1, 2]], rim[:, [2, 0]]])
        _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0,
                                       return_inverse=True, return_counts=True)
        horizon = edges[counts[inverse.reshape(-1)] == 1]
        if not len(horizon):
            continue
        fresh = np.column_stack([horizon, np.full(len(horizon), p, dtype=np.int64)])
        Nf, Of = _outward(V, fresh)
        F = np.concatenate([F[~seen], fresh])
        N = np.concatenate([N[~seen], Nf])
        O = np.concatenate([O[~seen], Of])
    return [(int(i), int(j), int(k)) for i, j, k in F]


def _vertices(mesh) -> np.ndarray:
    """Mesh vertices as an (n, 3) array.

    The bindings return a vector of Eigen vectors that numpy cannot convert wholesale, so
    the points have to be unwrapped one at a time.
    """
    pts = mesh.getVertices()
    return np.array([np.asarray(pts[i]).reshape(3) for i in range(len(pts))],
                    dtype=float) if len(pts) else np.zeros((0, 3))


def _parts(geometry) -> list:
    """Every mesh inside one collision geometry, whatever wrapper it arrived in."""
    getter = getattr(geometry, "getMeshes", None)
    if getter is not None:
        try:
            return list(getter())
        except Exception:
            pass
    return [geometry] if hasattr(geometry, "getVertices") else []


def _link_hulls(scene, transforms, name: str,
                unit: float) -> list[tuple[np.ndarray, list[tuple[int, int, int]]]]:
    """Every convex piece of one link's collision geometry, placed in the world.

    The vertices come back already hulled: Tesseract's own face lists are unusable -- on a
    39-vertex hull they name 36 vertices per face, sitting up to 0.92 m off the plane of
    the face they belong to -- but the vertices it returns are the hull's own, so hulling
    them here reproduces the collision shape exactly.
    """
    link = scene.getLink(name)
    if link is None or not len(link.collision):
        return []
    try:
        link_tf = np.array(transforms[name].matrix(), dtype=float)
    except Exception:
        link_tf = np.eye(4)
    hulls: list[tuple[np.ndarray, list[tuple[int, int, int]]]] = []
    for col in link.collision:
        tf = link_tf @ np.array(col.origin.matrix(), dtype=float)
        for part in _parts(col.geometry):
            V = _vertices(part)
            if not len(V):
                continue
            tris = _hull(V)
            if not tris:
                continue
            hulls.append(((V @ tf[:3, :3].T + tf[:3, 3]) * unit, tris))
    return hulls


def _write_obj(path: str, header: list[str], groups) -> int:
    """Write named convex pieces as OBJ ``o`` groups.  Returns the triangle count."""
    faces = 0
    with open(path, "w", encoding="utf-8") as fh:
        for line in header:
            fh.write("# %s\n" % line)
        offset = 1
        for group, (V, tris) in groups:
            fh.write("o %s\n" % group)
            for x, y, z in V:
                fh.write("v %.6f %.6f %.6f\n" % (x, y, z))
            for a, b, c in tris:
                fh.write("f %d %d %d\n" % (a + offset, b + offset, c + offset))
            offset += len(V)
            faces += len(tris)
    return faces


def _safe(name: str) -> str:
    """A link or locator name reduced to something usable as a filename."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:60]


def export_pair(env, man, directory: str, links: tuple[str, str], tag: str,
                log=print) -> str | None:
    """Write the two links of a blocked pair, as they sit right now, into one OBJ.

    Called from the placement diagnosis, where the environment is already holding the
    joint state that failed -- so what lands in the file is the geometry the collision
    check actually objected to, in world coordinates, rather than a reconstruction of it.
    Both links go into the one file so the overlap can be looked at without aligning two.
    """
    out_dir = os.path.join(directory, DIRNAME).replace("\\", "/")
    os.makedirs(out_dir, exist_ok=True)
    scene = env.getSceneGraph()
    state = env.getState()      # held: see export() on the dangling-temporary trap
    transforms = state.link_transforms
    unit = 1.0 / man.scale

    groups = []
    for name in links:
        hulls = _link_hulls(scene, transforms, name, unit)
        groups += [(f"{_safe(name)}_hull_{i:05d}", h) for i, h in enumerate(hulls)]
    if not groups:
        return None

    path = os.path.join(out_dir, f"blocked_{_safe(tag)}.obj").replace("\\", "/")
    faces = _write_obj(path, [
        f"weldpath blocked pose: {links[0]} against {links[1]}",
        tag,
        f"{len(groups)} convex pieces, world coordinates at the failing joint state, "
        f"units={man.units}",
    ], groups)
    log(f"      wrote the blocking geometry ({len(groups)} pieces, {faces} triangles) "
        f"to {path}")
    return path


def export(env, man, directory: str, log=print) -> str:
    """Write one OBJ per link into ``<directory>/collision_geometry``.

    Returns the directory written.  Each convex piece is its own ``o hull_NNNNN`` group,
    so the group count is the number of shapes that link contributes to a collision check.
    """
    out_dir = os.path.join(directory, DIRNAME).replace("\\", "/")
    os.makedirs(out_dir, exist_ok=True)
    scene = env.getSceneGraph()
    # The SWIG bindings hand back a temporary scene state; letting it die leaves
    # link_transforms dangling and segfaults the interpreter, so keep a reference.
    state = env.getState()
    transforms = state.link_transforms
    unit = 1.0 / man.scale                      # metres back to the manifest's units

    written = 0
    for name in list(env.getLinkNames()):
        t0 = time.time()
        link = scene.getLink(name)
        if link is None or not len(link.collision):
            continue

        log(f"  . {name:<22} hulling {len(link.collision)} collision geometries; each "
            f"convex piece is rebuilt from its vertices")
        hulls = _link_hulls(scene, transforms, name, unit)
        if not hulls:
            continue

        path = os.path.join(out_dir, f"{name}.obj").replace("\\", "/")
        faces = _write_obj(path, [
            f"weldpath collision geometry for link {name}",
            f"{len(hulls)} convex pieces, world coordinates at the current state, "
            f"units={man.units}",
        ], [("hull_%05d" % i, h) for i, h in enumerate(hulls)])
        log("  > %-22s %d convex pieces, %d triangles (%.1fs)"
            % (name, len(hulls), faces, time.time() - t0))
        written += 1

    log(f"wrote collision geometry for {written} links to {out_dir}")
    return out_dir
