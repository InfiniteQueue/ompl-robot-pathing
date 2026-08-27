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

import json
import os
import time

import numpy as np

from .meshprep import CACHE_DIRNAME

DIRNAME = "collision_geometry"

# The index that lets a re-export skip links whose file is already correct.  It lives with
# the decomposition cache rather than in the export directory: the export directory is the
# caller's and holds deliverables, and this is bookkeeping.  Keyed by absolute output path,
# so exporting the same scene to several directories does not have them fight over one
# entry.
CACHE_NAME = "export_index.json"
# Bumped when something that changes an exported file changes and the key would not
# otherwise notice it -- the hull routine, the OBJ header, the units written.
_CACHE_VERSION = 1

# _insert below is an incremental insertion sized for the job it has here: the points
# Tesseract hands back are already a hull's vertices, forty or so and well separated.  Fed
# a raw concave shell instead -- many points sitting within eps of a face -- it does not
# merely slow down, it diverges.  One 362-point shell of Assy_ST200_RH came back with
# 24,670,533 triangles after 109 seconds, where a hull of n points holds at most 2n - 4,
# here 720.  Size is not the trigger: a 1270-point shell alongside it hulled in 0.1s.
#
# So a large point set is cut down first, and cut down provably: the hull of a subset is
# contained in the hull of the whole, so a point inside that subset hull cannot be a vertex
# of the full one.  Taking the extremes along a spread of directions gives a subset hull
# already close to the answer, and what survives the cull is the true vertex set plus
# whatever lies within TOLERANCE of a face.  That shell then hulls in 0.029s.
#
# The cull is not free -- it is itself two hulls of up to 2 * DIRECTIONS points -- so it is
# only worth doing where the insertion might be in trouble.  Below HULL_CULL_MIN the point
# set is no bigger than the seed the cull would build, and the insertion runs alone.  The
# hulls this module exports are normally well under it; callers feeding raw shells in pass
# a lower bar.
DIRECTIONS = 160
TOLERANCE = 1e-6            # of the shell's own diagonal, so micrometres on a 300 mm part
HULL_CULL_MIN = 96


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


def _insert(V: np.ndarray) -> list[tuple[int, int, int]]:
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
    # Study coordinates run to six figures while the parts are a few hundred units across,
    # so every height below would otherwise be a difference of two large numbers, and the
    # decision "is this point outside that face" would be made in the round-off. Only the
    # face indices are returned, so shifting the points here changes nothing but precision.
    V = V - V.mean(axis=0)

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

    seeded = {a, b, c, d}
    # Furthest from the centre first. Adding the points that define the shape early means
    # most of the rest fail their single height test and never touch the face list, where
    # in file order the hull is rebuilt over and over as it grows outwards. The hull that
    # comes out is the same either way; only the work done to reach it changes.
    for index in np.argsort(-np.linalg.norm(V, axis=1)):
        p = int(index)
        if p in seeded:
            continue
        seen = (N @ V[p] - O) > eps
        if not seen.any():
            continue
        rim = F[seen]
        edges = np.concatenate([rim[:, [0, 1]], rim[:, [1, 2]], rim[:, [2, 0]]])
        # Every face is wound the same way round, so an edge interior to the lit patch
        # appears once in each direction and an edge on its rim appears only once. Finding
        # the rim is then a lookup of one integer key per edge, rather than a lexsort of
        # sorted vertex pairs -- the same answer for a fraction of the cost.
        key = edges[:, 0] * n + edges[:, 1]
        horizon = edges[~np.isin(edges[:, 1] * n + edges[:, 0], key)]
        if len(horizon) < 3:
            continue
        fresh = np.column_stack([horizon, np.full(len(horizon), p, dtype=np.int64)])
        Nf, Of = _outward(V, fresh)
        F = np.concatenate([F[~seen], fresh])
        N = np.concatenate([N[~seen], Nf])
        O = np.concatenate([O[~seen], Of])
    return [(int(i), int(j), int(k)) for i, j, k in F]


def _directions(n: int) -> np.ndarray:
    """``n`` roughly equidistant unit vectors, by the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.column_stack([np.cos(theta) * np.sin(phi),
                            np.sin(theta) * np.sin(phi), np.cos(phi)])


_DIRS = _directions(DIRECTIONS)


def _sane(tris, n: int) -> bool:
    """Euler's bound: a convex hull of ``n`` points has at most ``2n - 4`` faces."""
    return len(tris) <= max(2 * n - 4, 2)


def _hull(V: np.ndarray, cull_min: int | None = None) -> list[tuple[int, int, int]]:
    """Convex hull of ``V``, with the interior of the point set removed first.

    See HULL_CULL_MIN for why the cull is conditional and what it is defending
    against.  ``cull_min`` lowers that bar for a caller that knows its input is raw
    concave geometry rather than the hull vertices this module usually sees.
    """
    n = len(V)
    span = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0))) if n else 0.0
    if n < (HULL_CULL_MIN if cull_min is None else cull_min) or span <= 0.0:
        return _insert(V)
    proj = V @ _DIRS.T
    seed = np.unique(np.concatenate([proj.argmax(axis=0), proj.argmin(axis=0)]))
    if len(seed) < 4:
        return _insert(V)
    tris = _insert(V[seed])
    if not tris:
        return _insert(V)
    N, O = _outward(V[seed], np.asarray(tris, dtype=np.int64))
    outside = np.zeros(n, dtype=bool)
    for lo in range(0, n, 4096):        # the height table is n x faces; keep it small
        chunk = V[lo:lo + 4096]
        outside[lo:lo + 4096] = ((chunk @ N.T) - O).max(axis=1) > TOLERANCE * span
    keep = np.unique(np.concatenate([seed, np.nonzero(outside)[0]]))
    final = _insert(V[keep])
    if not _sane(final, len(keep)):
        # The cull did not remove enough to keep the insertion in hand.  The seed hull
        # is a valid hull of a subset, so falling back to it under-claims rather than
        # returning nonsense -- but it has never been needed, so say so if it ever is.
        print("    ! degenerate hull on %d points, falling back to the %d-point seed"
              % (len(keep), len(seed)), flush=True)
        return [(int(seed[a]), int(seed[b]), int(seed[c])) for a, b, c in tris]
    return [(int(keep[a]), int(keep[b]), int(keep[c])) for a, b, c in final]


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


def _index_path(man) -> str:
    return os.path.join(man.directory, CACHE_DIRNAME, CACHE_NAME).replace("\\", "/")


def _load_index(man) -> dict:
    try:
        with open(_index_path(man), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_index(man, index: dict) -> None:
    try:
        path = _index_path(man)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(index, fh, indent=1)
    except OSError:
        pass                # an index that cannot be written is a slow export, not a fault


def _link_key(man, scene, transforms, name: str, unit: float) -> list:
    """What an exported file depends on, cheap enough to be worth asking every time.

    Deliberately not a hash of the vertices.  Reading them goes through :func:`_vertices`,
    which unwraps the binding's points one at a time, and paying that for every link is a
    good part of the work the index exists to avoid.  The decomposition's own output stands
    in for the geometry instead: ``meshprep.prepare`` rewrites ``<link>.obj`` whenever any
    setting that could change the shells changes, so that file's size and mtime carry all
    of it.

    The transforms are here because :func:`export` writes world coordinates at the current
    state, so the same hulls at a different start pose are a different file.  Each collision
    origin is here too: those place the pieces against each other inside the one file.

    The gap this leaves: a link with no entry in the decomposition cache, whose geometry
    changed without its transforms moving, reads as current.  Deleting its OBJ rebuilds it,
    a missing file never being reused.
    """
    try:
        st = os.stat(os.path.join(man.directory, CACHE_DIRNAME, f"{name}.obj"))
        sig = [int(st.st_size), int(st.st_mtime)]
    except OSError:
        sig = None
    try:
        tf = np.round(np.array(transforms[name].matrix(), dtype=float), 9).tolist()
    except Exception:
        tf = "identity"
    link = scene.getLink(name)
    origins = [np.round(np.array(c.origin.matrix(), dtype=float), 9).tolist()
               for c in link.collision]
    return [_CACHE_VERSION, sig, round(float(unit), 9), tf, origins]


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

    A link whose file is already correct is left alone rather than rewritten -- see
    :func:`_link_key` for what "correct" is judged on.  Rebuilding one is not cheap, and
    the files land in a directory of the caller's that they may well be looking at.
    """
    out_dir = os.path.join(directory, DIRNAME).replace("\\", "/")
    os.makedirs(out_dir, exist_ok=True)
    scene = env.getSceneGraph()
    # The SWIG bindings hand back a temporary scene state; letting it die leaves
    # link_transforms dangling and segfaults the interpreter, so keep a reference.
    state = env.getState()
    transforms = state.link_transforms
    unit = 1.0 / man.scale                      # metres back to the manifest's units

    index = _load_index(man)
    written = reused = 0
    for name in list(env.getLinkNames()):
        t0 = time.time()
        link = scene.getLink(name)
        if link is None or not len(link.collision):
            continue

        path = os.path.join(out_dir, f"{name}.obj").replace("\\", "/")
        key = _link_key(man, scene, transforms, name, unit)
        entry = index.get(path)
        if entry is not None and entry.get("key") == key and os.path.isfile(path):
            log(f"  = {name:<22} already current ({entry['pieces']} convex pieces)")
            reused += 1
            continue

        log(f"  . {name:<22} hulling {len(link.collision)} collision geometries; each "
            f"convex piece is rebuilt from its vertices")
        hulls = _link_hulls(scene, transforms, name, unit)
        if not hulls:
            continue

        faces = _write_obj(path, [
            f"weldpath collision geometry for link {name}",
            f"{len(hulls)} convex pieces, world coordinates at the current state, "
            f"units={man.units}",
        ], [("hull_%05d" % i, h) for i, h in enumerate(hulls)])
        log("  > %-22s %d convex pieces, %d triangles (%.1fs)"
            % (name, len(hulls), faces, time.time() - t0))
        index[path] = {"key": key, "pieces": len(hulls), "triangles": faces}
        written += 1

    _save_index(man, index)
    log(f"wrote collision geometry for {written} links to {out_dir}"
        + (f", leaving {reused} already current" if reused else ""))
    return out_dir
