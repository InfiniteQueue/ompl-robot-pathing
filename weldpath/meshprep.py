"""Collision-geometry preparation.

Tesseract can collision-check the raw CAD meshes, but at roughly 700 ms per discrete
check the sampling planners never finish.  Convex geometry costs 1-5 ms instead.

A *single* convex hull per link is not an option: the weld gun is C-shaped, and one hull
fills the throat the panel has to sit in, so every weld would report a collision.  The
fix is a convex decomposition.  The CAD exports are flat triangle soups with no ``o``/``g``
grouping, but they are assemblies -- the gun is 948 disconnected shells, a panel is 127.
Splitting on connected components and writing each shell as its own ``o`` group makes
Tesseract build one ConvexMesh per shell, which keeps the gun throat open.

Output meshes are written in metres (URDF units) into a cache directory, and reused when
the source file is unchanged.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

CACHE_DIRNAME = "convex_cache"


# ---------------------------------------------------------------------------
# OBJ IO
# ---------------------------------------------------------------------------
def load_obj(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read vertices and triangles from an OBJ.  Polygons are fanned to triangles."""
    verts: list[tuple[str, str, str]] = []
    faces: list[tuple[int, int, int]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                p = line.split()
                verts.append((p[1], p[2], p[3]))
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) for t in line.split()[1:]]
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
    V = np.array(verts, dtype=np.float64) if verts else np.zeros((0, 3))
    F = (np.array(faces, dtype=np.int64) - 1) if faces else np.zeros((0, 3), dtype=np.int64)
    return V, F


def connected_shells(V: np.ndarray, F: np.ndarray, weld_tol: float = 1e-6) -> np.ndarray:
    """Label each triangle with the id of the connected shell it belongs to.

    Vertices are welded on a rounded coordinate key first, because CAD tessellation
    routinely duplicates vertices along shared edges; without welding a single solid
    would shatter into thousands of fragments.
    """
    if len(F) == 0:
        return np.zeros(0, dtype=np.int64)
    key = np.round(V / weld_tol).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    n = int(inv.max()) + 1
    parent = np.arange(n, dtype=np.int64)

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:            # path compression
            parent[a], a = root, parent[a]
        return root

    Fw = inv[F]
    for tri in Fw:
        ra = find(int(tri[0]))
        for other in (int(tri[1]), int(tri[2])):
            ro = find(other)
            if ro != ra:
                parent[ro] = ra
    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    return roots[Fw[:, 0]]


def decompose_to_obj(src: str, dst: str, scale: float, dedupe: bool = True,
                     min_extent: float = 40.0, max_shells: int = 80) -> dict:
    """Split ``src`` into connected shells and write them as ``o`` groups into ``dst``.

    Every shell becomes a convex hull, and each hull is a collision pair to test, so the
    raw shell count sets the cost of a collision check.  Two filters keep that in hand:
    shells whose bounding-box diagonal is under ``min_extent`` are dropped (fasteners and
    small internal parts, which sit inside the envelope of the bodies around them), and
    only the ``max_shells`` largest survive.  On the sample cell this takes the gun from
    948 shells to 80 and a collision check from 2.8 ms to 1.6 ms, which is the difference
    between the sampling planner timing out and solving in a few seconds.
    """
    t0 = time.time()
    V, F = load_obj(src)
    labels = connected_shells(V, F)
    unique = np.unique(labels)

    candidates: list[tuple[float, np.ndarray, np.ndarray, tuple]] = []
    for lab in unique:
        tris = F[labels == lab]
        vids = np.unique(tris)
        P = V[vids]
        extent = P.max(axis=0) - P.min(axis=0)
        diagonal = float(np.linalg.norm(extent))
        centre = tuple(np.round(P.mean(axis=0), 4))
        candidates.append((diagonal, vids, tris, centre))

    kept_all = len(candidates)
    if min_extent > 0:
        candidates = [c for c in candidates if c[0] >= min_extent]
    dropped_small = kept_all - len(candidates)

    groups: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set = set()
    candidates.sort(key=lambda c: -c[0])
    for diagonal, vids, tris, centre in candidates:
        if dedupe:
            # CAD exports frequently contain the same solid twice; drop exact repeats.
            sig = (len(tris), centre, round(diagonal, 4))
            if sig in seen:
                continue
            seen.add(sig)
        groups.append((vids, tris))
        if max_shells and len(groups) >= max_shells:
            break

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write("# weldpath convex-decomposition source=%s\n" % os.path.basename(src))
        fh.write("# %d triangles, %d shells (%d after dedupe), units=metres\n"
                 % (len(F), len(unique), len(groups)))
        offset = 1
        for gi, (vids, tris) in enumerate(groups):
            remap = {int(v): i + offset for i, v in enumerate(vids)}
            fh.write("o shell_%05d\n" % gi)
            for v in vids:
                x, y, z = V[v] * scale
                fh.write("v %.6f %.6f %.6f\n" % (x, y, z))
            for tri in tris:
                fh.write("f %d %d %d\n"
                         % (remap[int(tri[0])], remap[int(tri[1])], remap[int(tri[2])]))
            offset += len(vids)

    return {
        "triangles": int(len(F)),
        "shells": int(len(unique)),
        "shells_dropped_small": int(dropped_small),
        "shells_kept": len(groups),
        "seconds": round(time.time() - t0, 2),
    }


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
def _signature(path: str) -> list:
    st = os.stat(path)
    return [int(st.st_size), int(st.st_mtime)]


def prepare(directory: str, mesh_rel_paths: dict[str, str], scale: float,
            log=print, min_extent: float = 40.0, max_shells: int = 80) -> dict[str, str]:
    """Convex-decompose every mesh that needs it, reusing cached results.

    ``mesh_rel_paths`` maps link name -> mesh path relative to ``directory``.
    Returns link name -> absolute path of the prepared collision OBJ.
    """
    settings = [round(float(min_extent), 4), int(max_shells), round(float(scale), 9)]
    cache_dir = os.path.join(directory, CACHE_DIRNAME).replace("\\", "/")
    os.makedirs(cache_dir, exist_ok=True)
    index_path = os.path.join(cache_dir, "index.json")
    index: dict = {}
    if os.path.isfile(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as fh:
                index = json.load(fh)
        except (json.JSONDecodeError, OSError):
            index = {}

    out: dict[str, str] = {}
    for link, rel in mesh_rel_paths.items():
        src = os.path.join(directory, rel).replace("\\", "/")
        if not os.path.isfile(src):
            log(f"  ! missing mesh for {link}: {rel}")
            continue
        dst = os.path.join(cache_dir, f"{link}.obj").replace("\\", "/")
        sig = _signature(src)
        cached = index.get(link)
        if (cached and cached.get("sig") == sig and cached.get("settings") == settings
                and os.path.isfile(dst)):
            log(f"  = {link:<22} cached ({cached['stats']['shells_kept']} shells)")
            out[link] = dst
            continue
        stats = decompose_to_obj(src, dst, scale, min_extent=min_extent,
                                 max_shells=max_shells)
        log("  + %-22s %d tris, %d shells -> %d kept (%.1fs)"
            % (link, stats["triangles"], stats["shells"], stats["shells_kept"],
               stats["seconds"]))
        index[link] = {"sig": sig, "settings": settings, "stats": stats}
        out[link] = dst

    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1)
    return out
