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

import hashlib
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


def weld_index(V: np.ndarray, weld_tol: float = 1e-6) -> np.ndarray:
    """Map each vertex to an id shared by every vertex at the same rounded position.

    CAD tessellation routinely duplicates vertices along shared edges, so raw indices say
    nothing about which triangles actually meet.  Every topological question here -- what
    is one shell, what is closed -- has to be asked of the welded indices instead.
    """
    key = np.round(V / weld_tol).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    return inv.reshape(-1)


def connected_shells(V: np.ndarray, F: np.ndarray, weld_tol: float = 1e-6) -> np.ndarray:
    """Label each triangle with the id of the connected shell it belongs to."""
    if len(F) == 0:
        return np.zeros(0, dtype=np.int64)
    inv = weld_index(V, weld_tol)
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


def is_watertight(tris: np.ndarray) -> bool:
    """True when every edge is shared by exactly two triangles."""
    if len(tris) == 0:
        return False
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.sort(e, axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def mesh_volume(V: np.ndarray, tris: np.ndarray) -> float:
    """Enclosed volume by the divergence theorem.

    Only meaningful for a closed surface: on an open one the sum is origin-dependent, and
    since these meshes carry world coordinates metres from the origin it comes out
    arbitrarily large.  Callers must check ``is_watertight`` first.
    """
    if len(tris) == 0:
        return 0.0
    P = V - V[np.unique(tris)].mean(axis=0)
    a, b, c = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
    return float(abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


def hull_fill(V: np.ndarray, tris: np.ndarray, weld: np.ndarray | None = None) -> float:
    """How much of its own bounding box a shell fills, as a proxy for hull tightness.

    A machined block fills nearly all of it and hulls faithfully.  A C-yoke or a casting
    with a deep recess fills little, and its hull bridges the void -- which is what makes
    a hulled ``robot_base`` read 62.6 mm of penetration where the real part grazes by 7.1.

    Open shells get 0.0, i.e. "assume it needs refining": a surface encloses no volume, so
    there is nothing to compare, and an unclosed sheet is exactly the case where a hull
    over-claims the space behind it.

    Watertightness has to be judged on welded indices.  Asked of the raw ones it comes back
    false for *every* shell in the sample cell -- 0 of 6 panel shells and 0 of the first 200
    fixture shells, against 3 and 185 once welded -- which silently collapses the fill
    threshold into "always refine".
    """
    P = V[np.unique(tris)]
    box = float(np.prod(P.max(axis=0) - P.min(axis=0)))
    if box <= 0:
        return 1.0                          # planar: a hull is already exact
    if weld is None:
        weld = weld_index(V)
    if not is_watertight(weld[tris]):
        return 0.0
    return mesh_volume(V, tris) / box


# A shell is cut into cells whose hulls are meant to track the surface to about the cell
# size, but a triangle is indivisible once it is sorted into a cell, so a shell tessellated
# more coarsely than the grid cannot honour that -- a flat fixture face exported as two
# triangles metres across yields a metres-wide hull whatever the cell is set to.
# ``subdivide_to`` removes the floor by bisecting oversized triangles first.
SUBDIVISION_BUDGET = 400_000
SUBDIVISION_PASSES = 64


def subdivide_to(V: np.ndarray, tris: np.ndarray, target: float,
                 budget: int = SUBDIVISION_BUDGET) -> tuple[np.ndarray, np.ndarray]:
    """Bisect triangles until no edge is longer than ``target``.  Returns ``(V, tris)``.

    Longest-edge bisection: the longest edge is halved and the triangle replaced by the
    two halves.  This is exact -- the new vertex is the midpoint of an existing edge, so it
    lies on the original surface and the subdivided mesh occupies precisely the same space.
    Hulls built from it can therefore only get tighter, never inflate.

    Halving one edge rather than all three costs two triangles per pass instead of four and
    leaves the long thin triangles CAD produces looking less like slivers, at the price of
    hanging nodes where a split triangle meets an unsplit neighbour.  Those do not matter
    here: the pieces are only ever fed to a hull builder, which reads points and cares
    nothing for topology, and the watertightness tests all run earlier on the whole shell.

    ``budget`` caps the triangles one shell may reach.  Hit it and subdivision stops where
    it stands, which degrades to a coarser result rather than to a wrong one.
    """
    V = np.asarray(V, dtype=float)
    tris = np.asarray(tris, dtype=np.int64)
    if target <= 0.0 or not len(tris):
        return V, tris

    for _ in range(SUBDIVISION_PASSES):
        edges = np.stack([np.linalg.norm(V[tris[:, 1]] - V[tris[:, 0]], axis=1),
                          np.linalg.norm(V[tris[:, 2]] - V[tris[:, 1]], axis=1),
                          np.linalg.norm(V[tris[:, 0]] - V[tris[:, 2]], axis=1)], axis=1)
        longest = edges.argmax(axis=1)
        big = edges[np.arange(len(tris)), longest] > target
        count = int(big.sum())
        if not count or len(tris) + count > budget:
            break
        # Rotate each triangle so its longest edge is (0, 1); the third vertex is opposite.
        order = (longest[big, None] + np.arange(3)[None, :]) % 3
        r = np.take_along_axis(tris[big], order, axis=1)
        # One new vertex per distinct edge, so neighbours that split the same edge share it.
        key = np.stack([np.minimum(r[:, 0], r[:, 1]),
                        np.maximum(r[:, 0], r[:, 1])], axis=1)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        mid = len(V) + inv.reshape(-1)
        V = np.concatenate([V, 0.5 * (V[uniq[:, 0]] + V[uniq[:, 1]])])
        tris = np.concatenate([
            tris[~big],
            np.stack([r[:, 0], mid, r[:, 2]], axis=1),
            np.stack([mid, r[:, 1], r[:, 2]], axis=1)])
    return V, tris


def split_by_grid(V: np.ndarray, tris: np.ndarray, cell: float,
                  overlap: float = 0.25) -> tuple[np.ndarray, list[np.ndarray]]:
    """Cut a shell into a grid of cells, each of which is hulled separately.

    This is the whole reason a hull can be made more precise without a full convex
    decomposition: a cell-sized piece of a curved or recessed part is nearly convex, so
    the union of the cell hulls tracks the real surface to roughly the cell size instead
    of bridging the part end to end.

    Triangles larger than a cell are bisected down to it first, so that claim holds however
    coarsely the part was tessellated -- without it the grid can only ever be as fine as
    the triangles it was handed.  A triangle whose edges are all within the cell reaches at
    most two thirds of a cell from its centroid, which keeps each bucket's hull inside its
    own cell to within the accuracy the cell already promises.  ``V`` therefore grows, and
    is returned alongside the pieces; existing indices keep their meaning.

    Triangles are assigned by centroid, and also to any neighbouring cell within
    ``overlap`` of a cell width.  Without that padding the hulls would meet edge to edge
    with nothing spanning the seam, and a thin gap between two collision shapes is a hole
    a planner will happily drive the gun through.
    """
    V, tris = subdivide_to(V, tris, cell)
    P = V[np.unique(tris)]
    origin = P.min(axis=0)
    pad = overlap * cell
    a, b, c = V[tris[:, 0]], V[tris[:, 1]], V[tris[:, 2]]
    centroid = (a + b + c) / 3.0

    buckets: dict[tuple, list[int]] = {}
    lo = np.floor((centroid - pad - origin) / cell).astype(np.int64)
    hi = np.floor((centroid + pad - origin) / cell).astype(np.int64)
    for t in range(len(tris)):
        for ix in range(lo[t, 0], hi[t, 0] + 1):
            for iy in range(lo[t, 1], hi[t, 1] + 1):
                for iz in range(lo[t, 2], hi[t, 2] + 1):
                    buckets.setdefault((ix, iy, iz), []).append(t)
    return V, [tris[np.array(ids, dtype=np.int64)] for ids in buckets.values() if ids]


def _box_distances(lo: np.ndarray, hi: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Distance from each box to the nearest of ``points``; 0 where a point is inside.

    Chunked over the boxes and accumulated one axis at a time, so a 2M-triangle fixture
    against a few hundred focus points never builds a (triangles x points x 3) array.
    """
    out = np.empty(len(lo), dtype=float)
    for i in range(0, len(lo), 2048):
        a, b = lo[i:i + 2048], hi[i:i + 2048]
        d2 = np.zeros((len(a), len(points)))
        for k in range(3):
            gap = np.maximum(np.maximum(a[:, k, None] - points[None, :, k],
                                        points[None, :, k] - b[:, k, None]), 0.0)
            d2 += gap * gap
        out[i:i + 2048] = np.sqrt(d2.min(axis=1))
    return out


def split_near_focus(V: np.ndarray, tris: np.ndarray, cell: float,
                     points: np.ndarray | None, radius: float,
                     far_factor: float) -> tuple[np.ndarray, list[np.ndarray]]:
    """Grid-split a shell, finely near ``points`` and coarsely everywhere else.

    Refinement is only ever worth its cost where two pieces of geometry actually come
    close, and that is a local business: a few hundred millimetres around each weld on a
    panel, and the last stretch before the electrodes on the gun -- not the whole part.
    Since refinement runs after ``max_shells``, spending it on the far side of a fixture
    is what takes a 1136-shell part to 45222 and slows every later collision check.

    Distance is measured to each triangle's *bounding box*, not to its centroid.  CAD
    tessellation makes triangles of wildly different sizes -- a flat face on a fixture can
    be two triangles metres across -- and a centroid says nothing about where such a
    triangle reaches: its middle can be a metre clear of the focus while a corner all but
    touches it.  The box distance is a lower bound on the true one, so the error only ever
    runs towards refining something that did not need it.

    The band one cell wide either side of ``radius`` goes into *both* sets.  Without that
    the near and far hulls would meet edge to edge with nothing spanning the seam, which is
    a hole a planner will happily drive the gun through -- the same reason
    ``split_by_grid`` pads its cells.

    ``far_factor`` scales the cell used beyond the radius.  Deliberately a coarser grid
    rather than a single hull: one hull of everything far can bridge back *across* the
    near region, which is safe but can wall off an approach that was actually available.

    Classification happens before ``split_by_grid`` subdivides, so it is the original
    triangles that are sorted: an oversized one straddling the boundary lands in both sets
    and is then cut down within each.  ``V`` grows as a result and is returned with the
    pieces.
    """
    if points is None or len(points) == 0 or radius <= 0.0:
        return split_by_grid(V, tris, cell)

    coarse = cell * far_factor
    lo = V[np.unique(tris)].min(axis=0)
    hi = V[np.unique(tris)].max(axis=0)
    # Distance from each point to the shell's box: no triangle can be nearer than this, so
    # a shell every point is clear of skips the per-triangle measurement entirely.
    box = np.linalg.norm(np.maximum(np.maximum(lo - points, points - hi), 0.0), axis=1)
    if float(box.min()) > radius + cell:
        return split_by_grid(V, tris, coarse) if far_factor > 0 else (V, [tris])

    corners = np.stack((V[tris[:, 0]], V[tris[:, 1]], V[tris[:, 2]]))
    t_lo, t_hi = corners.min(axis=0), corners.max(axis=0)
    nearest = _box_distances(t_lo, t_hi, points)
    pieces: list[np.ndarray] = []
    near = tris[nearest <= radius + cell]
    far = tris[nearest > radius]
    if len(near):
        V, cut = split_by_grid(V, near, cell)
        pieces += cut
    if len(far):
        if far_factor > 0:
            V, cut = split_by_grid(V, far, coarse)
            pieces += cut
        else:
            pieces.append(far)
    return V, pieces


def decompose_to_obj(src: str, dst: str, scale: float, dedupe: bool = True,
                     min_extent: float = 40.0, max_shells: int = 80,
                     hull_cell: float = 0.0, fill_threshold: float = 0.75,
                     focus_points: np.ndarray | None = None,
                     focus_radius: float = 0.0, far_factor: float = 4.0) -> dict:
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
    weld = weld_index(V)
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

    # Refine the shells a single hull represents badly.  Done after the cap, so the cap
    # still means "this many parts" and refinement is priced separately.
    split_shells = 0
    if hull_cell > 0:
        refined: list[tuple[np.ndarray, np.ndarray]] = []
        for vids, tris in groups:
            P = V[vids]
            diagonal = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))
            if diagonal <= hull_cell or hull_fill(V, tris, weld) >= fill_threshold:
                refined.append((vids, tris))
                continue
            V, pieces = split_near_focus(V, tris, hull_cell, focus_points,
                                         focus_radius, far_factor)
            if len(pieces) < 2:
                refined.append((vids, tris))
                continue
            split_shells += 1
            refined.extend((np.unique(piece), piece) for piece in pieces)
        groups = refined

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
        "shells_refined": int(split_shells),
        "seconds": round(time.time() - t0, 2),
    }


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
def _signature(path: str) -> list:
    st = os.stat(path)
    return [int(st.st_size), int(st.st_mtime)]


DEFAULT_FILL = 0.75


def prepare(directory: str, mesh_rel_paths: dict[str, str], scale: float,
            log=print, min_extent: float = 40.0, max_shells: int = 80,
            hull_cell: float = 0.0, fill: float = DEFAULT_FILL,
            cells: dict[str, float] | None = None,
            focus: dict[str, tuple[np.ndarray, float]] | None = None,
            far_factor: float = 4.0) -> dict[str, str]:
    """Convex-decompose every mesh that needs it, reusing cached results.

    ``mesh_rel_paths`` maps link name -> mesh path relative to ``directory``.
    Returns link name -> absolute path of the prepared collision OBJ.

    ``focus`` maps a link to the points its refinement should concentrate around and the
    radius to hold to -- the welds for a panel, the tool centre point for the gun.  The
    points must be in the same frame and units as that link's source mesh, which is the
    caller's business to establish; a link with no entry refines everywhere.
    """
    focus = {k: (np.asarray(pts, dtype=float).reshape(-1, 3), float(r))
             for k, (pts, r) in (focus or {}).items() if pts is not None and len(pts)}
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
        cell = float((cells or {}).get(link, hull_cell))
        threshold = float(fill)
        points, radius = focus.get(link, (None, 0.0))
        settings = [round(float(min_extent), 4), int(max_shells), round(float(scale), 9),
                    round(cell, 4), round(threshold, 4)]
        if cell > 0.0:
            # Subdividing oversized triangles changed what a cell produces, so geometry
            # cached before it must not be reused.  Only appended where a cell is in play,
            # which leaves unrefined links reusing their existing cache entries.
            settings += ["subdiv1"]
        if radius > 0.0:
            # Geometry refined around one set of points must not be reused for another, so
            # the points are part of the key.  Appended rather than always present, so a
            # cache prepared before this option existed stays valid while it is left off.
            digest = hashlib.sha1(
                np.round(np.sort(points, axis=0), 3).tobytes()).hexdigest()[:12]
            settings += [round(radius, 4), round(float(far_factor), 4), digest]
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
        log(f"  . {link:<22} decomposing {os.path.getsize(src) / 1e6:.1f} MB of "
            f"mesh")
        stats = decompose_to_obj(src, dst, scale, min_extent=min_extent,
                                 max_shells=max_shells, hull_cell=cell,
                                 fill_threshold=threshold, focus_points=points,
                                 focus_radius=radius, far_factor=far_factor)
        near = f" within {radius:g} mm of a focus point" if radius > 0.0 else ""
        refined = (f", {stats['shells_refined']} split at {cell:g} mm below "
                   f"{threshold:g} fill{near}" if stats["shells_refined"] else "")
        log("  + %-22s %d tris, %d shells -> %d kept%s (%.1fs)"
            % (link, stats["triangles"], stats["shells"], stats["shells_kept"],
               refined, stats["seconds"]))
        index[link] = {"sig": sig, "settings": settings, "stats": stats}
        out[link] = dst

    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1)
    shells = sum(int(index[l]["stats"]["shells_kept"]) for l in out if l in index)
    log(f"  {shells} convex shells over {len(out)} links; every collision check from here "
        f"on is against these")
    return out
