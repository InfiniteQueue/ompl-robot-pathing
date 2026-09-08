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
# How far past its own bounds a cell reaches when claiming triangles, as a fraction of the
# cell. It exists so that two neighbouring hulls are not decided apart by round-off where
# their faces are exactly coplanar, and so that a cell catching two near-collinear triangles
# has enough points to hull. Both are numerical concerns, worth thousandths of a cell.
#
# It is not what stops the surface being covered: a triangle lands in exactly one cell when
# this is 0, and that cell's hull is built from its vertices, so the triangle is inside it
# either way. Every increment here inflates every hull -- centroids assigned to one cell
# span ``1 + 2 * overlap`` cells before the triangles' own reach is added -- so it buys
# robustness in millimetres and costs claimed space in centimetres.
DEFAULT_OVERLAP = 0.02

SUBDIVISION_BUDGET = 800_000
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
                  overlap: float = DEFAULT_OVERLAP) -> tuple[np.ndarray, list[np.ndarray]]:
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
    ``overlap`` of a cell width.  That padding buys nothing measurable and is best left at
    zero.  It was justified here on two grounds, and both were tested and failed:

    * *That coverage would otherwise be lost to round-off between coplanar faces.*  It is
      not.  A triangle is assigned whole and its cell's hull is built from that triangle's
      own vertices, so the triangle is inside the hull by construction -- containment of
      the identical float coordinates, not a comparison that round-off can decide either
      way.  Sampling 60,000 points over the surface of an axis-aligned bracket, the worst
      case the claim describes, put every one of them inside some hull at overlap zero, at
      every cell size tried.
    * *That a cell catching two near-collinear triangles needs the extra points to hull
      with.*  The reverse: the pad is where degenerate pieces come from.  A cell that only
      exists because material reached into it holds just that sliver -- often a single
      coplanar face -- and hulls to a two-triangle sheet.  On the same bracket at a 60 mm
      cell, overlap 0 gave no degenerate pieces and overlap 0.02 gave twenty.

    What it does cost is certain.  The centroids assigned to one cell span
    ``1 + 2 * overlap`` cells before the triangles' own reach is added, so every hull
    claims that much more empty space.  Worse, a flat face lying on a cell boundary -- and
    ``origin`` puts one there by construction, on the shell's own minimum -- has all of its
    triangles copied into the neighbouring cell, which then hulls the same material a
    second time.  Parts are modelled axis-aligned with features on round coordinates, so
    this is the common case rather than the corner one: 27.6% of the 7495 hulls on one
    ST200 casting were duplicates of another, half-metre hulls among them, and tilting the
    test bracket off the axes dropped its duplicate share from 50% to 6%.
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
                     far_cell: float,
                     overlap: float = DEFAULT_OVERLAP
                     ) -> tuple[np.ndarray, list[np.ndarray], int]:
    """Grid-split a shell, finely near ``points`` and coarsely everywhere else.

    Returns the vertices, the pieces, and how many leading pieces came from the fine set.
    The near pieces are emitted first, so that one number splits the list; the caller wants
    it to report what share of the hulls the focus radius actually bought, which is the
    figure that says whether the radius is set anywhere useful.

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

    The two sets are disjoint: a triangle is refined finely or coarsely, never both.  They
    used to overlap by a cell, on the same reasoning that once padded the grid cells -- that
    a seam between the near and far hulls would be a hole.  It is not.  Whichever set a
    triangle lands in, that set's cell hull is built from its vertices and so contains it,
    so the surface is covered exactly once and covered completely.

    Overlapping them was expensive in the one place it could least be afforded.  Material
    just outside the radius was hulled twice: once on the fine grid, and again on the
    coarse one -- and that coarse copy sits directly over the weld, which is precisely
    where a hull claiming space it should not is worst.

    ``far_cell`` is the cell used beyond the radius, in the same units as ``cell`` and
    independent of it: how coarse the far side may be is a property of the part, not a
    ratio of how fine the near side has to be.  Deliberately a coarser grid rather than a
    single hull: one hull of everything far can bridge back *across* the near region,
    which is safe but can wall off an approach that was actually available.  0 asks for
    that single hull anyway.

    Classification happens before ``split_by_grid`` subdivides, so it is the original
    triangles that are sorted, and an oversized one is sorted by the distance from its
    nearest corner.  That is why ``near`` reaches a cell past the radius: a triangle up to
    a cell across whose far end is just outside the radius still has material inside it, and
    the cell it is cut into afterwards has to be the fine one.  ``V`` grows as a result and
    is returned with the pieces.
    """
    if points is None or len(points) == 0 or radius <= 0.0:
        # No focus, so neither bucket applies: this shell is refined uniformly and the
        # caller does not ask for a split it did not request.
        V, pieces = split_by_grid(V, tris, cell, overlap)
        return V, pieces, 0

    coarse = far_cell
    lo = V[np.unique(tris)].min(axis=0)
    hi = V[np.unique(tris)].max(axis=0)
    # Distance from each point to the shell's box: no triangle can be nearer than this, so
    # a shell every point is clear of skips the per-triangle measurement entirely.
    box = np.linalg.norm(np.maximum(np.maximum(lo - points, points - hi), 0.0), axis=1)
    if float(box.min()) > radius + cell:
        if far_cell > 0:
            V, pieces = split_by_grid(V, tris, coarse, overlap)
            return V, pieces, 0
        return V, [tris], 0

    corners = np.stack((V[tris[:, 0]], V[tris[:, 1]], V[tris[:, 2]]))
    t_lo, t_hi = corners.min(axis=0), corners.max(axis=0)
    nearest = _box_distances(t_lo, t_hi, points)
    pieces: list[np.ndarray] = []
    near = tris[nearest <= radius + cell]
    far = tris[nearest > radius + cell]
    near_pieces = 0
    if len(near):
        V, cut = split_by_grid(V, near, cell, overlap)
        pieces += cut
        near_pieces = len(cut)
    if len(far):
        if far_cell > 0:
            V, cut = split_by_grid(V, far, coarse, overlap)
            pieces += cut
        else:
            pieces.append(far)
    return V, pieces, near_pieces


def merge_far(V: np.ndarray, groups: list[tuple[np.ndarray, np.ndarray]],
              points: np.ndarray | None, radius: float, near_cell: float,
              merge_cell: float, overlap: float = DEFAULT_OVERLAP
              ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]], int]:
    """Wrap everything beyond the focus in one grid, across component boundaries.

    Splitting can only ever divide a shell.  Nothing in the pipeline before this could
    *combine* two, because refinement runs inside a loop over connected components and a
    component is what becomes an ``o`` group and therefore a hull.  So a link whose CAD
    arrives already decomposed pays one hull per solid however coarse the cell is set --
    on the sample gun body, 895 of 997 components are smaller than the cell and no value
    of ``hull_cell`` or ``far_cell`` touches them at all.

    This pools the far groups' triangles and re-cuts them on a single grid, so one hull
    covers whatever falls in its cell no matter which solid it came from.  That is the
    same convex bridging a hull already does within a shell -- the wrap that closes a
    recess -- extended to the gaps between shells.

    The error runs the safe way.  A hull over pooled material contains the union of the
    hulls it replaces, so merging only ever *adds* space: it can report a collision that
    is not there, never miss one that is.  How much it adds is bounded by the cell, since
    ``split_by_grid`` bisects oversized triangles down to it first.

    Groups within ``radius + near_cell`` of a focus point are left exactly as they were,
    so the geometry nearest the work is untouched.  With no focus given every group is
    pooled, which is the whole-link case.

    Returns the vertices, the new group list, and how many groups went into the pool.
    """
    if merge_cell <= 0.0 or not groups:
        return V, groups, 0

    focused = points is not None and len(points) and radius > 0.0
    if focused:
        # One call over every group's box rather than one per group: _box_distances is
        # chunked over the boxes it is given, and handing it them one at a time throws
        # that away for a thousand round trips.
        lo = np.array([V[vids].min(axis=0) for vids, _ in groups])
        hi = np.array([V[vids].max(axis=0) for vids, _ in groups])
        keep = _box_distances(lo, hi, points) <= radius + near_cell
    else:
        keep = np.zeros(len(groups), dtype=bool)

    near = [g for g, k in zip(groups, keep) if k]
    far = [tris for (_, tris), k in zip(groups, keep) if not k]
    if not far:
        return V, groups, 0

    V, pieces = split_by_grid(V, np.concatenate(far), merge_cell, overlap)
    return V, near + [(np.unique(p), p) for p in pieces], len(far)


def _write_groups(dst: str, V: np.ndarray, F: np.ndarray,
                  groups: list[tuple[np.ndarray, np.ndarray]], scale: float,
                  src: str) -> None:
    """Write each group as its own ``o`` block, which is one convex hull to Tesseract."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write("# weldpath convex-decomposition source=%s\n" % os.path.basename(src))
        fh.write("# %d triangles, %d groups, units=metres\n" % (len(F), len(groups)))
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


def decompose_to_obj(src: str, dst: str, scale: float, dedupe: bool = True,
                     min_extent: float = 40.0, max_shells: int = 80,
                     hull_cell: float = 0.0, fill_threshold: float = 0.75,
                     focus_points: np.ndarray | None = None,
                     focus_radius: float = 0.0, far_cell: float = 0.0,
                     overlap: float = DEFAULT_OVERLAP, merge_cell: float = 0.0,
                     enclosed_probe: float = 0.0, enclosed_voxel: float = 0.0,
                     enclosed_keep: float = 0.0,
                     enclosed_dump: str | None = None) -> dict:
    """Split ``src`` into connected shells and write them as ``o`` groups into ``dst``.

    Every shell becomes a convex hull, and each hull is a collision pair to test, so the
    raw shell count sets the cost of a collision check.  Two filters keep that in hand:
    shells whose bounding-box diagonal is under ``min_extent`` are dropped (fasteners and
    small internal parts, which sit inside the envelope of the bodies around them), and
    only the ``max_shells`` largest survive.  On the sample cell this takes the gun from
    948 shells to 80 and a collision check from 2.8 ms to 1.6 ms, which is the difference
    between the sampling planner timing out and solving in a few seconds.

    A third filter joins them when ``enclosed_probe`` is set: shells nothing of that radius
    can reach from outside the link are dropped, which on assembled CAD is most of what the
    file contains.  See :mod:`weldpath.enclosed` -- in particular for why this one can lose
    a real collision where the other two cannot, and what ``enclosed_keep`` does about it.
    """
    t0 = time.time()
    V, F = load_obj(src)
    weld = weld_index(V)
    labels = connected_shells(V, F)
    unique = np.unique(labels)

    # Before ``min_extent``, and so before the cap: what cannot be touched is not geometry
    # this cell has any use for, and letting it compete for a place under ``max_shells``
    # would mean the cap spent on parts nothing can reach.
    sealed: set = set()
    enc_stats: dict = {}
    if enclosed_probe > 0.0:
        from . import enclosed as _enclosed
        sealed, enc_stats = _enclosed.sealed_labels(
            V, F, labels, probe=enclosed_probe,
            voxel=enclosed_voxel if enclosed_voxel > 0.0 else enclosed_probe,
            keep_extent=enclosed_keep)
        if enclosed_dump and sealed:
            # What was discarded, written beside the kept geometry so a change of probe can
            # be looked at rather than believed.  This filter removes real material, which
            # none of the others do, so it is the one that has to be inspectable.
            _write_groups(enclosed_dump, V, F,
                          [(np.unique(F[labels == lab]), F[labels == lab])
                           for lab in sorted(sealed)], scale, src)

    candidates: list[tuple[float, np.ndarray, np.ndarray, tuple]] = []
    for lab in unique:
        if int(lab) in sealed:
            continue
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
    # Counted only where a focus is actually in play.  Without one every hull is refined
    # the same way, so "near" and "far" are not two things and reporting a share of them
    # would invent a distinction the run never made.
    focused = (focus_points is not None and len(focus_points) and focus_radius > 0.0)
    near_hulls = far_hulls = 0
    if hull_cell > 0:
        refined: list[tuple[np.ndarray, np.ndarray]] = []
        for vids, tris in groups:
            P = V[vids]
            diagonal = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))
            if diagonal <= hull_cell or hull_fill(V, tris, weld) >= fill_threshold:
                refined.append((vids, tris))
                continue
            V, pieces, near = split_near_focus(V, tris, hull_cell, focus_points,
                                               focus_radius, far_cell, overlap)
            if len(pieces) < 2:
                refined.append((vids, tris))
                continue
            split_shells += 1
            if focused:
                near_hulls += near
                far_hulls += len(pieces) - near
            refined.extend((np.unique(piece), piece) for piece in pieces)
        groups = refined

    # Last, because it works on whatever the rest produced.  Splitting can only divide, so
    # every filter and refinement above leaves the count at one hull per surviving solid
    # at best; this is the only step that can bring two together.
    before = len(groups)
    V, groups, pooled = merge_far(V, groups, focus_points, focus_radius, hull_cell,
                                  merge_cell, overlap)

    _write_groups(dst, V, F, groups, scale, src)

    return {
        "triangles": int(len(F)),
        "shells": int(len(unique)),
        "shells_dropped_small": int(dropped_small),
        "shells_kept": len(groups),
        "shells_refined": int(split_shells),
        # Hulls the splitting produced, sorted by which side of the focus radius their
        # material fell.  Both 0 where nothing was split or no focus was given, and their
        # sum is below ``shells_kept``: a shell small enough or solid enough to pass the
        # refinement gate is one hull that was never sorted either way.
        "shells_near_focus": int(near_hulls),
        "shells_far_focus": int(far_hulls),
        # What the merge pooled and what it left.  Both absent where no merge cell was
        # given, so a cache entry can tell "did not merge" from "merged nothing".
        **({"merged_from": int(pooled),
            "merged_to": int(len(groups) - (before - pooled))} if pooled else {}),
        # Absent entirely when no probe was given, so a cache entry can be told apart from
        # one where the filter ran and found nothing to drop.
        **({"enclosed": enc_stats} if enc_stats else {}),
        "seconds": round(time.time() - t0, 2),
    }


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
def _signature(path: str) -> list:
    st = os.stat(path)
    return [int(st.st_size), int(st.st_mtime)]


DEFAULT_FILL = 0.75


def _focus_share(stats: dict) -> str:
    """What share of the split hulls landed inside the focus radius, as a phrase.

    Empty where the question does not arise -- nothing was split, or no focus was given --
    rather than "0%", which would read as a focus that caught nothing when in fact none
    was asked for.  Cached entries written before this was recorded also come back empty,
    which is why the summary counts how many links could not answer.
    """
    near = int(stats.get("shells_near_focus", 0))
    far = int(stats.get("shells_far_focus", 0))
    if near + far <= 0:
        return ""
    # The count goes in front of the share deliberately.  Without it the percentage reads
    # as a share of the kept total, and on a link whose CAD arrives already decomposed it
    # is nothing of the sort -- 32 hulls out of 1021 on one gun body, so the figure
    # describes 3% of the geometry and moving it moves almost nothing.  The denominator
    # has to be visible next to the number or the number argues for the wrong lever.
    return (f", {near + far} of them from splitting, "
            f"{100.0 * near / (near + far):.0f}% of those near the focus")


def _enclosed_share(stats: dict) -> str:
    """What the enclosure filter discarded on one link, as a phrase.

    Empty when no probe was given.  When one was and it dropped nothing the phrase still
    appears, saying so: a filter that ran and found nothing is a different fact from a
    filter that was switched off, and only one of them is a reason to widen the probe.
    """
    enc = stats.get("enclosed")
    if not enc:
        return ""
    total = int(stats.get("shells", 0)) or 1
    out = (f", {enc['sealed']} sealed inside ({100.0 * enc['sealed'] / total:.0f}% of "
           f"shells, {100.0 * enc['sealed_triangles'] / max(int(stats.get('triangles', 1)), 1):.0f}% "
           f"of triangles) dropped as unreachable by a {enc['probe']:g} mm probe")
    if enc.get("kept_large"):
        # Named on the line rather than left to the stats file.  These are components the
        # flood called sealed and the size backstop overruled, which is the one number that
        # says the two disagree -- and disagreement here is the signal that the probe or
        # the voxel is wrong, not a detail.
        out += (f" ({enc['kept_large']} spared by the size backstop)")
    if enc.get("coarsened"):
        out += f" [screened at {enc['voxel']:.0f} mm, coarsened to fit memory]"
    return out


def _merge_share(stats: dict) -> str:
    """What the merge pooled on one link, as a phrase.

    Reports the two counts rather than a ratio.  The ratio is the striking number and the
    one that misleads: a link is not "96% smaller", it has traded a count set by how the
    CAD was assembled for one set by the cell, and those are not the same quantity
    measured twice.
    """
    got = int(stats.get("merged_from", 0))
    if not got:
        return ""
    return f", {got} of them merged into {int(stats.get('merged_to', 0))}"


def prepare(directory: str, mesh_rel_paths: dict[str, str], scale: float,
            log=print, min_extent: float = 40.0, max_shells: int = 80,
            hull_cell: float = 0.0, fill: float = DEFAULT_FILL,
            cells: dict[str, float] | None = None,
            focus: dict[str, tuple[np.ndarray, float]] | None = None,
            far_cells: dict[str, float] | None = None, far_cell: float = 0.0,
            overlap: float = DEFAULT_OVERLAP,
            merge_cells: dict[str, float] | None = None, merge_cell: float = 0.0,
            enclosed_probes: dict[str, float] | None = None,
            enclosed_voxel: float = 0.0, enclosed_keep: float = 0.0,
            enclosed_dump: bool = False) -> dict[str, str]:
    """Convex-decompose every mesh that needs it, reusing cached results.

    ``mesh_rel_paths`` maps link name -> mesh path relative to ``directory``.
    Returns link name -> absolute path of the prepared collision OBJ.

    ``focus`` maps a link to the points its refinement should concentrate around and the
    radius to hold to -- the welds for a panel, the tool centre point for the gun.  The
    points must be in the same frame and units as that link's source mesh, which is the
    caller's business to establish; a link with no entry refines everywhere.

    ``far_cells`` maps a link to the cell size to use beyond that radius, falling back to
    ``far_cell``.  It is an absolute size rather than a multiple of the link's own cell:
    the two answer different questions, and tightening the near side is not a reason for
    the far side to follow it down.

    ``enclosed_probes`` maps a link to the radius of the smallest thing that could reach
    into it; shells nothing that size can touch from outside are dropped before anything
    else looks at them.  Per link and off by default, because it is per link that the
    question makes sense -- a gun body is full of motors and a panel is not -- and because
    it is the one filter here that can remove geometry a collision needed.

    ``merge_cells`` maps a link to the cell its far geometry is pooled and re-cut on, so
    one hull covers everything in a cell whatever solid it came from.  It is the only step
    that reduces a hull count set by how the CAD was assembled rather than by how the part
    is shaped -- see :func:`merge_far`.
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
        far = float((far_cells or {}).get(link, far_cell))
        probe = float((enclosed_probes or {}).get(link, 0.0))
        merge = float((merge_cells or {}).get(link, merge_cell))
        threshold = float(fill)
        points, radius = focus.get(link, (None, 0.0))
        settings = [round(float(min_extent), 4), int(max_shells), round(float(scale), 9),
                    round(cell, 4), round(threshold, 4)]
        if cell > 0.0 and round(float(overlap), 4) != 0.25:
            # The cell padding used to be fixed at 0.25 and was not part of the key. Naming
            # it only when it differs keeps every cache built before it was tunable valid.
            settings += ["ovl", round(float(overlap), 4)]
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
            settings += [round(radius, 4),
                         # "abs" marks the far cell as a size rather than the multiple of
                         # the near cell it used to be, so a cache holding a factor of 6
                         # is not mistaken for one holding a 6 mm cell.
                         "abs", round(far, 4), digest,
                         # The near and far sets no longer overlap, so geometry cached while
                         # they did holds a coarse duplicate of the refined material.
                         "disjoint"]
        if merge > 0.0:
            # Appended only where merging is on, so caches built before it stay valid.
            # The near cell and the radius are already in the key above, and they decide
            # which groups are pooled, so they need no second mention here.
            settings += ["merge", round(merge, 4)]
        if probe > 0.0:
            # Appended only where the filter is on, so every cache built before it existed
            # stays valid.  The voxel is in the key as well as the probe: it is a screening
            # resolution, not a tolerance, and a coarser one genuinely keeps different
            # shells rather than the same ones measured more loosely.
            settings += ["sealed", round(probe, 4), round(float(enclosed_voxel), 4),
                         round(float(enclosed_keep), 4)]
        src = os.path.join(directory, rel).replace("\\", "/")
        if not os.path.isfile(src):
            log(f"  ! missing mesh for {link}: {rel}")
            continue
        dst = os.path.join(cache_dir, f"{link}.obj").replace("\\", "/")
        # Beside the kept geometry, in the cache the run owns -- never in the study's own
        # mesh directory, which is the customer's input.
        dump = os.path.join(cache_dir, f"{link}.sealed.obj").replace("\\", "/")
        sig = _signature(src)
        cached = index.get(link)
        if (cached and cached.get("sig") == sig and cached.get("settings") == settings
                and os.path.isfile(dst)):
            log(f"  = {link:<22} cached ({cached['stats']['shells_kept']} shells"
                f"{_focus_share(cached['stats'])}"
                f"{_enclosed_share(cached['stats'])}"
                f"{_merge_share(cached['stats'])})")
            out[link] = dst
            continue
        log(f"  . {link:<22} decomposing {os.path.getsize(src) / 1e6:.1f} MB of "
            f"mesh")
        stats = decompose_to_obj(src, dst, scale, min_extent=min_extent,
                                 max_shells=max_shells, hull_cell=cell,
                                 fill_threshold=threshold, focus_points=points,
                                 focus_radius=radius, far_cell=far,
                                 overlap=overlap, merge_cell=merge,
                                 enclosed_probe=probe,
                                 enclosed_voxel=enclosed_voxel,
                                 enclosed_keep=enclosed_keep,
                                 enclosed_dump=(dump if enclosed_dump and probe > 0.0
                                                else None))
        near = ((f" within {radius:g} mm of a focus point, "
                 + (f"{far:g} mm beyond it" if far > 0.0 else "one hull beyond it"))
                if radius > 0.0 else "")
        refined = (f", {stats['shells_refined']} split at {cell:g} mm below "
                   f"{threshold:g} fill{near}" if stats["shells_refined"] else "")
        log("  + %-22s %d tris, %d shells -> %d kept%s%s%s%s (%.1fs)"
            % (link, stats["triangles"], stats["shells"], stats["shells_kept"],
               _enclosed_share(stats), refined, _focus_share(stats),
               _merge_share(stats), stats["seconds"]))
        index[link] = {"sig": sig, "settings": settings, "stats": stats}
        out[link] = dst

    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1)
    entries = [index[l]["stats"] for l in out if l in index]
    shells = sum(int(s["shells_kept"]) for s in entries)
    near = sum(int(s.get("shells_near_focus", 0)) for s in entries)
    far = sum(int(s.get("shells_far_focus", 0)) for s in entries)
    # A link that was split under a focus but reports neither count was cached before the
    # counts existed.  Named rather than folded in, because it is missing from the ratio's
    # denominator as well as its numerator and would drag the figure either way.
    stale = sum(1 for s in entries if s.get("shells_refined")
                and s.get("shells_near_focus") is None and focus)
    share = ""
    if near + far > 0:
        share = (f", {near + far} of them from splitting a focused shell and "
                 f"{100.0 * near / (near + far):.0f}% of those near a focus point"
                 + (f" (excluding {stale} link{'' if stale == 1 else 's'} cached before "
                    f"this was recorded)" if stale else ""))
    elif stale:
        share = (f", none of which report where they fell relative to the focus: all "
                 f"{stale} split link{'' if stale == 1 else 's'} were cached before this "
                 f"was recorded")
    log(f"  {shells} convex shells over {len(out)} links{share}; every collision check "
        f"from here on is against these")
    return out
