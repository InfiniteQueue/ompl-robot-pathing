"""Find the parts of a mesh nothing outside it can ever touch.

CAD of a welding gun or a fixture arrives as one file holding every solid the assembly
contains, and most of those solids are *internal*: motors, cabling, brackets, fasteners,
sealed inside a body that the robot only ever touches from the outside.  Each one still
becomes a convex hull, and each hull is a pair to test on every collision check, so on a
real gun body 314 of 997 components -- a third of them -- are geometry that can never
decide anything.

The test used here is reachability, not enclosure.  Rasterise the whole link to a voxel
grid, grow the solid by the radius of the smallest thing that could reach it, flood the
remaining air inward from beyond the bounding box, and ask of each connected component
whether any of its surface borders air the flood arrived at.  What that buys over the
obvious alternatives:

* **Booleans and volume tests** need watertight, non-self-intersecting input, which
  exported CAD reliably is not.  Nothing here consults topology at all; the grid does not
  care whether a solid is closed, doubled, or inside out.
* **Ray casting** answers "is there a straight line out", which is the wrong question --
  air bends around corners and a probe follows it.  A flood finds the bending path.
* **A depth or distance threshold** would drop a deep recess in the outer skin along with
  the internals.  Reachability keeps anything the flood can crawl into, however deep.

The grid is a **classifier and nothing else**.  It decides which components survive; it
never becomes collision geometry.  Every component that survives is passed on with its
original triangles, so the outer surface the robot actually plans against is unchanged to
the last vertex.

The direction of error is worth being explicit about, because it is the opposite of the
usual one.  Dropping a component *removes* real material, so a component wrongly called
sealed is a collision that will not be reported.  Two things hold that in check, and
neither is a substitute for looking at the output once:

* The probe dilation uses 6-connectivity, whose repeated application grows a diamond
  rather than a ball.  A diamond is contained in the ball of the same radius, so the
  solid is grown by *less* than asked, air is left *more* connected than it truly is, and
  the flood reaches further than the real probe would.  Every rounding here keeps
  components alive.
* ``keep_extent`` refuses to drop anything above a given size whatever the flood says,
  and the count is reported rather than silently applied.
"""
from __future__ import annotations

import numpy as np

# Empty margin around the mesh, in cells.  Must exceed any dilation applied afterwards:
# if the grown solid reaches the grid boundary the flood has nowhere to seed and the whole
# link reads as enclosed.  That failure is silent and total, so the margin is generous.
PAD_CELLS = 10

# Ceiling on the grid, in cells.  A bool grid is one byte per cell, and the sweeps below
# allocate a handful of temporaries of the same shape, so this is roughly ten times its
# own size in peak memory.  Beyond it the voxel is coarsened rather than the machine being
# asked for the array, since a coarser screen is still a screen and an allocation failure
# in the middle of preparation is not.
MAX_CELLS = 48_000_000


def dilate(mask: np.ndarray, n: int = 1) -> np.ndarray:
    """Grow a boolean grid by ``n`` cells in 6-connectivity."""
    for _ in range(int(n)):
        out = mask.copy()
        out[1:] |= mask[:-1]
        out[:-1] |= mask[1:]
        out[:, 1:] |= mask[:, :-1]
        out[:, :-1] |= mask[:, 1:]
        out[:, :, 1:] |= mask[:, :, :-1]
        out[:, :, :-1] |= mask[:, :, 1:]
        mask = out
    return mask


def _sweep(reach: np.ndarray, seg: np.ndarray, axis: int, reverse: bool) -> np.ndarray:
    """Propagate ``reach`` as far as it will go along one direction of one axis.

    ``seg`` numbers the maximal runs of free cells along ``axis`` -- a running count of
    blocked cells, so every free cell in one run shares a value and the value only ever
    increases along the axis.  Marking reached cells with their own run number and taking a
    running maximum then gives, at each cell, the highest run number reached at or before
    it; that equals the cell's own run exactly when something in the same run was reached.

    Doing it this way rather than by repeated single-cell dilation is what makes this
    usable on a fixture.  A dilation step moves the front one cell, so a flood down a long
    passage costs an iteration per cell of its length; a sweep crosses the whole passage at
    once, and the number of sweeps needed follows how many times the air has to turn a
    corner, not how far it runs.
    """
    if reverse:
        # Backwards the run numbers descend, so the running *minimum* of the reached ones
        # is the same argument with the order turned round: every later cell has a run
        # number at least this cell's, so a minimum equal to it means one of them shares it.
        m = np.flip(np.where(reach, seg, np.iinfo(seg.dtype).max), axis=axis)
        got = np.flip(np.minimum.accumulate(m, axis=axis), axis=axis)
    else:
        got = np.maximum.accumulate(np.where(reach, seg, -1), axis=axis)
    return got == seg


def flood_outside(free: np.ndarray) -> np.ndarray:
    """The cells of ``free`` connected to the outside of the grid.

    Seeded from the six faces rather than from a chosen point: the grid is padded well
    clear of the mesh, so its boundary is entirely outside air, and seeding all of it
    removes any question of which side a chosen point fell.
    """
    reach = np.zeros_like(free)
    reach[0], reach[-1] = free[0], free[-1]
    reach[:, 0], reach[:, -1] = free[:, 0], free[:, -1]
    reach[:, :, 0], reach[:, :, -1] = free[:, :, 0], free[:, :, -1]
    blocked = ~free
    segs = [np.cumsum(blocked, axis=a, dtype=np.int32) for a in range(3)]
    n = -1
    while True:
        for axis in range(3):
            for reverse in (False, True):
                # Masked here rather than once at the end of the round.  A blocked cell
                # carries the run number of the free cells *after* it, so a backward sweep
                # marks it, and left in place it would carry the flood through the wall on
                # the next axis -- air on both sides of a partition joining through it.
                reach |= _sweep(reach, segs[axis], axis, reverse) & free
        got = int(reach.sum())
        if got == n:
            return reach
        n = got


def _rasterise(V: np.ndarray, F: np.ndarray, labels: np.ndarray, cell: float,
               lo: np.ndarray, dims: np.ndarray):
    """Occupancy grid, plus the flat cell indices each component occupies.

    Triangles are point-sampled rather than exactly overlap-tested against each cell.  At a
    sample spacing below half a cell a triangle cannot step over one, and this is a screen,
    not the collision geometry -- an occasional extra cell costs nothing but a component
    kept.  Sampling rates are grouped so the work stays vectorised: one fixed rate would
    either miss a metre-wide fixture face or drown in samples for a millimetre-wide boss.

    Cells are recorded per component rather than one owner per cell.  Neighbouring
    components share cells, and a single owner array lets whichever rasterised last erase
    the rest of them.
    """
    occ = np.zeros(int(np.prod(dims)), bool)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    edge = np.maximum(np.abs(b - a).max(axis=1), np.abs(c - a).max(axis=1))
    need = np.ceil(edge / (cell * 0.4)).astype(np.int64)
    np.clip(need, 1, 256, out=need)
    per_comp: dict[int, list[np.ndarray]] = {}
    for m in np.unique(need):
        sel = np.where(need == m)[0]
        us, vs = np.meshgrid(np.linspace(0, 1, m + 1), np.linspace(0, 1, m + 1))
        keep = (us + vs) <= 1.0
        us, vs = us[keep], vs[keep]
        chunk = max(1, 4_000_000 // max(len(us), 1))
        for start in range(0, len(sel), chunk):
            ch = sel[start:start + chunk]
            pts = (a[ch][:, None, :] + us[None, :, None] * (b - a)[ch][:, None, :]
                   + vs[None, :, None] * (c - a)[ch][:, None, :])
            idx = np.floor((pts.reshape(-1, 3) - lo) / cell).astype(np.int64)
            np.clip(idx, 0, dims - 1, out=idx)
            flat = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
            occ[flat] = True
            comp = np.repeat(labels[ch], len(us))
            order = np.argsort(comp, kind="stable")
            comp, flat = comp[order], flat[order]
            bounds = np.searchsorted(comp, np.unique(comp))
            for cid, s, e in zip(np.unique(comp), bounds,
                                 list(bounds[1:]) + [len(comp)]):
                per_comp.setdefault(int(cid), []).append(np.unique(flat[s:e]))
    return (occ.reshape(dims),
            {k: np.unique(np.concatenate(v)) for k, v in per_comp.items()})


def sealed_labels(V: np.ndarray, F: np.ndarray, labels: np.ndarray, probe: float,
                  voxel: float, keep_extent: float = 0.0) -> tuple[set, dict]:
    """Which component labels a probe of radius ``probe`` can never reach from outside.

    ``V`` and ``F`` are the whole link; ``labels`` is one component id per triangle, as
    :func:`weldpath.meshprep.connected_shells` produces.  Returns the set of labels to drop
    and a dict of what was decided, for the caller to report.

    ``keep_extent`` is a refusal, not a threshold: a component whose bounding-box diagonal
    reaches it is kept however sealed it looks.  Something that large being called
    unreachable is more likely a rasterisation the geometry defeated than a motor, and the
    count comes back in the stats so it can be looked at rather than assumed.
    """
    stats: dict = {"voxel": float(voxel), "probe": float(probe),
                   "sealed": 0, "sealed_triangles": 0, "kept_large": 0, "largest": 0.0}
    if probe <= 0.0 or voxel <= 0.0 or not len(F):
        return set(), stats

    lo = V.min(axis=0) - PAD_CELLS * voxel
    hi = V.max(axis=0) + PAD_CELLS * voxel
    dims = np.ceil((hi - lo) / voxel).astype(np.int64) + 1
    if int(np.prod(dims)) > MAX_CELLS:
        # Coarsen rather than allocate.  Cubing the ratio back out of the cell count gives
        # the factor that brings the grid under the ceiling in one step; the probe is not
        # touched, so the answer stays the one that was asked for, screened more crudely.
        voxel *= float(np.cbrt(int(np.prod(dims)) / MAX_CELLS)) * 1.02
        lo = V.min(axis=0) - PAD_CELLS * voxel
        hi = V.max(axis=0) + PAD_CELLS * voxel
        dims = np.ceil((hi - lo) / voxel).astype(np.int64) + 1
        stats["voxel"] = float(voxel)
        stats["coarsened"] = True

    occ, per_comp = _rasterise(V, F, labels, voxel, lo, dims)
    grow = int(np.ceil(probe / voxel))
    if grow >= PAD_CELLS:
        # The pad has to outlast both dilations or the flood is seeded into solid.  A probe
        # this large against this voxel is a configuration error rather than a case to
        # handle, and answering it wrongly would drop the whole link.
        raise ValueError(f"enclosure probe {probe:g} mm needs a voxel above "
                         f"{probe / (PAD_CELLS - 1):.1f} mm, not {voxel:g}")
    reach = flood_outside(~dilate(occ, grow))
    # A component is reachable if any cell it occupies lies within the probe's grasp of air
    # the flood arrived at.  grow + 1 because the solid was grown by grow before flooding,
    # so reachable air stops that far short of the surface it is touching.
    near = dilate(reach, grow + 1).reshape(-1)

    ids, counts = np.unique(labels, return_counts=True)
    sealed: set = set()
    sealed_tris = kept_large = 0
    largest = 0.0
    for lab, n_tris in zip(ids, counts):
        flat = per_comp.get(int(lab))
        if flat is None or bool(near[flat].any()):
            continue
        P = V[np.unique(F[labels == lab])]
        diagonal = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))
        if keep_extent > 0.0 and diagonal >= keep_extent:
            kept_large += 1
            continue
        sealed.add(int(lab))
        sealed_tris += int(n_tris)
        largest = max(largest, diagonal)

    stats.update(sealed=len(sealed), sealed_triangles=sealed_tris,
                 kept_large=kept_large, largest=round(largest, 1))
    return sealed, stats
