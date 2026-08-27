"""Convex visualisations of a mesh at a range of --hull-cell-mm values.

What --export-collision-geometry writes is the hulls read back out of a loaded
environment.  There is no environment here and no manifest to place these meshes in one,
so this takes the same geometry from the other end: decompose_to_obj splits a mesh into
the `o shell_NNNNN` groups the URDF hands Tesseract, and Tesseract builds one convex hull
per group at load time because the mesh is tagged tesseract:make_convex="true".  Hulling
each group with hullexport's own _hull therefore produces the shapes Bullet would test,
without needing the cell.

Refinement is left unfocused -- no weld or TCP proximity -- so the cell size is the only
thing that differs between the files.  Vertices stay in the source mesh's own units, so
each output drops straight on top of its input in a viewer.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = r"C:\Users\christopherf\Documents\Local Code Projects\Tesseract\Tesseract"
sys.path.insert(0, ROOT)

from weldpath import hullexport, meshprep

MESH_DIR = r"C:\Users\christopherf\Documents\TestFiles\Test Studies\Tesseract\meshes"
OUT_DIR = r"C:\Users\christopherf\Documents\TestFiles\Test Studies\Tesseract\cell_size_scaling"
MESHES = ["gun_body", "gun_moving_tip",
          "static_CX430_4707_ST200_000_R_1", "static_Assy_ST200_RH_0"]
CELLS = [15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0]

# main.py's defaults for everything the sweep is not varying.
MIN_EXTENT = 10.0          # --min-shell-mm
MAX_SHELLS = 1500          # --max-shells
FILL = 0.75                # --hull-fill
OVERLAP = 0.02             # --hull-cell-overlap
SCALE = 1.0                # source units, so the output overlays the source


# One parse per mesh instead of one per cell size: the largest of these is 149 MB, and
# decompose_to_obj would otherwise re-read it eight times.  Copies go out rather than the
# cached arrays themselves, because refinement appends vertices to the array it is given.
_loaded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_real_load = meshprep.load_obj


def cached_load(path: str):
    key = os.path.abspath(path)
    if key not in _loaded:
        t0 = time.time()
        _loaded[key] = _real_load(path)
        print("    parsed %.1f MB in %.1fs" % (os.path.getsize(path) / 1e6,
                                               time.time() - t0), flush=True)
    V, F = _loaded[key]
    return V.copy(), F


meshprep.load_obj = cached_load


# hullexport._hull is an incremental insertion sized for its own job: the points Tesseract
# hands back are already a hull's vertices, forty or so of them and well separated.  Here
# it is given the raw concave shell instead, and on geometry with many points sitting
# within eps of a face it does not merely slow down -- it diverges.  One 362-point shell
# of Assy_ST200_RH came back with 24,670,533 triangles after 109 seconds, where a hull of
# n points can have at most 2n - 4, here 720.  Size is not the trigger: a 1270-point shell
# alongside it hulls correctly in 0.1s.
#
# So the point set is cut down first, and cut down provably: the hull of a subset is
# contained in the hull of the whole, so a point inside that subset hull cannot be a vertex
# of the full one.  Taking the extremes along a spread of directions gives a subset hull
# that is already nearly the answer, and what survives the cull is the true vertex set plus
# whatever lies within TOLERANCE of a face.  That removes the near-coplanar crowd the
# degeneracy feeds on, and the same shell then hulls in 0.029s.
DIRECTIONS = 160
TOLERANCE = 1e-6          # of the shell's own diagonal, so micrometres on a 300 mm part


def _directions(n: int) -> np.ndarray:
    """``n`` roughly equidistant unit vectors, by the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.column_stack([np.cos(theta) * np.sin(phi),
                            np.sin(theta) * np.sin(phi), np.cos(phi)])


_DIRS = _directions(DIRECTIONS)


def _sane(tris, n: int) -> bool:
    """Euler's bound: a convex hull of ``n`` points has at most ``2n - 4`` triangles."""
    return len(tris) <= max(2 * n - 4, 2)


def fast_hull(V: np.ndarray, threshold: int = 16) -> list[tuple[int, int, int]]:
    """``hullexport._hull``, with the interior of the point set removed first."""
    n = len(V)
    span = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0))) if n else 0.0
    if n <= threshold or span <= 0.0:
        return hullexport._hull(V)
    proj = V @ _DIRS.T
    seed = np.unique(np.concatenate([proj.argmax(axis=0), proj.argmin(axis=0)]))
    if len(seed) < 4:
        return hullexport._hull(V)
    tris = hullexport._hull(V[seed])
    if not tris:
        return hullexport._hull(V)
    N, O = hullexport._outward(V[seed], np.asarray(tris, dtype=np.int64))
    outside = np.zeros(n, dtype=bool)
    for lo in range(0, n, 4096):            # the height table is n x faces; keep it small
        chunk = V[lo:lo + 4096]
        outside[lo:lo + 4096] = ((chunk @ N.T) - O).max(axis=1) > TOLERANCE * span
    keep = np.unique(np.concatenate([seed, np.nonzero(outside)[0]]))
    final = hullexport._hull(V[keep])
    if not _sane(final, len(keep)):
        # The cull did not remove enough to keep the insertion in hand.  The seed hull is
        # a valid hull of a subset, so falling back to it under-claims rather than
        # returning nonsense -- but it has never been needed, so say so if it ever is.
        print("    ! degenerate hull on %d points, falling back to the %d-point seed"
              % (len(keep), len(seed)), flush=True)
        return [(int(seed[a]), int(seed[b]), int(seed[c])) for a, b, c in tris]
    return [(int(keep[a]), int(keep[b]), int(keep[c])) for a, b, c in final]


def read_groups(path: str):
    """The `o` groups of an OBJ, as (name, vertices, triangles) with local indices."""
    groups, name, verts, tris, base = [], None, [], [], 0

    def flush():
        if name is not None and verts:
            groups.append((name, np.asarray(verts, dtype=float),
                           np.asarray(tris, dtype=int).reshape(-1, 3)))

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line[2:].split()])
            elif line.startswith("f "):
                tris.append([int(x.split("/")[0]) - 1 - base for x in line[2:].split()])
            elif line.startswith("o "):
                flush()
                base += len(verts)
                name, verts, tris = line[2:].strip(), [], []
    flush()
    return groups


def sweep(stem: str, cells, out_dir: str, scratch: str):
    src = os.path.join(MESH_DIR, stem + ".obj")
    rows = []
    for cell in cells:
        t0 = time.time()
        tmp = os.path.join(scratch, "shells_%s_%g.obj" % (stem, cell))
        stats = meshprep.decompose_to_obj(src, tmp, SCALE, min_extent=MIN_EXTENT,
                                          max_shells=MAX_SHELLS, hull_cell=cell,
                                          fill_threshold=FILL, focus_points=None,
                                          focus_radius=0.0, overlap=OVERLAP)
        groups = read_groups(tmp)
        hulls, verts = [], 0
        for name, V, _tris in groups:
            tris = fast_hull(V)
            if not tris:
                continue
            hulls.append((name.replace("shell", "hull"), (V, tris)))
            verts += len(V)
        dst = os.path.join(out_dir, "%s_cell%03gmm.obj" % (stem, cell))
        faces = hullexport._write_obj(dst, [
            "weldpath convex collision geometry, source=%s.obj" % stem,
            "hull-cell-mm=%g, hull-fill=%g, min-shell-mm=%g, max-shells=%d, overlap=%g"
            % (cell, FILL, MIN_EXTENT, MAX_SHELLS, OVERLAP),
            "refinement unfocused; %d hulls, %d triangles of source mesh"
            % (len(hulls), stats["triangles"]),
            "units are the source mesh's own, so this overlays it directly",
        ], hulls)
        os.remove(tmp)
        dt = time.time() - t0
        rows.append((cell, len(hulls), verts, faces, dt))
        print("    %5.0f mm -> %5d hulls, %7d verts, %7d tris  (%5.1fs)"
              % (cell, len(hulls), verts, faces, dt), flush=True)
    return rows


if __name__ == "__main__":
    scratch = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(OUT_DIR, exist_ok=True)
    stems = sys.argv[1].split(",") if len(sys.argv) > 1 else MESHES
    cells = [float(c) for c in sys.argv[2].split(",")] if len(sys.argv) > 2 else CELLS
    for stem in stems:
        print("  %s" % stem, flush=True)
        sweep(stem, cells, OUT_DIR, scratch)
