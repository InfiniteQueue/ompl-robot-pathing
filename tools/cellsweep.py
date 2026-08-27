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


# hullexport._hull culls the interior of a large point set before hulling it, which is
# what makes the raw concave shells here tractable -- see HULL_CULL_MIN there for the
# divergence it defends against.  Its own bar is set for the hull vertices Tesseract
# hands that module; these shells are raw geometry, so the cull is asked for from much
# lower down.
CULL_MIN = 16


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
            tris = hullexport._hull(V, cull_min=CULL_MIN)
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
