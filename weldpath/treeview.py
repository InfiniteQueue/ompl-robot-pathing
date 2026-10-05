"""Write a failed segment's Cartesian trees out as something that can be looked at.

Called only when a segment has failed, from :meth:`weldpath.toolpath.ToolpathPlanner.run`,
with whatever :mod:`weldpath.treetrace` kept.  Three files land in the study directory --
the one the manifest was read from, which is the one directory this program is always
given:

* ``weldpath-treeview.js``   -- the viewer, written once and shared
* ``<segment>.treedata.js``  -- one segment's trees, geometry and poses
* ``<segment>.treeview.html``-- a stub that loads those two, so it opens on a double click

Why a ``.js`` data file rather than ``.json``: a browser opening a page from ``file://``
refuses to ``fetch`` a sibling file, so a page that loaded its data that way would come up
empty on the one machine this is for.  A classic ``<script src>`` to a file in the same
directory is allowed, so the data assigns itself to a global and the stub pulls both in.
Nothing is fetched and nothing is served.

**The geometry written is the geometry the planner collided against** -- the convex
decomposition, through :func:`weldpath.hullexport._link_hulls`, not the source meshes.
Those differ, and badly on the C-shaped castings in this cell: hulls over-report contact.
A viewer showing the source meshes would show a gun comfortably clear of something the
planner was certain it had hit, which is worse than showing nothing at all.

Static objects are written already placed in the world.  Moving links are written in their
own frames and placed per node from a baked transform, which is why ``_link_hulls`` is
handed an empty transform map for them: its lookup fails and it falls back to the identity,
leaving the geometry where the link's own frame puts it.

**Every search the segment kept goes into the one file**, each as a pair of trees under a
``solve`` index, and the viewer filters them the way it filters anything else.  A segment
runs the tree many times -- successive seeds, once per gun opening, again in phase two,
again per half of a detour -- and the questions worth asking are about the set: which
opening got furthest, whether a second half failed where the first half had no trouble.
One tree per file could not answer either, and writing a file per search would mean
comparing them by opening two windows.

The pose budget is therefore shared.  ``max_poses`` is a **total** over every tree in the
file, apportioned by node count in :func:`_share`, not a limit each tree gets to itself: a
baked pose costs a state load and a clearance query, which is a collision call, so a
per-tree limit would have multiplied the slowest part of this by the number of searches
kept and the file size with it.
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np

from . import hullexport

FORMAT = "weldpath-treeview/2"
# The viewer ships as ``treeview.js`` inside the package and lands beside the data
# under a name that says where it came from, the export directory being full of
# other people's files.
SRC = "treeview.js"
APP = "weldpath-treeview.js"

# Rounding for what goes in the file.  Positions land in manifest units, so 1 decimal is a
# tenth of a millimetre on this cell; rotations are unit quaternions, where 5 decimals is
# far finer than anything the geometry is known to.  It is worth being deliberate about:
# the baked poses are most of the file, and JSON spends a character per digit.
_POS_DP = 2
_ROT_DP = 5


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name))


def _quat(R: np.ndarray) -> list[float]:
    """A rotation matrix as ``[x, y, z, w]``.

    Shepperd's branch: take the largest of the four to divide by, so the square root is
    never taken of something near zero.  Done here rather than reached for from a library
    because this module must not add a dependency to a planner that has none.
    """
    m = np.asarray(R, dtype=float)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
             (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s,
             (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] >= m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s,
             (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s,
             0.25 * s, (m[1, 0] - m[0, 1]) / s]
    return [round(float(v), _ROT_DP) for v in q]


def _pieces(scene, transforms, name: str, unit: float) -> list[dict]:
    """One link's convex pieces as flat vertex and index arrays."""
    out = []
    for V, tris in hullexport._link_hulls(scene, transforms, name, unit):
        out.append({
            "v": [round(float(x), _POS_DP) for x in np.asarray(V).reshape(-1)],
            "f": [int(i) for tri in tris for i in tri],
        })
    return out


def _geometry(cell, man, log) -> dict:
    """The static cell in world coordinates, and every moving link in its own frame."""
    scene = cell.env.getSceneGraph()
    # Held: the bindings hand back a temporary whose link_transforms dangle if it dies.
    state = cell.env.getState()
    transforms = state.link_transforms
    unit = 1.0 / man.scale

    static, moving = [], []
    for s in man.static_objects:
        got = _pieces(scene, transforms, s.name, unit)
        if got:
            static.append({"name": s.name, "category": s.category, "pieces": got})
    for link in man.all_links():
        if not link.mesh:
            continue
        # Empty map on purpose: _link_hulls' lookup raises, it falls back to the identity,
        # and the geometry comes back in the link's own frame to be placed per node.
        got = _pieces(scene, {}, link.name, unit)
        if got:
            moving.append({"name": link.name, "pieces": got})
    log(f"      treeview: {len(static)} static objects, {len(moving)} moving links")
    return {"static": static, "moving": moving}


def _bake(cell, man, q: np.ndarray, links: list[str]) -> dict:
    """Where every moving link sits at ``q``, plus what the tool reads there.

    One state load for the whole set rather than a forward-kinematics call per link, which
    is the same trick ``toolpath._export_blocked`` uses: the environment already computes
    every link transform when the state is set, so asking per link pays for that again.
    """
    cell.set_state(q)
    state = cell.env.getState()         # held, as above
    transforms = state.link_transforms
    unit = 1.0 / man.scale
    pose = {}
    for name in links:
        try:
            T = np.array(transforms[name].matrix(), dtype=float)
        except Exception:
            continue
        pose[name] = {"p": [round(float(v * unit), _POS_DP) for v in T[:3, 3]],
                      "q": _quat(T[:3, :3])}
    return pose


def _share(counts: list[int], budget: int) -> list[int]:
    """Split one pose ``budget`` over trees of ``counts`` nodes, biggest trees first served.

    Proportional to node count, so a search that explored twice as far gets twice the
    clickable nodes, and walked largest first so the rounding lands on the trees where one
    pose either way matters least.  Every non-empty tree is guaranteed at least one, or a
    file with more trees than budget would contain whole searches with nothing clickable in
    them -- and a search with no pose at all is the one case where the viewer can draw the
    shape but never answer "where was the robot here".  That floor is the one thing here
    that can exceed the budget, and only where there are more non-empty trees than poses
    asked for, by at most one per tree.

    ``budget`` of 0 means no cap, which it also means at the flag, and comes back as 0 per
    tree for :func:`_tree` to read the same way.
    """
    if budget <= 0:
        return [0] * len(counts)
    out = [0] * len(counts)
    left = int(budget)
    order = sorted(range(len(counts)), key=lambda i: -counts[i])
    pool = sum(counts)
    for n, i in enumerate(order):
        if counts[i] <= 0:
            continue
        # One each is reserved for the trees still to come, so a tiny budget spreads over
        # the searches rather than being spent entirely on the first one.
        rest = sum(1 for j in order[n + 1:] if counts[j] > 0)
        want = int(round(left * counts[i] / pool)) if pool else left
        out[i] = max(1, min(want, max(1, left - rest)))
        left -= out[i]
        pool -= counts[i]
        if left <= 0:
            left = 0
    return out


def _tree(cell, man, tree, links: list[str], limit: int, log) -> dict:
    """One tree: its nodes with baked poses and clearance, and its edges' tool paths.

    An edge is drawn along ``_Node.chain`` -- the states the steer actually validated to
    get there -- rather than as a straight line between the two nodes.  The chain is the
    motion; a line between the nodes is a shortcut that was never checked and often has no
    inverse kinematics at all, which is precisely the shape this search exists to find.

    ``limit`` caps how many nodes get a baked pose, a pose being most of the file's size.
    Every node is still written, so the structure is whole and every edge still draws; the
    ones past the cap are simply not clickable, and the viewer says so.  Chosen uniformly
    rather than by taking the first N, so a capped tree still has poses spread over the
    whole of what it explored instead of only near its root.  This tree's share of the
    file's one budget, worked out by :func:`_share`; 0 is no cap.  What it reports is left
    to the caller, which is the only place that knows how many trees there are.
    """
    nodes = list(tree.nodes)
    n = len(nodes)
    keep = set(range(n))
    if 0 < limit < n:
        keep = set(int(i) for i in np.linspace(0, n - 1, limit).round())

    out_q, out_parent, out_clr, out_tcp, out_pose = [], [], [], [], []
    for i, node in enumerate(nodes):
        out_q.append([round(float(v), 6) for v in np.asarray(node.q).reshape(-1)])
        out_parent.append(int(node.parent))
        # Already in manifest units: the tree stores cell.pose_mm, which is fk with the
        # scale taken out.  The edges below go through fk and have to divide it out
        # themselves, and the two have to agree or the nodes float off their own edges.
        out_tcp.append([round(float(v), _POS_DP)
                        for v in np.asarray(node.pose)[:3, 3]])
        if i in keep:
            out_clr.append(round(float(cell.clearance_mm(node.q)), 2))
            out_pose.append(_bake(cell, man, node.q, links))
        else:
            out_clr.append(None)
            out_pose.append(None)

    edges = []
    for i, node in enumerate(nodes):
        if node.parent < 0 or not len(node.chain):
            continue
        path = []
        for q in node.chain:
            p = np.asarray(cell.fk(np.asarray(q, dtype=float)))[:3, 3] / man.scale
            path.append([round(float(v), _POS_DP) for v in p])
        edges.append({"a": int(node.parent), "b": int(i), "path": path})

    return {"count": n, "q": out_q, "parent": out_parent, "clearance": out_clr,
            "tcp": out_tcp, "pose": out_pose, "edges": edges}


def _chord(cell, man, qa: np.ndarray, qb: np.ndarray, links: list[str],
           samples: int) -> dict:
    """The straight joint chord between the endpoints, sampled, with its clearance.

    Always written, tree or no tree.  A segment can fail with no Cartesian tree at all --
    no band in force, a locator that never placed, the clock gone before the tree got a
    turn -- and a file that said nothing in those cases would be a puzzle rather than a
    diagnostic.  This is the one thing that always exists.
    """
    a, b = np.asarray(qa, dtype=float), np.asarray(qb, dtype=float)
    out_tcp, out_clr, out_pose, out_q = [], [], [], []
    for t in np.linspace(0.0, 1.0, max(2, samples)):
        q = a + t * (b - a)
        out_q.append([round(float(v), 6) for v in q])
        out_tcp.append([round(float(v), _POS_DP)
                        for v in np.asarray(cell.fk(q))[:3, 3] / man.scale])
        out_clr.append(round(float(cell.clearance_mm(q)), 2))
        out_pose.append(_bake(cell, man, q, links))
    return {"q": out_q, "tcp": out_tcp, "clearance": out_clr, "pose": out_pose}


def write_app(directory: str, log=print) -> str | None:
    """Put the viewer itself in ``directory``, overwriting an older copy.

    Overwritten rather than left alone: it is this package's own file, so a stale one beside
    fresh data is a viewer that may not understand it.
    """
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), SRC)
    if not os.path.isfile(src):
        log(f"      treeview: the viewer is missing from the install ({src})")
        return None
    dst = os.path.join(directory, APP).replace("\\", "/")
    shutil.copyfile(src, dst)
    return dst


def write(cell, man, directory: str, *, segment: str, failure: str,
          qa, qb, kept=None, searches: int = 0, dropped: int = 0, opening=None,
          max_poses: int = 0, chord_samples: int = 24, log=print) -> str | None:
    """Write one failed segment's viewer files.  Returns the HTML to open, or ``None``.

    ``kept`` is every search :mod:`weldpath.treetrace` held, in the order it was offered;
    ``searches`` and ``dropped`` are how many were offered and how many the limit turned
    away, which is what lets the file say it is a sample rather than the whole of it.

    Never raises: a diagnostic that takes the run down with it is worse than no diagnostic.
    The caller is already handling a failure and must be left to report it.
    """
    try:
        os.makedirs(directory, exist_ok=True)
        gun = set()
        for device in man.devices[1:]:
            gun.update(link.name for link in device.links if link.mesh)
        links = [link.name for link in man.all_links() if link.mesh]

        data = {
            "format": FORMAT,
            "segment": segment,
            "failure": failure,
            "units": man.units,
            "searches": searches,
            "dropped": dropped,
            "gun_opening_mm": None if opening is None else float(opening),
            "clearance": {
                "margin": round(float(cell.obstacle_clearance / man.scale), 3),
                "probe": round(float(cell.probe_mm), 3),
            },
            "links": {"all": links, "gun": sorted(gun),
                      "arm": [n for n in links if n not in gun]},
            "geometry": _geometry(cell, man, log),
            "solves": [],
            "trees": [],
        }
        # Only where there are two endpoints to draw one between.  A segment that failed at
        # a locator has none -- that is what it failed on -- and the file is written anyway,
        # so this cannot be allowed to be the thing that stops it.
        if qa is not None and qb is not None:
            data["chord"] = _chord(cell, man, qa, qb, links, chord_samples)
        # Flattened to trees before the budget is split, so the apportionment is one call
        # over the whole file rather than a budget handed down and divided twice.
        flat = []
        for s, search in enumerate(kept or []):
            flat.append((s, "start", search.start))
            flat.append((s, "goal", search.goal))
        shares = _share([len(t.nodes) for _, _, t in flat], max_poses)
        for s, search in enumerate(kept or []):
            data["solves"].append({
                "id": int(getattr(search, "order", s + 1)),
                "label": str(search.label),
                "nodes": int(len(search.start.nodes) + len(search.goal.nodes)),
            })
        for (s, name, tree), share in zip(flat, shares):
            data["trees"].append({"solve": s, "name": name,
                                  **_tree(cell, man, tree, links, share, log)})
        baked = sum(sum(1 for p in t["pose"] if p is not None) for t in data["trees"])
        if data["solves"]:
            log(f"      treeview: {len(data['solves'])} search"
                f"{'' if len(data['solves']) == 1 else 'es'} of {searches} offered"
                + (f", {dropped} dropped by --treeview-searches" if dropped else "")
                + f"; {baked} of {sum(t['count'] for t in data['trees'])} nodes have a "
                f"baked pose (--treeview-max-poses)")

        stem = _safe(segment)
        body = json.dumps(data, separators=(",", ":"), allow_nan=False)
        js = os.path.join(directory, f"{stem}.treedata.js").replace("\\", "/")
        with open(js, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("window.WELDPATH_TREE = ")
            fh.write(body)
            fh.write(";\n")

        html = os.path.join(directory, f"{stem}.treeview.html").replace("\\", "/")
        with open(html, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(_STUB.format(title=segment, data=f"{stem}.treedata.js", app=APP))
        write_app(directory)

        size = os.path.getsize(js) / 1e6
        total = sum(t["count"] for t in data["trees"])
        log(f"      treeview: wrote {total} nodes over {len(data['trees'])} trees in "
            f"{len(data['solves'])} search{'' if len(data['solves']) == 1 else 'es'} "
            f"({size:.1f} MB) -- open {html}")
        return html
    except Exception as exc:            # a diagnostic must not mask the failure it reports
        log(f"      treeview: could not write the failure view: "
            f"{type(exc).__name__}: {exc}")
        return None


_STUB = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>weldpath: {title}</title>
</head>
<body>
<script src="{data}"></script>
<script src="{app}"></script>
</body>
</html>
"""
