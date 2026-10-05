"""The biggest Cartesian tree a segment built, kept in case the segment fails.

Mirrors :mod:`weldpath.stagetrace`: off until ``start`` is called, and every call is a
no-op while it is, so :func:`weldpath.cartesian.plan_cartesian` can offer its trees
unconditionally without knowing whether anyone is listening.

Nothing here touches the cell, the geometry or the disk.  A segment that succeeds must not
pay for a diagnostic it will never write, so the trees are held exactly as the search left
them and :mod:`weldpath.treeview` does all of the work -- forward kinematics, hulls,
serialising -- only once a segment has actually failed.

**Biggest is most nodes over both trees, and both are kept.**  The two grow towards each
other and the interesting thing is the gap between them: which one got how far, and where
they stopped short of meeting.  Keeping the deeper branch of one tree would hide exactly
that, and keeping one tree of the pair would hide half of it.

Holding a reference to the trees keeps their nodes alive after the search that built them
has returned.  That is the point, and it is bounded: one pair of trees per segment, dropped
by ``start`` on the next one and by ``stop`` at the end of this one.
"""
from __future__ import annotations

_on = False
_best: tuple[str, object, object] | None = None
_runs = 0
_nodes = 0


def start() -> None:
    """Begin keeping trees for one segment, dropping anything held for the last."""
    global _on, _best, _runs, _nodes
    _on, _best, _runs, _nodes = True, None, 0, 0


def stop() -> None:
    """Stop keeping, and let go of whatever was held."""
    global _on, _best, _runs, _nodes
    _on, _best, _runs, _nodes = False, None, 0, 0


def active() -> bool:
    """Whether anything is listening, for a caller deciding whether to bother."""
    return _on


def offer(label: str, start_tree, goal_tree) -> None:
    """Offer one search's pair of trees; kept only if it explored more than the best so far.

    Called whatever the search returned.  A run that *solved* is offered too, because a
    segment can fail long after a transit solved -- a later transit in the same segment,
    or the validation at the end -- and the biggest tree the segment built is still the
    most informative thing it has.
    """
    global _best, _runs, _nodes
    if not _on:
        return
    _runs += 1
    try:
        size = len(start_tree.nodes) + len(goal_tree.nodes)
    except Exception:                   # diagnostics must never break a search
        return
    if size > _nodes:
        _best, _nodes = (label, start_tree, goal_tree), size


def kept() -> tuple[str, object, object] | None:
    """The biggest pair offered, or ``None`` where no search ran or none was offered."""
    return _best


def searches() -> int:
    """How many searches were offered, for the report to say what it is a sample of."""
    return _runs
