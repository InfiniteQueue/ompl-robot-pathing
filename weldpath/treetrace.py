"""Every Cartesian tree a segment built, kept in case the segment fails.

Mirrors :mod:`weldpath.stagetrace`: off until ``start`` is called, and every call is a
no-op while it is, so :func:`weldpath.cartesian.plan_cartesian` can offer its trees
unconditionally without knowing whether anyone is listening.

Nothing here touches the cell, the geometry or the disk.  A segment that succeeds must not
pay for a diagnostic it will never write, so the trees are held exactly as the search left
them and :mod:`weldpath.treeview` does all of the work -- forward kinematics, hulls,
serialising -- only once a segment has actually failed.

**Both trees of a pair are kept, and every pair is offered.**  The two grow towards each
other and the interesting thing is the gap between them: which one got how far, and where
they stopped short of meeting.  Keeping the deeper branch of one tree would hide exactly
that, and keeping one tree of the pair would hide half of it.

**Several searches are kept, not just the biggest one.**  A segment runs the tree many
times -- successive seeds until the solve budget is spent, once per gun opening, again in
phase two, and again per half of a detour -- and they fail in different places.  Which
opening got furthest, and whether a second half failed where the first half had no trouble,
are questions about the *set* of searches; the single biggest tree cannot answer either, and
reading them one failed run at a time is what this replaced.

What that costs is memory, and it is the reason for a limit.  Holding a reference keeps a
search's nodes alive after the search that built them has returned, and a pair can run to
thousands of nodes with a validated chain hanging off each one.  ``limit`` bounds how many
pairs are held at once; past it the **smallest** kept pair makes way for a bigger one, so
what survives is the ``limit`` biggest.  The count offered and the count dropped are both
reported, because a cap that quietly kept less would be indistinguishable from a segment
that searched less.

What is lost to the cap, then, is the searches that died early -- and those are informative
too: a tree of four nodes says the endpoint is boxed in at that opening.  Raising the limit
is how to see them.  ``limit=0`` keeps everything offered.

Everything held is dropped by ``start`` on the next segment and by ``stop`` at the end of
this one.
"""
from __future__ import annotations


class Search:
    """One offered pair of trees, in the order it was offered."""

    __slots__ = ("order", "label", "nodes", "start", "goal")

    def __init__(self, order: int, label: str, nodes: int, start, goal):
        self.order = order
        self.label = label
        self.nodes = nodes
        self.start = start
        self.goal = goal


_on = False
_kept: list[Search] = []
_runs = 0
_dropped = 0
_limit = 0


def start(limit: int = 0) -> None:
    """Begin keeping trees for one segment, dropping anything held for the last.

    ``limit`` is how many pairs may be held at once; 0 holds everything offered.
    """
    global _on, _kept, _runs, _dropped, _limit
    _on, _kept, _runs, _dropped = True, [], 0, 0
    _limit = max(0, int(limit))


def stop() -> None:
    """Stop keeping, and let go of whatever was held."""
    global _on, _kept, _runs, _dropped
    _on, _kept, _runs, _dropped = False, [], 0, 0


def active() -> bool:
    """Whether anything is listening, for a caller deciding whether to bother."""
    return _on


def offer(label: str, start_tree, goal_tree) -> None:
    """Offer one search's pair of trees.

    Called whatever the search returned.  A run that *solved* is offered too, because a
    segment can fail long after a transit solved -- a later transit in the same segment,
    or the validation at the end -- and what that run explored is still part of the
    picture.
    """
    global _runs, _dropped
    if not _on:
        return
    _runs += 1
    try:
        size = len(start_tree.nodes) + len(goal_tree.nodes)
    except Exception:                   # diagnostics must never break a search
        return
    got = Search(_runs, str(label), size, start_tree, goal_tree)
    if not _limit or len(_kept) < _limit:
        _kept.append(got)
        return
    # Full.  The smallest kept pair makes way, and only for something bigger, so the set
    # held is always the biggest seen so far rather than the most recent.
    small = min(_kept, key=lambda s: s.nodes)
    if size <= small.nodes:
        _dropped += 1
        return
    _kept.remove(small)
    _kept.append(got)
    _dropped += 1


def kept() -> list[Search]:
    """Every pair still held, in the order it was offered.

    Offer order rather than size order: the labels count runs, so a list that read 1, 2, 5
    says which runs went missing, where a list sorted by size says nothing about either.
    """
    return sorted(_kept, key=lambda s: s.order)


def searches() -> int:
    """How many searches were offered, for the report to say what it is a sample of."""
    return _runs


def dropped() -> int:
    """How many offered pairs the limit turned away."""
    return _dropped
