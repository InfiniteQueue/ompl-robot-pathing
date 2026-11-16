"""Every search tree a segment built, kept in case the segment fails.

Mirrors :mod:`weldpath.stagetrace`: off until ``start`` is called, and every call is a
no-op while it is, so :func:`weldpath.cartesian.plan_cartesian` and
:func:`weldpath.jointtree.plan_joint` can offer their trees unconditionally without
knowing whether anyone is listening.

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

**Both searches are kept, Cartesian and joint.**  A segment can run the Cartesian tree on
the direct transit and the joint tree in the sampling planner's slot, and the pair of them
is the comparison worth having: the straight tool move and the joint chord on the same
endpoints, failing where each fails.  ``Search.kind`` says which is which, so one file
answers that without two runs and two windows.

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
    """One offered pair of trees, in the order it was offered.

    ``kind`` says which search built the pair, and it is not decoration: the two kinds
    hold different things.  A Cartesian node carries the tool pose it was reached at and
    the states its edge validated to get there; a joint node carries neither, a joint
    chord being the same points every time it is asked for.  :mod:`weldpath.treeview`
    derives both from ``q`` as it writes the file, which is where that work belongs --
    only a failed segment reads it.  The viewer shows the kind per search because an edge
    is a straight tool move in one and a joint chord in the other.
    """

    __slots__ = ("order", "label", "kind", "nodes", "start", "goal")

    def __init__(self, order: int, label: str, kind: str, nodes: int, start, goal):
        self.order = order
        self.label = label
        self.kind = kind
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


def offer(label: str, start_tree, goal_tree, kind: str = "cartesian") -> None:
    """Offer one search's pair of trees.

    Called whatever the search returned.  A run that *solved* is offered too, because a
    segment can fail long after a transit solved -- a later transit in the same segment,
    or the validation at the end -- and what that run explored is still part of the
    picture.

    ``kind`` is ``"cartesian"`` or ``"joint"``.  Both go in the one list, because a
    segment runs both -- the Cartesian tree on the direct transit, the joint tree in the
    sampling planner's slot -- on the same two endpoints in the same cell, and the
    comparison is the whole point of having kept them.

    **No kind that ran is ever absent from what is kept.**  The size rule on its own would
    not hold that.  A joint search that solved goes in at six nodes while a Cartesian pair
    runs to hundreds, so a segment with a full set of big Cartesian pairs would turn away
    every joint pair offered, and the file would say the joint tree never ran when it ran
    and solved.  So a kind not yet kept takes the place of the smallest pair of a kind that
    has more than one, whatever its own size, and no pair is ever evicted if its kind would
    go with it.  Within that, size decides as it did: the biggest seen so far.

    A ``limit`` below the number of kinds cannot hold them all, and then the first kind to
    fill it is the one kept -- there is no room for the second, and dropping one of these
    to make it would be the same loss by another name.
    """
    global _runs, _dropped
    if not _on:
        return
    _runs += 1
    try:
        size = len(start_tree.nodes) + len(goal_tree.nodes)
    except Exception:                   # diagnostics must never break a search
        return
    got = Search(_runs, str(label), str(kind), size, start_tree, goal_tree)
    if not _limit or len(_kept) < _limit:
        _kept.append(got)
        return
    _dropped += 1
    # Full.  A pair may only make way if its kind survives without it -- either another
    # pair of that kind is kept, or the one arriving is of that kind itself.
    counts: dict[str, int] = {}
    for s in _kept:
        counts[s.kind] = counts.get(s.kind, 0) + 1
    room = [s for s in _kept if s.kind == got.kind or counts[s.kind] > 1]
    if not room:                        # every kept pair is the last of its kind
        return
    small = min(room, key=lambda s: s.nodes)
    # A kind nobody is holding goes in whatever its size; past that, only something bigger
    # displaces anything, so the set held is the biggest seen rather than the most recent.
    if got.kind in counts and size <= small.nodes:
        return
    _kept.remove(small)
    _kept.append(got)


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


def kinds() -> dict:
    """How many kept pairs each kind has, for a report saying both searches are in there.

    The counts and not just the names: a file holding six Cartesian pairs and one joint
    pair is a different thing to read from one holding three of each, and the viewer says
    so rather than leaving it to be counted off the panel.
    """
    out: dict[str, int] = {}
    for s in _kept:
        out[s.kind] = out.get(s.kind, 0) + 1
    return out
