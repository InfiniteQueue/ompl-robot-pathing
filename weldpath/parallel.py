"""Run independent solve and polish attempts on threads, without scrambling the log.

**Why threads and not processes.**  The planners here are Python loops, and a Python loop
cannot run beside another one -- but almost none of the time goes on the loop.  It goes
inside the bindings: a state load, a ``contactTest``, an inverse kinematics solve, and
those release the interpreter lock.  Measured on this machine with a trivial two-joint
cell, a Python counter kept **92%** of its solo rate while another thread drove collision
checks, and splitting 8000 checks over two threads took 371 ms against 524 ms on one, a
1.41x speedup.  That ratio is the floor rather than the figure to expect: in that test the
C++ work per iteration was microseconds, where on a real cell one check is 0.13 ms in open
air and 3.6 ms near a panel, so the share of the time that parallelises is far higher
there.

Processes were the alternative and they are the wrong trade for this program.  There is no
``fork`` on Windows, so each worker would load the environment itself -- minutes of mesh
preparation and a second copy of the geometry per worker -- and every route would have to
be pickled back.  Threads share the geometry outright: see :meth:`weldpath.cell.Cell.worker`,
where a worker's whole private cost is its own broadphase proxies and its own caches.

**What is parallel is attempts, never one attempt.**  Every search and every polish here is
a loop that reads what the last iteration left, so nothing inside one of them is safe to
split.  What is independent is the *attempts*: a tree search at one gun opening knows
nothing about the one at another, and a polish run from one seed knows nothing about the
run from the next.  Those already existed as a set and were already ranked as a set -- this
runs that set at once rather than one after another.

**The log stays in the order the attempts were submitted.**  A parallel phase whose lines
interleaved would be worse than a slower one: this program's log is read as a narrative,
every figure is attributed to a run, and attempts here finish out of order by nature.  So
each attempt writes to a buffer of its own and :func:`run` replays those buffers in
submission order, flushing each as soon as every attempt before it has finished.  The
transcript is therefore identical in shape to a sequential run -- the same lines, the same
order -- and only the timings differ.

**One worker is not "parallel with one thread".**  At ``workers <= 1`` :func:`run` calls
each attempt inline on the calling thread with the caller's own ``log``, so there is no
pool, no buffer and no thread.  That is the path a run takes with ``--workers 1``, and it
has to be indistinguishable from the code as it was before any of this existed.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

__all__ = ["configure", "setting", "threads", "run", "Buffer"]

# Above this many threads the contention on the interpreter lock -- which every attempt
# still takes to run its own Python -- costs more than the extra worker brings back, and
# the per-worker broadphase proxies start to be worth counting.  A cap rather than a
# setting because the figure is about this program's shape, not about a machine; ``auto``
# is what adapts to the machine.
MAX_WORKERS = 8


# What ``--workers`` resolved to, set once at the start of a run.  A setting rather than
# an argument because the places that want it are five and six calls below the flag --
# ``main`` to the planner to a transit to a leg to the refinement to the polish -- and
# threading one integer through every signature on that path would say nothing at any of
# them.  It is written once, before anything plans, and only read afterwards, so it is not
# shared mutable state in any sense that matters to the threads it configures.
_workers = 1


def configure(value: int) -> int:
    """Resolve ``--workers`` and hold it for the run.  Returns what it resolved to.

    ``0`` is auto: one thread per core bar one, capped at :data:`MAX_WORKERS`.  The core
    left out is not spare -- the parent thread ranks and refines what the attempts return,
    and the rest of the machine belongs to whoever is using it.
    """
    global _workers
    if value and int(value) > 0:
        _workers = int(value)
    else:
        cores = os.cpu_count() or 1
        _workers = max(1, min(MAX_WORKERS, cores - 1))
    return _workers


def setting() -> int:
    """How many threads this run may use.  1 until :func:`configure` says otherwise."""
    return _workers


def threads(jobs: int, workers: int | None = None) -> int:
    """How many threads to put ``jobs`` attempts on: never more threads than attempts.

    ``workers`` overrides the run's setting, which is what the tests use and what a caller
    with a reason to stay sequential passes 1 for.
    """
    want = setting() if workers is None else int(workers)
    return max(1, min(want, int(jobs)))


class Buffer:
    """A ``log`` that keeps its lines instead of printing them.

    Handed to one attempt so that what it prints stays attributable to it.  Nothing here
    is synchronised: a buffer belongs to exactly one attempt, and therefore to exactly one
    thread, for its whole life.
    """

    __slots__ = ("lines",)

    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, line: str = "") -> None:
        self.lines.append(line)

    def flush_to(self, log) -> None:
        for line in self.lines:
            log(line)
        self.lines.clear()


def run(jobs, workers: int, log=print) -> list:
    """Run each attempt, return what they returned, in submission order.

    ``jobs`` is a sequence of callables taking one argument -- the ``log`` that attempt
    should write to -- and returning whatever the caller wants back.  An attempt that
    raises takes the phase down with it, as it would have done sequentially; the planners
    wired to this catch their own ``PlanningError`` inside the attempt, because a search
    that failed is a result here rather than an error.

    The buffers are flushed in order and as early as order allows, so a phase of eight
    attempts prints the first the moment it is done rather than holding everything to the
    end.  What that costs is that a slow first attempt holds back the lines of the quick
    ones behind it -- which is the price of a log that reads the same as a sequential one,
    and it is worth paying.
    """
    jobs = list(jobs)
    if not jobs:
        return []
    if workers <= 1 or len(jobs) == 1:
        # Inline, with the caller's own log: the sequential path, unchanged.
        return [job(log) for job in jobs]

    buffers = [Buffer() for _ in jobs]
    results: list = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs)),
                            thread_name_prefix="weldpath") as pool:
        futures = [pool.submit(job, buf) for job, buf in zip(jobs, buffers)]
        for i, future in enumerate(futures):
            # Waiting on them in order is what makes the output ordered.  It costs
            # nothing: every attempt is already running, and the last one finishes when
            # it finishes whichever order they are read in.
            results[i] = future.result()
            buffers[i].flush_to(log)
    return results
