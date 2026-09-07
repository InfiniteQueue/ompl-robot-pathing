"""Finding poses worth routing a difficult transit through.

The finder walks :data:`weldpath.fallback.methods.METHODS` in order, testing each
candidate the method offers until one passes, and moves on to the next method either way.
Every method that succeeds contributes one pose, so the transit gets a list in preference
order rather than a single answer -- the planner tries them in turn, and a method that
found somewhere unhelpful costs one more attempt rather than the whole transit.

This is per transit, not per run.  Both ends of the move enter every method and the
validity rule, so a pose found for one transit says nothing about the next; the previous
scheme, which took the first via in the study and used it everywhere, was cheap precisely
because it ignored all of that.

A walk ends when the arm runs out of reach.  A step with no inverse-kinematics solution
stops that method: everything further along the same line is unreachable too, so there is
nothing left to test.  ``methods.MAX_STEPS`` is only a backstop for a direction that stays
reachable indefinitely, and is set high enough not to bind on a real arm.

It is also lazy.  Most transits solve directly and never ask, and a search that walks
three methods to the end of the robot's reach is not something to spend on every pair for
the sake of the few that need it.  ``FallbackFinder.poses`` is called only once the direct
attempts have failed.
"""
from __future__ import annotations

import time

import numpy as np

from . import methods, neutral
from .validity import NO_IK, Validator


class FallbackFinder:
    """Builds fallback poses for one cell, on demand, transit by transit."""

    def __init__(self, cell, man, *, fallback_mm: float, step_mm: float, log=print):
        self.cell = cell
        self.man = man
        self.fallback_mm = float(fallback_mm)
        self.step_mm = float(step_mm)
        self.log = log

    def announce(self) -> None:
        """Say what the search will do, once, before any transit needs it."""
        self.log(f"  fallback poses: keeping {self.fallback_mm:g} mm off the parts, "
                 f"stepping {self.step_mm:g} mm at a time until the arm runs out of "
                 f"reach")
        self.log(f"  a neutral pose is {neutral.describe()}")

    def poses(self, qa: np.ndarray, qb: np.ndarray,
              pose_a: np.ndarray, pose_b: np.ndarray) -> list[np.ndarray]:
        """Fallback poses for the transit from ``qa`` to ``qb``, most preferred first."""
        ctx = methods.Context(cell=self.cell, man=self.man,
                              qa=np.asarray(qa, dtype=float),
                              qb=np.asarray(qb, dtype=float),
                              pose_a=np.asarray(pose_a, dtype=float),
                              pose_b=np.asarray(pose_b, dtype=float),
                              step_mm=self.step_mm)
        validator = Validator(self.cell, qa, qb, self.fallback_mm)
        out: list[np.ndarray] = []
        for method in methods.METHODS:
            out.extend(self._run(method, ctx, validator))
        if not out:
            self.log("      no fallback pose was found by any method; this transit will "
                     "be attempted directly or not at all")
        return out

    def _run(self, method: methods.Method, ctx: methods.Context,
             validator: Validator) -> list[np.ndarray]:
        """One method, timed and reported.  A list so a method may yet return none."""
        started = time.perf_counter()
        tried = 0
        last = ""
        try:
            for pose in method.generate(ctx):
                tried += 1
                verdict = validator.check(pose)
                if verdict.valid:
                    self.log(f"      fallback pose from {method.name}: found after "
                             f"{tried} step{'' if tried == 1 else 's'}, "
                             f"{verdict.clearance_mm:.0f} mm clear, in "
                             f"{time.perf_counter() - started:.2f}s")
                    return [verdict.q]
                last = verdict.reason
                if verdict.reason is NO_IK:
                    # Out of arm.  This is what ends a walk in practice, and it ends it
                    # for a better reason than a step count could: the robot cannot go
                    # further this way, so neither can the search.  Everything past here
                    # is unreachable too, and testing it would only cost solves.
                    last = "the arm runs out of reach"
                    break
        except Exception as exc:                # a method must not take the transit with it
            self.log(f"      fallback pose from {method.name}: failed after "
                     f"{time.perf_counter() - started:.2f}s -- "
                     f"{type(exc).__name__}: {exc}")
            return []
        # Nothing found.  ``tried`` separates "every candidate was rejected" from "the
        # method had nothing to offer", which are different problems: the first is a cell
        # too tight for the fallback distance, the second is a method that could not get
        # started -- no retreat axis, or a neutral pose the tool is already standing at.
        if tried:
            self.log(f"      fallback pose from {method.name}: none in {tried} step"
                     f"{'' if tried == 1 else 's'} ({last}), in "
                     f"{time.perf_counter() - started:.2f}s")
        else:
            self.log(f"      fallback pose from {method.name}: no candidates to try, "
                     f"in {time.perf_counter() - started:.2f}s")
        return []
