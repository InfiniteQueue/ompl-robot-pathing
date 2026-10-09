"""Finding the poses a difficult transit detours through.

**One fallback per end, and the detour runs through both.**  The route attempted is
``w1 -> f1 -> f2 -> w2``: ``f1`` is the start locator's own fallback and ``f2`` the end
locator's, so each is a place to stand just clear of the pocket that end sits in.  Each is
looked for by its end's own two methods in turn -- out along the locator's retreat axis
first, and where that yields nothing, out towards that end's nearest neutral pose; see
:mod:`weldpath.fallback.methods`.  The second is the *next closest* answer available for
that end rather than a different end's answer being borrowed.

This replaced a list of interchangeable candidates for a single via.  Three methods were
walked in preference order -- retreat from the start, retreat from the end, a line out of
the midpoint -- and whichever poses they produced were offered to the planner as
alternatives, one detour attempted through each in turn until one worked.  Two things were
wrong with it.  The poses were not alternatives: "retreat from the end locator" is the end
locator's extraction, and offering it as the remedy for a *start* locator that could not
be backed out answers a different question.  And one via means one leg out of each pocket,
so the transit still had to get from the start locator's pocket all the way to the end
locator's in a single move -- the move that had just failed directly.  Two vias put a leg
in open air between them, which is the leg that was missing.

``max_poses`` is how many ends are given one: 2 is both and the default, 1 looks for the
start's alone, 0 looks for none at all and leaves a transit that cannot be flown directly
to fail directly.  The cap is applied *here* rather than by slicing the list the planner
gets, because a method costs real time whether or not its pose is used -- it solves
inverse kinematics at every step it takes -- so a cap that only hid the pose afterwards
would save none of it.

This is per transit, not per run.  Both ends of the move enter every method and the
validity rule, so a pose found for one transit says nothing about the next; the previous
scheme, which took the first via in the study and used it everywhere, was cheap precisely
because it ignored all of that.

A walk ends when the arm runs out of reach.  A step with no inverse-kinematics solution
stops that method: everything further along the same line is unreachable too, so there is
nothing left to test.  ``methods.MAX_STEPS`` is only a backstop for a direction that stays
reachable indefinitely, and is set high enough not to bind on a real arm.

It is also lazy.  Most transits solve directly and never ask, and a search that walks to
the end of the robot's reach is not something to spend on every pair for the sake of the
few that need it.  ``FallbackFinder.pair`` is called only once the direct attempts have
failed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import methods, neutral
from .validity import NO_IK, Validator

# Two configurations this close in every joint are the same pose.  Reached where both ends
# fall through to their neutral walks and those walks converge, which is possible when the
# two locators sit almost on top of each other -- a transit between two welds a few
# millimetres apart.  A detour through the same pose twice is a leg that does not move.
SAME_POSE_RAD = 1e-6


@dataclass
class Fallback:
    """One end's fallback pose, and where it came from.

    ``end`` is the end of the transit this belongs to, and it is what the planner names
    the pose by -- in the log, in a failure message, and when it screens the gun openings
    the detour may be planned at.
    """
    end: str
    q: np.ndarray
    found_by: str
    clearance_mm: float


class FallbackFinder:
    """Builds fallback poses for one cell, on demand, transit by transit."""

    def __init__(self, cell, man, *, fallback_mm: float, step_mm: float,
                 max_poses: int = len(methods.ENDS), log=print):
        self.cell = cell
        self.man = man
        self.fallback_mm = float(fallback_mm)
        self.step_mm = float(step_mm)
        # Clamped rather than rejected.  There are two ends and each has exactly one
        # fallback, so a higher figure is the same request as 2; a command line carried
        # over from the arrangement that walked three methods should keep working, and
        # ``announce`` says what the number was read as.
        self.asked_poses = max(0, int(max_poses))
        self.max_poses = min(self.asked_poses, len(methods.ENDS))
        self.log = log

    def announce(self) -> None:
        """Say what the search will do, once, before any transit needs it."""
        if not self.max_poses:
            self.log("  fallback poses: none will be looked for, so a transit with no "
                     "direct route fails rather than detouring")
            return
        if self.asked_poses > self.max_poses:
            self.log(f"  fallback poses: {self.asked_poses} asked for, but a transit has "
                     f"{len(methods.ENDS)} ends and each has one fallback, so "
                     f"{self.max_poses} is every pose there is")
        named = [e.name for e in methods.ENDS[:self.max_poses]]
        ends = " and ".join([", ".join(named[:-1]), named[-1]] if len(named) > 2
                            else named)
        self.log(f"  fallback poses: one {'each ' if len(named) > 1 else ''}for the "
                 f"{ends} locator{'' if len(named) == 1 else 's'}, keeping "
                 f"{self.fallback_mm:g} mm off the parts, stepping {self.step_mm:g} mm "
                 f"at a time until the arm runs out of reach")
        self.log("  each end is backed out along its own retreat axis first, then towards "
                 "its own nearest neutral pose if that finds nothing")
        self.log(f"  a neutral pose is {neutral.describe()}")

    def pair(self, qa: np.ndarray, qb: np.ndarray,
             pose_a: np.ndarray, pose_b: np.ndarray) -> list[Fallback]:
        """The fallback poses for the transit from ``qa`` to ``qb``, in route order.

        Up to one per end, and an end that yields nothing simply contributes nothing: a
        detour through one pose is still a detour, and refusing to attempt one because the
        other end could not be backed out would throw away the half that worked.
        """
        if not self.max_poses:
            # Nothing is searched and nothing is logged per transit.  The count was
            # announced once at the start of the run, and repeating it on every transit
            # that fails directly would say the same thing dozens of times.
            return []
        ctx = methods.Context(cell=self.cell, man=self.man,
                              qa=np.asarray(qa, dtype=float),
                              qb=np.asarray(qb, dtype=float),
                              pose_a=np.asarray(pose_a, dtype=float),
                              pose_b=np.asarray(pose_b, dtype=float),
                              step_mm=self.step_mm)
        validator = Validator(self.cell, qa, qb, self.fallback_mm)
        out: list[Fallback] = []
        for end in methods.ENDS[:self.max_poses]:
            got = self._for_end(end, ctx, validator)
            if got is None:
                continue
            same = next((f for f in out if self._same(f.q, got.q)), None)
            if same is not None:
                # Both walks arrived at one pose.  Kept once: a leg from a pose to itself
                # is not a move, and the detour is simply the one-via shape instead.
                self.log(f"      the {end.name} locator's fallback is the same pose as "
                         f"the {same.end} locator's, so the detour runs through it once")
                continue
            out.append(got)
        if not out:
            self.log("      no fallback pose was found for either end; this transit will "
                     "be attempted directly or not at all")
        return out

    def _for_end(self, end: methods.End, ctx: methods.Context,
                 validator: Validator) -> Fallback | None:
        """One end's fallback: its retreat axis first, then its walk towards neutral."""
        for method in (end.primary, end.secondary):
            got = self._run(end, method, ctx, validator)
            if got is not None:
                return got
        self.log(f"      no fallback pose for the {end.name} locator by either method")
        return None

    def _run(self, end: methods.End, method: methods.Method, ctx: methods.Context,
             validator: Validator) -> Fallback | None:
        """One method, timed and reported.  ``None`` where it found nothing."""
        started = time.perf_counter()
        tried = 0
        last = ""
        try:
            for pose in method.generate(ctx):
                tried += 1
                verdict = validator.check(pose)
                if verdict.valid:
                    self.log(f"      {end.name} locator's fallback pose from "
                             f"{method.name}: found after {tried} "
                             f"step{'' if tried == 1 else 's'}, "
                             f"{verdict.clearance_mm:.0f} mm clear, in "
                             f"{time.perf_counter() - started:.2f}s")
                    return Fallback(end.name, verdict.q, method.name,
                                    verdict.clearance_mm)
                last = verdict.reason
                if verdict.reason is NO_IK:
                    # Out of arm.  This is what ends a walk in practice, and it ends it
                    # for a better reason than a step count could: the robot cannot go
                    # further this way, so neither can the search.  Everything past here
                    # is unreachable too, and testing it would only cost solves.
                    last = "the arm runs out of reach"
                    break
        except Exception as exc:                # a method must not take the transit with it
            self.log(f"      {end.name} locator's fallback pose from {method.name}: "
                     f"failed after {time.perf_counter() - started:.2f}s -- "
                     f"{type(exc).__name__}: {exc}")
            return None
        # Nothing found.  ``tried`` separates "every candidate was rejected" from "the
        # method had nothing to offer", which are different problems: the first is a cell
        # too tight for the fallback distance, the second is a method that could not get
        # started -- no retreat axis, or a neutral pose the tool is already standing at.
        if tried:
            self.log(f"      {end.name} locator's fallback pose from {method.name}: none "
                     f"in {tried} step{'' if tried == 1 else 's'} ({last}), in "
                     f"{time.perf_counter() - started:.2f}s")
        else:
            self.log(f"      {end.name} locator's fallback pose from {method.name}: no "
                     f"candidates to try, in {time.perf_counter() - started:.2f}s")
        return None

    @staticmethod
    def _same(a: np.ndarray, b: np.ndarray) -> bool:
        """Whether two configurations are the same pose as far as a detour is concerned."""
        return bool(np.max(np.abs(np.asarray(a, dtype=float)
                                  - np.asarray(b, dtype=float))) <= SAME_POSE_RAD)
