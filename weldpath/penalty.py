"""Cost multiplier discouraging routes that run close to the panels and tooling.

A collision check is a hard yes/no: a move that clears the panel by half a millimetre is
just as legal as one that clears it by half a metre.  That is fine for safety and poor for
route quality, because the shortest path almost always hugs the obstacle.  This turns
proximity into cost instead, so a route that skims the tooling has to *earn* its place by
being much shorter rather than marginally shorter.

The multiplier is applied to time, not distance -- "a second spent here counts as N
seconds" -- which is why it composes with the travel metric the shortcut pass already
uses.  Between ``max_mm`` and ``min_mm`` it follows ``x**2`` on ``[0, 1]``, stretched to
fit, so cost climbs slowly at first and sharply as the gun closes on the part::

    u = (max_mm - d) / (max_mm - min_mm)     clamped to [0, 1]
    factor = 1 + (multiplier - 1) * u**2

At ``max_mm`` and beyond the factor is exactly 1, so open space is unaffected; at
``min_mm`` and below it saturates at ``multiplier``.  Saturating rather than continuing to
climb matters: without it, the pass would spend its whole budget fighting over the last
millimetre of a clearance that is already as bad as it is allowed to get.

This is a *soft* preference and is deliberately not a safety mechanism -- what the robot
is actually forbidden to do is set by the collision margins in :mod:`weldpath.cell`.
"""
from __future__ import annotations


class ClearancePenalty:
    """Maps a clearance in millimetres to a cost multiplier."""

    def __init__(self, max_mm: float = 50.0, min_mm: float = 3.0,
                 multiplier: float = 50.0, cutoff_mm: float = 0.0,
                 enabled: bool = True):
        if max_mm <= min_mm:
            raise ValueError(
                f"clearance penalty needs max ({max_mm:g} mm) above min ({min_mm:g} mm)")
        if multiplier < 1.0:
            raise ValueError(
                f"clearance penalty multiplier must be at least 1, got {multiplier:g}")
        self.max_mm = float(max_mm)
        self.min_mm = float(min_mm)
        self.multiplier = float(multiplier)
        # A cutoff below max_mm truncates the shallow end of the curve without reshaping
        # what remains, so the proximity query only has to look out that far.  Values
        # outside (0, max_mm) mean "no cutoff".
        self.cutoff_mm = (float(cutoff_mm) if 0.0 < cutoff_mm < self.max_mm
                          else self.max_mm)
        self.enabled = bool(enabled) and self.multiplier > 1.0

    @property
    def probe_mm(self) -> float:
        """How far the proximity query has to see.  Nothing beyond this changes cost."""
        return self.cutoff_mm

    def factor(self, distance_mm: float) -> float:
        """Cost multiplier at a given clearance.  1.0 means no penalty."""
        if not self.enabled or distance_mm >= self.cutoff_mm:
            return 1.0
        u = (self.max_mm - distance_mm) / (self.max_mm - self.min_mm)
        u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
        return 1.0 + (self.multiplier - 1.0) * u * u

    def describe(self) -> str:
        if not self.enabled:
            return "clearance penalty: off"
        out = (f"clearance penalty: 1x at {self.max_mm:g} mm rising as x^2 to "
               f"{self.multiplier:g}x at {self.min_mm:g} mm and below")
        if self.cutoff_mm < self.max_mm:
            out += (f"; ignored beyond {self.cutoff_mm:g} mm, so the factor steps "
                    f"straight to {self.factor(self.cutoff_mm - 1e-9):.1f}x there")
        return out
