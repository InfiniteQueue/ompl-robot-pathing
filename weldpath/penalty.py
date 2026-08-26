"""Cost multiplier discouraging routes that run close to the panels and tooling.

A collision check is a hard yes/no: a move that clears the panel by half a millimetre is
just as legal as one that clears it by half a metre.  That is fine for safety and poor for
route quality, because the shortest path almost always hugs the obstacle.  This turns
proximity into cost instead, so a route that skims the tooling has to *earn* its place by
being much shorter rather than marginally shorter.

The multiplier is applied to time, not distance -- "a second spent here counts as N
seconds" -- which is why it composes with the travel metric the shortcut pass already
uses.  Between ``max_mm`` and ``min_mm`` it follows a power curve on ``[0, 1]``,
stretched to fit::

    u = (max_mm - d) / (max_mm - min_mm)     clamped to [0, 1]
    factor = 1 + (multiplier - 1) * u**exponent

At ``max_mm`` and beyond the factor is exactly 1, so open space is unaffected; at
``min_mm`` and below it saturates at ``multiplier``.  Both ends are fixed whatever the
exponent is -- it changes only how the climb is distributed between them, so the peak
penalty is the one thing that stays put while the shape is tuned.

The exponent is what decides where the curve can tell one clearance from another, and the
whole span is only so much resolution to spend.  Above 1 it is spent near ``min_mm``: at 2
the cost climbs slowly at first and sharply as the gun closes on the part, and higher
still concentrates almost all of the difference into the last few millimetres, which is
what a wide ``max_mm`` needs if "very close" is not to cost much the same as "close".
Below 1 the curve is concave and spends its resolution at the open end instead, so a route
is pushed away from the part early and then hardly cares how close it finally comes.  Saturating rather than continuing to
climb matters: without it, the pass would spend its whole budget fighting over the last
millimetre of a clearance that is already as bad as it is allowed to get.

This is a *soft* preference and is deliberately not a safety mechanism -- what the robot
is actually forbidden to do is set by the collision margins in :mod:`weldpath.cell`.
"""
from __future__ import annotations

import math


class ClearancePenalty:
    """Maps a clearance in millimetres to a cost multiplier."""

    def __init__(self, max_mm: float = 50.0, min_mm: float = 3.0,
                 multiplier: float = 50.0, cutoff_mm: float = 0.0,
                 exponent: float = 2.0, enabled: bool = True):
        if max_mm <= min_mm:
            raise ValueError(
                f"clearance penalty needs max ({max_mm:g} mm) above min ({min_mm:g} mm)")
        if multiplier < 1.0:
            raise ValueError(
                f"clearance penalty multiplier must be at least 1, got {multiplier:g}")
        if not exponent > 0.0:
            # 0 would make every clearance below max_mm cost the full multiplier, turning
            # the soft preference into a second collision margin; negative diverges.
            raise ValueError(
                f"clearance penalty exponent must be above 0, got {exponent:g}")
        self.max_mm = float(max_mm)
        self.min_mm = float(min_mm)
        self.multiplier = float(multiplier)
        self.exponent = float(exponent)
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
        return 1.0 + (self.multiplier - 1.0) * u ** self.exponent

    def describe(self) -> str:
        if not self.enabled:
            return "clearance penalty: off"
        out = (f"clearance penalty: 1x at {self.max_mm:g} mm rising as "
               f"x^{self.exponent:g} to {self.multiplier:g}x at {self.min_mm:g} mm "
               f"and below")
        if self.cutoff_mm < self.max_mm:
            out += (f"; ignored beyond {self.cutoff_mm:g} mm, so the factor steps "
                    f"straight to {self.factor(self.cutoff_mm - 1e-9):.1f}x there")
        return out

class SteppedPenalty:
    """Maps a clearance to a cost multiplier along a decaying exponential.

    An alternative to :class:`ClearancePenalty`, specified the way a process engineer would
    rather think about it: not "what shape between these two clearances" but "how fast does
    the penalty fall off as the gun backs away".  That rate is given as a **step** -- every
    ``step_mm`` of extra clearance multiplies the penalty by ``step_factor`` -- so the curve
    is fixed by a rate rather than by an exponent whose meaning depends on the span it is
    stretched over::

        g(x) = K * step_factor ** (x / step_mm) + e        x >= 0
        g(x) = g(0) + g'(0) * x                            x <  0
        factor = 1 + max(g(x), 0)

    which is ``1 + a * b**(-x + c) + e`` with ``b = step_factor ** (-1 / step_mm)``.

    ``K`` and ``e`` are not given.  They are solved from the two ends the user does supply:
    the multiplier at zero clearance and the clearance at which the penalty is spent::

        K = (multiplier - 1) / (1 - step_factor ** (zero_mm / step_mm))
        e = -K * step_factor ** (zero_mm / step_mm)

    **The offset is what lets the curve reach zero, and it is why the step rule holds of
    the exponential term rather than of the multiplier itself.**  A quantity that is
    multiplied by a constant factor every step never reaches zero -- it only halves and
    halves again -- so "falls off by this factor per step" and "costs nothing at all beyond
    this clearance" cannot both be exactly true.  ``e`` buys the second at the price of the
    first: the decay between two neighbouring steps is very close to ``step_factor`` while
    the penalty is large, and drifts as it approaches zero.  The alternative -- an exact
    step rule truncated at ``zero_mm`` -- leaves the factor stepping discontinuously there,
    which the passes would read as a cost cliff.

    Below zero clearance the curve is a straight line: same value and same gradient at the
    origin, continuing at that slope.  The gun is inside the part there, which is a place
    the *collision* margins are responsible for; all this has to do is keep getting worse
    without the exponential's curvature turning a deep penetration into an astronomical
    number that swamps every other term in a route's cost.
    """

    def __init__(self, multiplier: float = 8.0, zero_mm: float = 200.0,
                 step_mm: float = 25.0, step_factor: float = 0.7,
                 enabled: bool = True):
        if multiplier < 1.0:
            raise ValueError(
                f"stepped penalty multiplier must be at least 1, got {multiplier:g}")
        if zero_mm <= 0.0:
            raise ValueError(
                f"stepped penalty needs a positive zero clearance, got {zero_mm:g} mm")
        if step_mm <= 0.0:
            raise ValueError(
                f"stepped penalty needs a positive step length, got {step_mm:g} mm")
        if not 0.0 < step_factor < 1.0:
            # At 1 the penalty never decays and every clearance costs the same; above it the
            # penalty grows with clearance, which is the wrong way round.
            raise ValueError(
                f"stepped penalty step factor must be between 0 and 1, got "
                f"{step_factor:g}")
        self.multiplier = float(multiplier)
        self.zero_mm = float(zero_mm)
        self.step_mm = float(step_mm)
        self.step_factor = float(step_factor)
        self.enabled = bool(enabled) and self.multiplier > 1.0

        self._decay = math.log(self.step_factor) / self.step_mm       # < 0
        spent = math.exp(self._decay * self.zero_mm)                  # factor at zero_mm
        if spent >= 1.0 - 1e-12:
            raise ValueError(
                f"stepped penalty decays too slowly to reach zero at {self.zero_mm:g} mm: "
                f"raise --stepped-penalty-step-factor's bite or lower the zero clearance")
        self.k = (self.multiplier - 1.0) / (1.0 - spent)
        self.e = -self.k * spent
        # Gradient at the origin, which the sub-zero line inherits.  Negative: the penalty
        # falls as the gun backs away, so the line rises as it presses in.
        self.slope = self.k * self._decay

    @property
    def probe_mm(self) -> float:
        """How far the proximity query has to see.  Nothing beyond this changes cost."""
        return self.zero_mm

    def excess(self, distance_mm: float) -> float:
        """The penalty above 1, before the floor is applied."""
        x = float(distance_mm)
        if x < 0.0:
            return (self.multiplier - 1.0) + self.slope * x
        return self.k * math.exp(self._decay * x) + self.e

    def factor(self, distance_mm: float) -> float:
        """Cost multiplier at a given clearance.  1.0 means no penalty."""
        if not self.enabled or distance_mm >= self.zero_mm:
            return 1.0
        over = self.excess(distance_mm)
        return 1.0 + over if over > 0.0 else 1.0

    def describe(self) -> str:
        if not self.enabled:
            return "clearance penalty: off"
        marks = [m for m in (0.0, self.step_mm, 2.0 * self.step_mm) if m < self.zero_mm]
        walk = ", ".join(f"{self.factor(m):.2f}x at {m:g} mm" for m in marks)
        return (f"clearance penalty (stepped): {self.multiplier:g}x at touching, falling "
                f"by {self.step_factor:g} every {self.step_mm:g} mm to 1x at "
                f"{self.zero_mm:g} mm -- {walk}; below 0 mm it continues as a straight "
                f"line at {-self.slope:.3f}x per mm")
