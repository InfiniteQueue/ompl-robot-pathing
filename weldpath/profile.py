"""Trapezoidal velocity profile used to schedule waypoints in time.

A move ramps velocity up, holds it, then ramps it down, so displacement follows an S
curve. ``blend`` selects the shape:

* ``0`` -- flat velocity: no ramps, constant speed for the whole move.
* ``1`` -- bang-bang: no cruise at all, accelerating then decelerating throughout, which
  makes the velocity profile a triangle.

``blend`` is the share of the move spent accelerating or decelerating, so each individual
ramp occupies ``blend / 2`` of the total time and the cruise occupies ``1 - blend``.

Normalising time to ``tau = t / T`` and velocity to its peak, with ``a = blend / 2``::

    v(tau) = tau / a           0     <= tau <= a        (ramp up)
             1                 a     <= tau <= 1 - a    (cruise)
             (1 - tau) / a     1 - a <= tau <= 1        (ramp down)

Integrating gives a mean velocity of ``1 - a`` times the peak, which is why adding ramps
makes a move take longer for the same commanded speed. The displacement fraction is::

    sigma(tau) = tau^2 / A                 tau <= a
                 (tau - a/2) / (1 - a)     a <= tau <= 1 - a
                 1 - (1 - tau)^2 / A       tau >= 1 - a

with ``A = 2a(1 - a)``. Scheduling a waypoint means going the other way -- the waypoint
sits at a known fraction of the path, and we need the time at which the profile reaches
it -- so :meth:`MotionProfile.time_fraction` inverts that piecewise.
"""
from __future__ import annotations

import math


class MotionProfile:
    """Maps position along a move to time along a trapezoidal velocity profile."""

    def __init__(self, blend: float = 0.5):
        if not 0.0 <= blend <= 1.0:
            raise ValueError(f"profile blend must be within [0, 1], got {blend}")
        self.blend = float(blend)
        self.ramp = self.blend / 2.0          # each ramp, as a fraction of total time

    def __repr__(self) -> str:
        return (f"MotionProfile(blend={self.blend:g}, ramp={self.ramp:g}, "
                f"duty={self.duty:g})")

    @property
    def duty(self) -> float:
        """Mean velocity as a fraction of peak velocity."""
        return 1.0 - self.ramp

    def duration(self, distance: float, peak_speed: float) -> float:
        """Time to cover ``distance`` when ``peak_speed`` is the commanded maximum."""
        if distance <= 0.0 or peak_speed <= 0.0:
            return 0.0
        return distance / (peak_speed * self.duty)

    def time_fraction(self, travelled: float) -> float:
        """Fraction of the total time at which ``travelled`` of the distance is covered.

        ``travelled`` is a fraction in [0, 1]; the result is a fraction in [0, 1].
        """
        sigma = min(1.0, max(0.0, float(travelled)))
        a = self.ramp
        if a <= 0.0:                           # flat velocity: time tracks distance
            return sigma
        area = 2.0 * a * (1.0 - a)
        sigma_ramp = a * a / area              # distance covered by the end of the ramp
        if sigma <= sigma_ramp:
            return math.sqrt(sigma * area)
        if sigma >= 1.0 - sigma_ramp:
            return 1.0 - math.sqrt((1.0 - sigma) * area)
        return sigma * (1.0 - a) + a / 2.0

    def schedule(self, distances: list[float], peak_speed: float) -> list[float]:
        """Times for a run of waypoints separated by ``distances``.

        ``distances`` holds the step between consecutive waypoints, so the result is one
        longer than the input: the first waypoint is at time zero. The profile spans the
        whole run, starting and finishing at rest.
        """
        total = float(sum(distances))
        if total <= 0.0:
            return [0.0] * (len(distances) + 1)
        span = self.duration(total, peak_speed)
        times = [0.0]
        travelled = 0.0
        for step in distances:
            travelled += step
            times.append(span * self.time_fraction(travelled / total))
        return times
