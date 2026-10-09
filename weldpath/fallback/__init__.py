"""Finding poses to route a difficult transit through.

Kept apart from the planner on purpose.  What counts as a neutral pose, what counts as a
valid fallback point and where the search looks are all expected to change, and none of
them is a planning question -- the planner's interest ends at "here is a pose, try routing
through it".

  :mod:`neutral`   what a neutral pose is; edit ``CONDITIONS`` and nothing else
  :mod:`retreat`   which way is "back" from a locator, from the gun's own bulk
  :mod:`validity`  whether a candidate point is somewhere the robot can be parked
  :mod:`methods`   where each end's fallback is looked for; edit ``ENDS``
  :mod:`finder`    walks each end's methods, times them, reports what it found
"""
from .finder import Fallback, FallbackFinder

__all__ = ["Fallback", "FallbackFinder"]
