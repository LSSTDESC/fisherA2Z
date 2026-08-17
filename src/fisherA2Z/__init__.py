from importlib.metadata import PackageNotFoundError, version as _version
import numpy as np

__all__ = ["cli"]

try:
    # Read the version recorded at install time rather than hard-coding it, so
    # it cannot drift from the git tag the release was built from.
    __version__ = _version("fisher-a2z")
except PackageNotFoundError:  # not installed, e.g. running from a source tree
    __version__ = "unknown"

from fisherA2Z import *

# monkey patch np.trapz
try:
    _ = np.trapz
except AttributeError:
    setattr(np, 'trapz', np.trapezoid)
