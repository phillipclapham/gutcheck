"""Test-run guard: no test may resolve the player's real data dir.

Set before any test module imports gutcheck, so it holds whatever order the tests run in. A
throwaway HOME also covers the default branch: resolving ~/.gutcheck can copy ~/.flowpoker, and
that must never happen against a real home from a test.
"""

import os
import tempfile

_root = tempfile.mkdtemp(prefix="gutcheck-tests-")
os.environ["HOME"] = os.path.join(_root, "home")
os.makedirs(os.environ["HOME"])
os.environ["GUTCHECK_DATA_DIR"] = os.path.join(_root, "data")   # unconditional: an exported real dir must not survive
os.environ.pop("FLOWPOKER_DATA_DIR", None)
