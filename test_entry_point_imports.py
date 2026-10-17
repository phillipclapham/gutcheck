"""Guard the installed entry-point path — the `gutcheck/` package refactor (2026-06-30).

BEFORE the refactor this file guarded the flat-layout footgun: every top-level module had to be
hand-listed in pyproject `py-modules` or the strict editable finder wouldn't expose it and the
INSTALLED entry point crashed while every cwd-run test passed (it bit `gutcheck-mcp`/doctrine
2026-06-22 AND `gutcheck`/settings 2026-06-23). The package refactor RETIRES that footgun —
`packages = ["gutcheck"]` ships every submodule automatically.

What still needs guarding, and what these tests now cover:
  1. The packaging config stays coherent — package declared, entry points point INTO the package,
     the bundled data files are declared as package-data, and the retired `py-modules` list is gone.
  2. The entry-point modules import cleanly from a cwd OUTSIDE the repo, so resolution relies on the
     INSTALLED finder (not cwd on sys.path). Importing `gutcheck.server` + `gutcheck.mcp_server`
     transitively imports every app module (server→game→engine/archetypes/…, coach_daemon→coach/…),
     so this also proves every package-relative `from . import` resolves. If it fails after a move,
     the fix is `pip install -e .` (refresh the editable finder).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

_REPO = Path(__file__).parent


class TestPackagingConfig(unittest.TestCase):
    """STRUCTURAL guard on pyproject: the package + entry points + package-data stay coherent, and
    the retired flat-layout `py-modules` list does not creep back."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = tomllib.loads((_REPO / "pyproject.toml").read_text())

    def test_gutcheck_package_discovered(self):
        # Discovery is auto via [tool.setuptools.packages.find] include=["gutcheck*"] — this ships
        # the package AND future-proofs against a dropped subpackage (e.g. gutcheck/providers/).
        find = self.cfg["tool"]["setuptools"]["packages"]["find"]
        self.assertIn("gutcheck*", find.get("include", []),
                      "packages.find must include 'gutcheck*' so the package + any future "
                      "subpackage ships automatically.")

    def test_version_single_sourced(self):
        # pyproject [project] version and gutcheck.__version__ are written in two places — keep
        # them in agreement so the next bump can't drift them apart.
        import gutcheck
        self.assertEqual(self.cfg["project"]["version"], gutcheck.__version__,
                         "pyproject [project] version != gutcheck.__version__ — sync them.")

    def test_py_modules_retired(self):
        # The flat-layout footgun. If this key is back, the hand-registration hazard is back.
        self.assertNotIn("py-modules", self.cfg["tool"]["setuptools"],
                         "`py-modules` is retired by the gutcheck/ package — do not reintroduce it.")

    def test_entry_points_target_the_package(self):
        scripts = self.cfg["project"]["scripts"]
        self.assertEqual(scripts.get("gutcheck"), "gutcheck.server:main")
        self.assertEqual(scripts.get("gutcheck-mcp"), "gutcheck.mcp_server:main")

    def test_bundled_data_declared_as_package_data(self):
        # A non-editable wheel carries static/ + doctrine.md ONLY if they're declared here.
        pdata = self.cfg["tool"]["setuptools"]["package-data"]["gutcheck"]
        joined = " ".join(pdata)
        self.assertIn("doctrine.md", joined,
                      "doctrine.md must be package-data or the wheel drops it.")
        self.assertTrue(any("static" in p for p in pdata),
                        "static/ must be package-data or the wheel drops the table UI.")

    def test_bundled_data_files_exist_in_the_package(self):
        self.assertTrue((_REPO / "gutcheck" / "doctrine.md").is_file())
        self.assertTrue((_REPO / "gutcheck" / "static" / "index.html").is_file())


class TestEntryPointImportsWithoutCwd(unittest.TestCase):
    def _import_from_outside_repo(self, module: str) -> subprocess.CompletedProcess:
        # cwd OUTSIDE the repo: if `module` isn't exposed by the editable finder (a stale install),
        # this fails exactly the way the gutcheck-mcp entry point did at launch — the cwd no longer
        # masks a module that's importable only because the repo happens to be on sys.path.
        with tempfile.TemporaryDirectory() as outside:
            return subprocess.run([sys.executable, "-c", f"import {module}"],
                                  cwd=outside, capture_output=True, text=True)

    def test_package_importable_via_installed_finder(self):
        r = self._import_from_outside_repo("gutcheck")
        self.assertEqual(r.returncode, 0,
                         "`import gutcheck` failed from outside the repo — the editable install is "
                         f"stale; re-run `pip install -e .`:\n{r.stderr}")

    def test_submodule_importable_via_installed_finder(self):
        # `import gutcheck` alone is a weak guard (the __init__ imports nothing). Import a leaf
        # SUBMODULE too, so a discovery config that somehow shipped the package but not its modules
        # would be caught here rather than only via the transitive entry-point imports below.
        r = self._import_from_outside_repo("gutcheck.game")
        self.assertEqual(r.returncode, 0,
                         "`import gutcheck.game` failed from outside the repo — packages.find may "
                         f"not be shipping submodules; re-run `pip install -e .`:\n{r.stderr}")

    def test_server_entrypoint_imports_clean(self):
        # `gutcheck.server` is the `gutcheck` entry point; its import chain reaches game → engine/
        # archetypes/equity/ranges/history/paths and coach/coach_daemon/keystore/leakstore/settings/
        # seam — so this proves every package-relative import in the app graph resolves. main() is
        # gated on __main__, so importing the module is side-effect-free.
        r = self._import_from_outside_repo("gutcheck.server")
        self.assertEqual(r.returncode, 0,
                         "`import gutcheck.server` (the gutcheck entry point) failed from outside "
                         f"the repo — stale install or a broken relative import:\n{r.stderr}")

    def test_mcp_server_entrypoint_imports_clean(self):
        # the gutcheck-mcp entry point; transitively imports doctrine (the exact 2026-06-22 crash).
        r = self._import_from_outside_repo("gutcheck.mcp_server")
        self.assertEqual(r.returncode, 0,
                         "`import gutcheck.mcp_server` (the gutcheck-mcp entry point) failed from "
                         f"outside the repo:\n{r.stderr}")

    def test_coach_daemon_module_run_path_imports_clean(self):
        # the opt-in launchd agent runs `python -m gutcheck.coach_daemon run`; guard that import path.
        r = self._import_from_outside_repo("gutcheck.coach_daemon")
        self.assertEqual(r.returncode, 0,
                         "`import gutcheck.coach_daemon` (the `-m gutcheck.coach_daemon` launchd "
                         f"path) failed from outside the repo:\n{r.stderr}")


if __name__ == "__main__":
    unittest.main()
