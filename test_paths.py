"""Tests for the data-dir resolution the whole packaging slice rests on.

The other suites isolate by passing EXPLICIT tmp paths to Game()/coach, so they never exercise
the GUTCHECK_DATA_DIR default-resolution that production relies on. These do — including the
load-bearing property: setting the env var AFTER the modules are imported still redirects a bare
Game() (lazy, per-call resolution), so no test/smoke can corrupt the real bankroll.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from gutcheck import coach
from gutcheck import history
from gutcheck import paths
from gutcheck.game import Game



import contextlib
import unittest.mock


@contextlib.contextmanager
def _isolated_home(home):
    """HOME -> ``home``, both data-dir env vars unset, the per-process migration latch reset."""
    saved = {k: os.environ.get(k) for k in ("HOME", paths.ENV_VAR, paths.LEGACY_ENV_VAR)}
    latch = paths._migrate_checked
    os.environ["HOME"] = home
    os.environ.pop(paths.ENV_VAR, None)
    os.environ.pop(paths.LEGACY_ENV_VAR, None)
    paths._migrate_checked = False
    try:
        yield Path(home)
    finally:
        paths._migrate_checked = latch
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class LegacyMigration(unittest.TestCase):
    """The rename from flowpoker: ~/.flowpoker is COPIED to ~/.gutcheck once, never moved."""

    def _legacy(self, home: Path) -> Path:
        old = home / ".flowpoker"
        (old / "sessions").mkdir(parents=True)
        (old / "career.json").write_text('{"bankroll": 3662}')
        (old / "sessions" / "s1.json").write_text("{}")
        (old / "secrets.json").write_text("{}")
        os.chmod(old / "secrets.json", 0o600)
        return old

    def test_copies_legacy_and_keeps_the_original(self):
        with _isolated_home(tempfile.mkdtemp()) as home:
            old = self._legacy(home)
            new = paths.data_dir()
            self.assertEqual(new, home / ".gutcheck")
            self.assertEqual((new / "career.json").read_text(), '{"bankroll": 3662}')
            self.assertTrue((new / "sessions" / "s1.json").exists())
            self.assertEqual(os.stat(new / "secrets.json").st_mode & 0o777, 0o600)
            self.assertEqual((old / "career.json").read_text(), '{"bankroll": 3662}')   # backup intact
            self.assertEqual([x for x in home.iterdir() if "migrating" in x.name], [])  # no temp left

    def test_existing_new_dir_is_never_overwritten(self):
        with _isolated_home(tempfile.mkdtemp()) as home:
            self._legacy(home)
            (home / ".gutcheck").mkdir()
            (home / ".gutcheck" / "career.json").write_text('{"bankroll": 4000}')
            paths.data_dir()
            self.assertEqual((home / ".gutcheck" / "career.json").read_text(), '{"bankroll": 4000}')

    def test_no_migration_under_an_override(self):
        with _isolated_home(tempfile.mkdtemp()) as home:
            self._legacy(home)
            scratch = tempfile.mkdtemp()
            os.environ[paths.ENV_VAR] = scratch
            self.assertEqual(paths.data_dir(), Path(scratch))
            self.assertFalse((home / ".gutcheck").exists())

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "chmod 0 does not block root")
    def test_a_failed_copy_fails_closed_and_retries(self):
        # an unreadable file aborts the copy: no empty ~/.gutcheck may be left behind (it would skip
        # the migration on every later start), the old data is untouched, and the next start retries
        with _isolated_home(tempfile.mkdtemp()) as home:
            old = self._legacy(home)
            os.chmod(old / "career.json", 0)
            try:
                with self.assertRaises(paths.LegacyMigrationError):
                    paths.data_dir()
                self.assertFalse((home / ".gutcheck").exists())
                with self.assertRaises(paths.LegacyMigrationError):
                    paths.ensure_data_dir()                            # still fails closed, still no dir
                self.assertFalse((home / ".gutcheck").exists())
                os.chmod(old / "career.json", 0o644)                   # cause fixed -> it copies
                self.assertEqual((paths.data_dir() / "career.json").read_text(), '{"bankroll": 3662}')
            finally:
                os.chmod(old / "career.json", 0o644)
            self.assertEqual([x for x in home.iterdir() if "migrating" in x.name], [])

    def test_a_concurrent_winner_is_success_not_an_error(self):
        # two processes starting together: the loser's rename fails because the winner's dir landed
        with _isolated_home(tempfile.mkdtemp()) as home:
            self._legacy(home)
            real_rename = os.rename

            def winner_lands_first(src, dst):
                real_rename(src, str(dst) + ".winner")
                real_rename(str(dst) + ".winner", dst)                 # the winner's copy is in place
                raise OSError(66, "Directory not empty")

            with unittest.mock.patch("gutcheck.paths.os.rename", winner_lands_first):
                self.assertEqual(paths.data_dir(), home / ".gutcheck")
            self.assertEqual((home / ".gutcheck" / "career.json").read_text(), '{"bankroll": 3662}')
            self.assertEqual([x for x in home.iterdir() if "migrating" in x.name], [])

    def test_legacy_env_var_still_redirects(self):
        with _isolated_home(tempfile.mkdtemp()) as home:
            scratch = tempfile.mkdtemp()
            os.environ[paths.LEGACY_ENV_VAR] = scratch
            self.assertEqual(paths.data_dir(), Path(scratch))
            os.environ[paths.ENV_VAR] = tempfile.mkdtemp()             # the new name wins when both set
            self.assertEqual(paths.data_dir(), Path(os.environ[paths.ENV_VAR]))
            self.assertFalse((home / ".gutcheck").exists())

class DataDirResolution(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get(paths.ENV_VAR)
        self.tmp = tempfile.mkdtemp()
        os.environ[paths.ENV_VAR] = self.tmp

    def tearDown(self):
        if self._saved is None:
            os.environ.pop(paths.ENV_VAR, None)
        else:
            os.environ[paths.ENV_VAR] = self._saved

    def test_env_override_redirects_every_path(self):
        d = Path(self.tmp)
        self.assertEqual(paths.data_dir(), d)
        self.assertEqual(paths.session_path(), d / "session.json")
        self.assertEqual(paths.career_path(), d / "career.json")
        self.assertEqual(paths.reports_dir(), d / "sessions")
        self.assertEqual(paths.coach_db(), d / "coach.db")

    def test_default_is_home_gutcheck_not_repo(self):
        # exercise the default branch under a throwaway HOME: resolving the default may run the
        # one-time ~/.flowpoker copy, which must never touch the real home from a test
        home = tempfile.mkdtemp()
        with _isolated_home(home):
            self.assertEqual(paths.data_dir(), Path(home) / ".gutcheck")
        # the original corruption class was repo-local state — the default must be off the repo
        self.assertFalse(str(paths.data_dir()).startswith(str(Path(__file__).parent)))

    def test_bare_game_redirects_after_import(self):
        # the load-bearing property: env set AFTER `from game import Game` (top of file) still
        # redirects, because resolution is per-call. A bare Game() lands under the tmp dir and
        # never touches ~/.gutcheck — this is the structural fix for the 2026-06-18 corruption.
        g = Game()
        g.sit_down(hero="T")
        self.assertEqual(g.path, Path(self.tmp) / "session.json")        # positive: the exact location
        self.assertEqual(g.career_path, Path(self.tmp) / "career.json")
        self.assertTrue((Path(self.tmp) / "career.json").exists())
        self.assertFalse(str(g.career_path).startswith(str(Path.home() / ".gutcheck")))  # ...and NOT the real home

    def test_ensure_data_dir_creates_dir_and_sessions(self):
        sub = Path(self.tmp) / "nested"
        os.environ[paths.ENV_VAR] = str(sub)
        out = paths.ensure_data_dir()
        self.assertEqual(out, sub)
        self.assertTrue((sub / "sessions").is_dir())

    def test_ensure_data_dir_rejects_a_file(self):
        f = Path(self.tmp) / "iam_a_file"
        f.write_text("x")
        os.environ[paths.ENV_VAR] = str(f)
        with self.assertRaises(NotADirectoryError):
            paths.ensure_data_dir()

    def test_relative_data_dir_is_rejected(self):
        os.environ[paths.ENV_VAR] = "relative/data"     # would diverge between server & daemon cwd
        with self.assertRaises(ValueError):
            paths.data_dir()


class ResetData(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get(paths.ENV_VAR)
        self.tmp = tempfile.mkdtemp()
        os.environ[paths.ENV_VAR] = self.tmp
        paths.ensure_data_dir()

    def tearDown(self):
        if self._saved is None:
            os.environ.pop(paths.ENV_VAR, None)
        else:
            os.environ[paths.ENV_VAR] = self._saved
        coach._INITIALIZED.discard(str(Path(self.tmp) / "coach.db"))
        history._INITIALIZED.discard(str(Path(self.tmp) / "history.db"))   # symmetric WAL-cache cleanup

    def test_reset_backs_up_career_and_clears_the_rest(self):
        g = Game()
        g.sit_down(hero="T")
        v = g.new_hand()
        while v.get("in_hand") and v.get("to_act") == "hero":
            v = g.act("call")
        g.get_up()                                     # writes a session report into sessions/
        coach.ask("decision", {"x": 1})                # creates coach.db at the default (tmp) path
        d = Path(self.tmp)
        # plant the REAL pid-suffixed temp shapes (<name>.<pid>.tmp) to prove the sweeps catch them
        (d / "career.json.99999.tmp").write_text("{}")
        (d / "sessions" / "ghost.json.99999.tmp").write_text("{}")   # <id>.json.<pid>.tmp — NOT *.json.tmp
        self.assertTrue((d / "career.json").exists())
        self.assertTrue((d / "coach.db").exists())

        removed = paths.reset_data()

        self.assertFalse((d / "career.json").exists())                  # backed up, not deleted
        self.assertTrue((d / "career.reset-backup.json").exists())
        self.assertFalse((d / "coach.db").exists())
        self.assertFalse((d / "career.json.99999.tmp").exists())            # temp swept
        self.assertFalse((d / "sessions" / "ghost.json.99999.tmp").exists())  # report temp swept
        self.assertEqual(list((d / "sessions").glob("*.json")), [])     # reports gone
        self.assertTrue((d / "sessions").is_dir())                      # dir itself kept
        self.assertTrue(any("career.json" in r for r in removed))

    def test_reset_clears_coach_init_cache(self):
        rid = coach.ask("decision", {"x": 1})          # primes coach._INITIALIZED for this path
        coach.set_meta(rid, {"modeled_equity": {"hero_pct": 50}})
        paths.reset_data()                             # unlinks coach.db; must drop the init cache
        # a fresh op must re-create the table, not hit "no such table: requests"
        rid2 = coach.ask("decision", {"y": 2})
        self.assertEqual(coach.poll(rid2)["status"], "pending")


if __name__ == "__main__":
    unittest.main(verbosity=2)
