"""gutcheck settings-store unit suite (stdlib unittest, zero deps).

Covers the settings menu's backing store: defaults when absent, persistence + round-trip,
and the two robustness disciplines (corrupt/non-object tolerated → defaults; atomic write
leaves no temp + preserves unknown keys). The HTTP surface (/api/settings GET/POST) is
verified live during the apparatus / play gate.

    ./venv/bin/python test_settings.py
"""

import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

from gutcheck import coach_daemon
from gutcheck import coach_watch
from gutcheck import game as G
from gutcheck import paths
from gutcheck import seam as seam_mod
from gutcheck import server
from gutcheck import settings


class TestSettings(unittest.TestCase):
    def setUp(self):
        # isolate the data dir so no test can touch ~/.gutcheck (the same structural fix paths.py
        # documents); honored by paths.settings_path() via GUTCHECK_DATA_DIR.
        self.tmp = tempfile.mkdtemp()
        self._prev = os.environ.get(paths.ENV_VAR)
        os.environ[paths.ENV_VAR] = self.tmp

    def tearDown(self):
        if self._prev is None:
            os.environ.pop(paths.ENV_VAR, None)
        else:
            os.environ[paths.ENV_VAR] = self._prev

    def _file(self) -> Path:
        return Path(self.tmp) / "settings.json"

    def test_default_when_absent(self):
        self.assertTrue(settings.coaching_enabled())              # default ON, no file written yet
        self.assertEqual(settings.load(), dict(settings.DEFAULTS))

    def test_set_persists_and_roundtrips(self):
        settings.set_coaching_enabled(False)
        self.assertFalse(settings.coaching_enabled())
        # a fresh on-disk read (a new process would see this) reflects the persisted value
        self.assertFalse(json.loads(self._file().read_text())["coaching_enabled"])
        settings.set_coaching_enabled(True)
        self.assertTrue(settings.coaching_enabled())

    def test_corrupt_file_falls_back_to_defaults(self):
        self._file().write_text("{ not valid json")
        self.assertEqual(settings.load(), dict(settings.DEFAULTS))  # tolerated, never raised
        self.assertTrue(settings.coaching_enabled())

    def test_non_object_json_falls_back(self):
        self._file().write_text("[1, 2, 3]")
        self.assertEqual(settings.load(), dict(settings.DEFAULTS))

    def test_stat_prefs_defaults(self):
        self.assertEqual(settings.stat_prefs(),
                         {"equity_format": "band_pct", "show_pot_odds": True, "show_observed_read": True})

    def test_stat_prefs_validates_each_field(self):
        self._file().write_text('{"equity_format": "bogus", "show_pot_odds": "yes", "show_observed_read": 0}')
        p = settings.stat_prefs()
        self.assertEqual(p["equity_format"], "band_pct")         # bad enum → default
        self.assertTrue(p["show_pot_odds"])                      # non-bool string → default
        self.assertTrue(p["show_observed_read"])                 # int 0 → default (not falsy-coerced)

    def test_apply_patch_valid_partial(self):
        out, err = settings.apply_patch({"equity_format": "band", "show_pot_odds": False})
        self.assertIsNone(err)
        self.assertEqual(out["equity_format"], "band")
        self.assertFalse(out["show_pot_odds"])
        self.assertTrue(out["show_observed_read"])               # untouched key keeps its default
        self.assertTrue(out["coaching_enabled"])                 # …and so does coaching

    def test_apply_patch_rejects_unknown_and_bad_values(self):
        self.assertIn("unknown setting", settings.apply_patch({"bogus": 1})[1])
        self.assertIsNotNone(settings.apply_patch({"equity_format": "nope"})[1])
        self.assertIsNotNone(settings.apply_patch({"show_pot_odds": 1})[1])   # int 1 is not a bool
        self.assertIsNotNone(settings.apply_patch({})[1])                     # empty patch → error
        self.assertIsNotNone(settings.apply_patch(None)[1])                   # non-dict → error
        self.assertTrue(settings.coaching_enabled())                         # no rejected write landed

    def test_coach_provider_defaults_and_validates(self):
        self.assertEqual(settings.coach_provider(), "claude")               # default
        self._file().write_text('{"coach_provider": "codex"}')
        self.assertEqual(settings.coach_provider(), "codex")               # valid enum honored
        self._file().write_text('{"coach_provider": "bogus"}')
        self.assertEqual(settings.coach_provider(), "claude")             # bad enum → default

    def test_apply_patch_coach_provider(self):
        out, err = settings.apply_patch({"coach_provider": "codex"})
        self.assertIsNone(err)
        self.assertEqual(out["coach_provider"], "codex")
        self.assertIsNotNone(settings.apply_patch({"coach_provider": "gpt9"})[1])   # bad enum rejected

    def test_openai_provider_and_dials_default_and_validate(self):
        # the 4th provider + its PER-VENDOR dials. coach_provider accepts openai_api; openai_model/effort
        # default + validate independently of the anthropic api_model/api_effort, so switching provider can
        # never strand the other vendor's id.
        self.assertEqual(settings.openai_model(), "gpt-5.5")            # default flagship
        self.assertEqual(settings.openai_effort(), "low")              # default cheapest-meaningful
        out, err = settings.apply_patch({"coach_provider": "openai_api"})
        self.assertIsNone(err)
        self.assertEqual(out["coach_provider"], "openai_api")
        out, err = settings.apply_patch({"openai_model": "gpt-5.4-mini", "openai_effort": "high"})
        self.assertIsNone(err)
        self.assertEqual((out["openai_model"], out["openai_effort"]), ("gpt-5.4-mini", "high"))
        # the full effort range is offered: 'none' (cheapest floor) and 'xhigh' (deepest) now both validate.
        self.assertIsNone(settings.apply_patch({"openai_effort": "none"})[1])
        self.assertIsNone(settings.apply_patch({"openai_effort": "xhigh"})[1])
        # bad enums rejected; an Anthropic id is NOT a valid openai_model (per-vendor namespaces), and
        # 'minimal' is NOT a valid effort on the offered gpt-5.4/5.5 tiers (replaced by 'none') → rejected.
        self.assertIsNotNone(settings.apply_patch({"openai_model": "claude-opus-4-8"})[1])
        self.assertIsNotNone(settings.apply_patch({"openai_effort": "minimal"})[1])
        # …and the anthropic dials are untouched by the openai writes
        self.assertEqual(settings.api_model(), "claude-opus-4-8")

    def test_openai_model_bad_disk_value_falls_back(self):
        self._file().write_text('{"openai_model": "gpt-9-turbo", "openai_effort": "ludicrous"}')
        self.assertEqual(settings.openai_model(), "gpt-5.5")          # bad id → default
        self.assertEqual(settings.openai_effort(), "low")            # bad effort → default

    def test_only_real_bool_on_disk_is_honored(self):
        # a real JSON bool is honored; any other on-disk value falls back to the DEFAULT rather than
        # being truthiness-coerced (a hand-edited "false" must NOT read as True for a cost toggle).
        self._file().write_text('{"coaching_enabled": false}')
        self.assertFalse(settings.coaching_enabled())            # real bool honored
        self._file().write_text('{"coaching_enabled": "yes"}')
        self.assertTrue(settings.coaching_enabled())             # non-bool → default (True), not truthy "yes"
        self._file().write_text('{"coaching_enabled": "false"}')
        self.assertTrue(settings.coaching_enabled())             # the footgun: string "false" → default, not coerced
        self._file().write_text('{"coaching_enabled": 0}')
        self.assertTrue(settings.coaching_enabled())             # int 0 → default, not falsy-coerced to off

    def test_unknown_keys_preserved_through_write(self):
        self._file().write_text('{"future_knob": 7}')
        loaded = settings.load()
        self.assertEqual(loaded["future_knob"], 7)               # forward-compat: a newer build's knob survives
        self.assertTrue(loaded["coaching_enabled"])              # …and the known default is filled in
        settings.set_coaching_enabled(False)                     # a write must not drop the unknown key
        on_disk = json.loads(self._file().read_text())
        self.assertEqual(on_disk["future_knob"], 7)
        self.assertFalse(on_disk["coaching_enabled"])

    def test_atomic_write_leaves_no_temp(self):
        settings.set_coaching_enabled(False)
        self.assertEqual(list(Path(self.tmp).glob("settings.json.*.tmp")), [])  # os.replace consumed it

    def test_facts_floor_honors_stat_prefs(self):
        # the integration the slice exists for: the stub reads settings.stat_prefs(), so PERSISTED prefs
        # actually shape the facts output (band vs band+%, pot-odds line, observed-read line).
        ctx = {
            "hero": {"strength": "top pair", "hole": ["As", "Kd"]},
            "board": ["Ah", "7c", "2d"], "street": "flop",
            "pot_odds": {"call": 20, "pot": 60, "breakeven_pct": 25},
            "modeled_equity": {"hero_pct": 82, "band": "crushing", "heads_up": True,
                               "vs": {"name": "Cleo", "range_pct": 40}},
            "villains": [{"name": "Cleo", "in_hand": True, "observed": {"vpip_pct": 61}}],
        }
        req = {"id": 1, "kind": "decision", "context": ctx}
        full = coach_watch.stub_answer(req, facts_mode=True)      # defaults: band+%, pot odds + read on
        self.assertIn("(~82%)", full)                            # the precise hero equity number
        self.assertIn("Pot odds", full)
        self.assertIn("Reads:", full)
        settings.apply_patch({"equity_format": "band", "show_pot_odds": False, "show_observed_read": False})
        lean = coach_watch.stub_answer(req, facts_mode=True)
        self.assertNotIn("(~82%)", lean)                         # band-only: the precise % is gone…
        self.assertIn("crushing", lean)                          # …but the felt band stays
        self.assertNotIn("Pot odds", lean)
        self.assertNotIn("Reads:", lean)


class TestSettingsRoute(unittest.TestCase):
    """The /api/settings GET + POST routes over a REAL in-process server (the live_commit harness
    idiom). The frontend reads coaching state here and the toggle writes it; the worker reads the
    same store, so this is the seam between the menu and the cost floor."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._prev_dir = os.environ.get("GUTCHECK_DATA_DIR")
        os.environ["GUTCHECK_DATA_DIR"] = self.tmp
        paths.ensure_data_dir()
        d = Path(self.tmp)
        self.g = G.Game(d / "session.json", d / "career.json", d / "sessions",
                        d / "calibration.json", d / "history.db")
        server.GAME = self.g                              # the GET guard requires GAME+SEAM non-None
        server.SEAM = seam_mod.Seam(self.g)
        self._prev_built_in = server.COACH_BUILT_IN
        server.COACH_BUILT_IN = True                      # pretend the built-in worker is up (the common run)
        self.httpd = HTTPServer(("127.0.0.1", 0), server.Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        server.COACH_BUILT_IN = self._prev_built_in       # don't leak the global across test files
        if self._prev_dir is None:
            os.environ.pop("GUTCHECK_DATA_DIR", None)
        else:
            os.environ["GUTCHECK_DATA_DIR"] = self._prev_dir

    def _get(self, path):
        return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}").read())

    def _post(self, path, body):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        return json.loads(urllib.request.urlopen(req).read())

    def test_get_defaults_on(self):
        s = self._get("/api/settings")
        self.assertTrue(s["coaching_enabled"])             # no file → default ON
        self.assertTrue(s["coach_built_in"])               # the harness set the worker up
        self.assertEqual(s["equity_format"], "band_pct")   # the stat prefs ride the same payload
        self.assertTrue(s["show_pot_odds"])
        self.assertTrue(s["show_observed_read"])
        self.assertEqual(s["coach_provider"], "claude")    # …and the provider (slice c)

    def test_post_provider_persists(self):
        out = self._post("/api/settings", {"coach_provider": "codex"})
        self.assertEqual(out["coach_provider"], "codex")
        self.assertEqual(self._get("/api/settings")["coach_provider"], "codex")   # persisted

    def test_metered_dials_default_and_persist(self):
        s = self._get("/api/settings")
        self.assertEqual(s["api_model"], "claude-opus-4-8")     # the cost dial defaults to the flagship
        self.assertEqual(s["api_effort"], "low")               # the fidelity dial defaults to low
        out = self._post("/api/settings", {"api_model": "claude-haiku-4-5", "api_effort": "high"})
        self.assertEqual(out["api_model"], "claude-haiku-4-5")
        self.assertEqual(out["api_effort"], "high")
        self.assertEqual(self._get("/api/settings")["api_model"], "claude-haiku-4-5")   # persisted
        # xhigh validates at the settings surface (model-blind, like api_model); the worker degrades it to
        # 'high' on a non-Opus model. The picker only offers it for Opus, but the stored value is accepted.
        self.assertEqual(self._post("/api/settings", {"api_effort": "xhigh"})["api_effort"], "xhigh")

    def test_post_bad_metered_dial_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self._post("/api/settings", {"api_model": "gpt-5"})           # not an allowed Anthropic tier
        self.assertEqual(cm.exception.code, 400)
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self._post("/api/settings", {"api_effort": "ludicrous"})      # not a valid effort
        self.assertEqual(cm.exception.code, 400)

    def test_openai_dials_default_persist_and_carry_key_status(self):
        s = self._get("/api/settings")
        self.assertEqual(s["openai_model"], "gpt-5.5")          # openai cost dial default
        self.assertEqual(s["openai_effort"], "low")
        self.assertFalse(s["openai_key"]["set"])                # no openai key set yet
        out = self._post("/api/settings", {"coach_provider": "openai_api",
                                           "openai_model": "gpt-5.4", "openai_effort": "medium"})
        self.assertEqual(out["coach_provider"], "openai_api")
        self.assertEqual((out["openai_model"], out["openai_effort"]), ("gpt-5.4", "medium"))
        again = self._get("/api/settings")
        self.assertEqual(again["openai_model"], "gpt-5.4")      # persisted
        self.assertEqual(again["api_model"], "claude-opus-4-8") # …the anthropic dial is untouched

    def test_post_bad_openai_dial_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self._post("/api/settings", {"openai_model": "claude-opus-4-8"})   # Anthropic id ≠ openai tier
        self.assertEqual(cm.exception.code, 400)
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self._post("/api/settings", {"openai_effort": "minimal"})         # invalid on gpt-5.4/5.5 (→ 'none')
        self.assertEqual(cm.exception.code, 400)

    def test_post_stat_pref_persists(self):
        out = self._post("/api/settings", {"equity_format": "band", "show_pot_odds": False})
        self.assertEqual(out["equity_format"], "band")
        self.assertFalse(out["show_pot_odds"])
        self.assertTrue(out["coaching_enabled"])           # a partial patch leaves other keys alone
        self.assertEqual(self._get("/api/settings")["equity_format"], "band")  # persisted

    def test_post_bad_stat_value_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self._post("/api/settings", {"equity_format": "bogus"})
        self.assertEqual(cm.exception.code, 400)
        self.assertEqual(settings.stat_prefs()["equity_format"], "band_pct")   # rejected write didn't land

    def test_get_reports_no_builtin_worker(self):
        server.COACH_BUILT_IN = False                      # the --no-coach case
        self.assertFalse(self._get("/api/settings")["coach_built_in"])

    def test_post_toggles_off_and_persists(self):
        out = self._post("/api/settings", {"coaching_enabled": False})
        self.assertFalse(out["coaching_enabled"])          # the write echoes the new state
        self.assertFalse(self._get("/api/settings")["coaching_enabled"])  # a later GET sees it
        self.assertFalse(settings.coaching_enabled())      # …and the worker's store reflects it
        self._post("/api/settings", {"coaching_enabled": True})           # toggle back
        self.assertTrue(self._get("/api/settings")["coaching_enabled"])

    def test_post_non_bool_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self._post("/api/settings", {"coaching_enabled": 1})   # JSON 1 is int, not bool → 400
        self.assertEqual(cm.exception.code, 400)
        self.assertTrue(settings.coaching_enabled())               # rejected write left the store untouched

    def test_post_missing_key_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self._post("/api/settings", {"nope": True})
        self.assertEqual(cm.exception.code, 400)

    def test_post_non_dict_body_rejected_cleanly(self):
        # a `null` / list / string body must 400 with the clean message, not an AttributeError 500
        # leaking through the generic catch (the container guard, L1 LOW-2).
        for raw in ("null", "[1,2,3]", '"nope"'):
            req = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/settings",
                                         data=raw.encode(), headers={"Content-Type": "application/json"},
                                         method="POST")
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(req)
            self.assertEqual(cm.exception.code, 400)
            self.assertIn(b"non-empty object", cm.exception.read())   # apply_patch's clean container error

    def test_run_coach_clears_built_in_on_worker_crash(self):
        # COACH_BUILT_IN means "the built-in worker is RUNNING", not "we tried to start it" — so a
        # crashed worker loop must flip it False, or the settings surface offers a live toggle +
        # "talk it through" prose nothing can answer (L1 MED-1).
        server.COACH_BUILT_IN = True
        orig = coach_daemon.run

        def boom(**_kw):
            raise RuntimeError("worker boom")

        coach_daemon.run = boom
        try:
            server._run_coach(Path(self.tmp) / "coach.db", threading.Event())
        finally:
            coach_daemon.run = orig
        self.assertFalse(server.COACH_BUILT_IN)               # the surface now tells the truth
        self.assertFalse(self._get("/api/settings")["coach_built_in"])


if __name__ == "__main__":
    unittest.main()
