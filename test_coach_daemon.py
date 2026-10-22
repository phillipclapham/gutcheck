"""gutcheck coach-daemon unit suite (stdlib unittest, zero deps).

Covers everything in the Slice-2 worker that does NOT need a real `claude` call: prompt
construction, the claude→stub graceful-degradation fallback, channel integration + idempotency,
the signal-clean run loop, env-stripping, and the coach_watch.format_spot refactor. The actual
`claude -p` invocation (call_claude) is verified via `python -m gutcheck.coach_daemon smoke` — it needs a
logged-in `claude` and spends a real call, so here it's stubbed via the injectable runner.

    ./venv/bin/python test_coach_daemon.py
"""

import json
import os
import shutil
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path

from gutcheck import coach
from gutcheck import coach_watch
from gutcheck import coach_daemon
from gutcheck import leakstore
from gutcheck.leakstore import LeakStore

HAS_ANNEAL = leakstore.anneal_available()
_SKIP_ANNEAL = "anneal-memory not installed (the [anneal] extra) — leak-seed is optional"


_PREV_DATA_DIR = None
_TMP_DATA_DIR = None


def setUpModule():
    # the facts-floor stub now reads settings.stat_prefs() (slice b) → isolate the data dir so every
    # stub-bearing test sees DEFAULT prefs, regardless of any real ~/.gutcheck/settings.json on the box.
    global _PREV_DATA_DIR, _TMP_DATA_DIR
    _PREV_DATA_DIR = os.environ.get("GUTCHECK_DATA_DIR")
    _TMP_DATA_DIR = tempfile.mkdtemp()
    os.environ["GUTCHECK_DATA_DIR"] = _TMP_DATA_DIR


def tearDownModule():
    if _PREV_DATA_DIR is None:
        os.environ.pop("GUTCHECK_DATA_DIR", None)
    else:
        os.environ["GUTCHECK_DATA_DIR"] = _PREV_DATA_DIR


def _decision_ctx():
    return {
        "hand_no": 12,
        "hero": {"hole": ["As", "Kd"], "position": "BTN", "strength": "strong (top pair)",
                 "stack": 980},
        "board": ["Ah", "7c", "2d"],
        "street": "flop",
        "pot": 60,
        "pot_odds": {"call": 20, "pot": 60, "breakeven_pct": 25},
        "legal": {"can_fold": True, "can_check": False, "can_raise": True,
                  "min_raise_to": 40, "max_raise_to": 980},
        "action_this_hand": ["Cleo bets 20"],
        "villains": [
            {"name": "Cleo", "position": "BB", "stack": 540, "in_hand": True,
             "observed": {"hands": 18, "vpip_pct": 61, "pf_raises": 1, "showdowns": 4, "net": -120},
             "_truth": {"type": "station"}},
            {"name": "Mara", "position": "CO", "stack": 1010, "in_hand": False,
             "observed": {"hands": 18, "vpip_pct": 11, "pf_raises": 3, "showdowns": 1, "net": 80},
             "_truth": {"type": "rock"}},
        ],
    }


def _review_ctx():
    return {
        "hand_no": 12,
        "hero": {"hole": ["As", "Kd"], "position": "BTN"},
        "final_board": ["Ah", "7c", "2d", "9s", "3h"],
        "pot": 240,
        "result": {"hero_net": 120, "winners": [{"name": "you", "amount": 240}],
                   "showdown": [{"name": "Cleo", "hole": ["Kh", "7h"]}]},
        "action_by_street": ["flop: Cleo bets 20, you raise 60, Cleo calls",
                             "river: you bet 90, Cleo calls"],
        "villains": [
            {"name": "Cleo", "position": "BB",
             "observed": {"hands": 19, "vpip_pct": 61, "pf_raises": 1, "showdowns": 5, "net": -240},
             "_truth": {"type": "station"}},
        ],
    }


class TestFormatSpotRefactor(unittest.TestCase):
    def test_format_spot_has_no_respond_tail(self):
        req = {"id": 7, "kind": "decision", "context": _decision_ctx()}
        spot = coach_watch.format_spot(req)
        self.assertIn("Coaching request #7", spot)
        self.assertIn("Your hand:", spot)
        self.assertNotIn("gutcheck.coach respond", spot)  # the human-only hint must NOT leak into claude

    def test_format_request_appends_respond_tail(self):
        req = {"id": 7, "kind": "decision", "context": _decision_ctx()}
        full = coach_watch.format_request(req)
        self.assertTrue(full.startswith(coach_watch.format_spot(req)))
        self.assertIn("gutcheck.coach respond --id 7", full)  # warm-session view still gets the hint


class TestBuildPrompt(unittest.TestCase):
    def test_prompt_carries_pedagogy_and_spot_decision(self):
        req = {"id": 1, "kind": "decision", "context": _decision_ctx()}
        p = coach_daemon.build_prompt(req)
        self.assertIn("Exploitation contains balance", p)  # strategy frame present (now from doctrine.md)
        self.assertIn("named what it forbids", p)          # P3(b): the read→action doctrine is wired in
        self.assertIn("NEVER name or reveal", p)           # pedagogy guardrail present
        self.assertIn("Your hand:", p)                     # the spot is embedded
        self.assertIn("Cleo", p)
        self.assertNotIn("gutcheck.coach respond", p)     # no human-worker tail in the LLM prompt

    def test_prompt_carries_spot_review(self):
        req = {"id": 2, "kind": "review", "context": _review_ctx()}
        p = coach_daemon.build_prompt(req)
        self.assertIn("Result: you +120", p)
        self.assertIn("final board", p)


class TestAnswerOne(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "coach_test.db"

    def test_claude_success_writes_claude_prose(self):
        rid = coach.ask("decision", _decision_ctx(), self.tmp)
        req = coach.get(rid, self.tmp)
        responded, source = coach_daemon.answer_one(
            req, runner=lambda _p: "Bet big — Cleo pays off everything.", db=self.tmp)
        self.assertTrue(responded)
        self.assertEqual(source, "claude")
        self.assertEqual(coach.poll(rid, self.tmp)["response"], "Bet big — Cleo pays off everything.")

    def test_empty_claude_falls_back_to_stub(self):
        rid = coach.ask("decision", _decision_ctx(), self.tmp)
        req = coach.get(rid, self.tmp)
        responded, source = coach_daemon.answer_one(req, runner=lambda _p: "", db=self.tmp)
        self.assertTrue(responded)
        self.assertEqual(source, "stub")                   # graceful degradation
        resp = coach.poll(rid, self.tmp)["response"]
        self.assertIn("ai coach unavailable", resp.lower())  # the honest degraded facts-only fallback, never a hang
        self.assertNotIn("Recommendation", resp)           # facts only — no template bet/fold/call call

    def test_whitespace_only_claude_falls_back_to_stub(self):
        rid = coach.ask("review", _review_ctx(), self.tmp)
        req = coach.get(rid, self.tmp)
        _responded, source = coach_daemon.answer_one(req, runner=lambda _p: "   \n  ", db=self.tmp)
        self.assertEqual(source, "stub")

    def test_double_answer_is_noop(self):
        rid = coach.ask("decision", _decision_ctx(), self.tmp)
        req = coach.get(rid, self.tmp)
        coach_daemon.answer_one(req, runner=lambda _p: "first answer", db=self.tmp)
        # a second worker picks up the same (now-stale) req object and tries to answer
        responded, _src = coach_daemon.answer_one(req, runner=lambda _p: "second answer", db=self.tmp)
        self.assertFalse(responded)                        # idempotent — first answer stands
        self.assertEqual(coach.poll(rid, self.tmp)["response"], "first answer")


class TestFactsFloor(unittest.TestCase):
    """The facts floor = code-true facts, NO recommendation. Two framings: a DEGRADED head when the AI
    coach is unavailable, and CLEAN facts (no head, no scripted sign-off) when coaching is deliberately
    OFF — a template call taught with confidence trains worse reading (disentangle-(2))."""

    def test_decision_stub_degraded_has_head_and_facts(self):
        out = coach_watch.stub_answer({"id": 1, "kind": "decision", "context": _decision_ctx()})
        self.assertIn("ai coach unavailable", out.lower())     # the degraded head (facts_mode=False default)
        self.assertNotIn("Recommendation", out)                # the confident steer is gone
        for verb in ("Bet for value", "Fold —", "Raise for value", "Marginal", "Call —"):
            self.assertNotIn(verb, out)                        # none of the old template calls
        self.assertNotIn("Your read, your call", out)          # the scripted sign-off is gone (a template tell)
        self.assertIn("strong (top pair)", out)                # the FACTS survive: hand strength
        self.assertIn("need 25%", out)                         # …and pot odds (code-computed)
        self.assertIn("Cleo reads as", out)                    # …and the visible VPIP read

    def test_decision_stub_facts_mode_is_clean_no_head(self):
        # deliberate OFF: just the facts — no "unavailable" head, no scripted sign-off (Phill's ask).
        out = coach_watch.stub_answer({"id": 1, "kind": "decision", "context": _decision_ctx()}, facts_mode=True)
        self.assertNotIn("unavailable", out.lower())           # NO announcement that coaching is off
        self.assertNotIn("Your read, your call", out)
        self.assertIn("strong (top pair)", out)                # …just the facts
        self.assertIn("need 25%", out)

    def test_review_stub_is_facts_only_no_imperatives(self):
        ctx = _review_ctx()
        ctx["decision_points"] = [
            {"street": "flop", "board": ["Ah", "7c", "2d"], "equity": 82, "band": "crushing",
             "to_call": 0, "pot": 60, "vs": "Cleo's range", "action": "bet 40"},
        ]
        out = coach_watch.stub_answer({"id": 2, "kind": "review", "context": ctx})
        self.assertIn("ai coach unavailable", out.lower())     # degraded head (facts_mode=False default)
        self.assertIn("won the pot (+120)", out)               # result fact
        self.assertIn("82%", out)                              # exact-equity decision point (ground truth)
        self.assertNotIn("Read the board texture", out)        # the old imperative is gone
        self.assertNotIn("make sure you charged it", out)
        self.assertNotIn("size your value bets", out)


class TestRunLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "coach_test.db"

    def test_external_stop_event_ends_polling_loop(self):
        # the in-process path (server.py): a polling run (once=False) ends when the stop_event is set,
        # and install_signal_handlers=False must NOT raise even though signal.signal would here.
        ev = threading.Event()
        coach.ask("decision", _decision_ctx(), self.tmp)
        answered = []

        def runner(_prompt):
            answered.append(1)
            ev.set()                                  # stop right after the first answer
            return "coached"

        rc = coach_daemon.run(once=False, interval=0.01, runner=runner, db=self.tmp,
                              stop_event=ev, install_signal_handlers=False, startup_probe=False)
        self.assertEqual(rc, 0)                       # returned instead of polling forever
        self.assertEqual(len(answered), 1)            # the pending request was coached before the stop

    def test_run_in_worker_thread_no_signal_error(self):
        # the exact server.py shape: the full loop runs in a NON-main thread (no signal handlers) and
        # is stopped via the event. A regression that registered signals off-main would raise here.
        coach.ask("decision", _decision_ctx(), self.tmp)
        ev = threading.Event()
        err = {}

        def worker():
            try:
                coach_daemon.run(once=False, interval=0.01, runner=lambda _p: "coached",
                                 db=self.tmp, stop_event=ev, install_signal_handlers=False,
                                 startup_probe=False)
            except Exception as e:                    # signal.signal off-main-thread → ValueError
                err["e"] = repr(e)

        t = threading.Thread(target=worker)
        t.start()
        time.sleep(0.15)                              # let it answer + poll a few times
        ev.set()
        t.join(timeout=3)
        self.assertFalse(t.is_alive())                # the stop_event ended the loop
        self.assertEqual(err, {})                     # and it never touched signals off the main thread

    def test_run_in_thread_with_signals_requested_does_not_raise(self):
        # codex L3 LOW-1: even if a thread caller FORGETS install_signal_handlers=False, the
        # main-thread auto-gate prevents the off-thread `signal only works in main thread` ValueError.
        # (install_signal_handlers defaults True here — the footgun path.)
        coach.ask("decision", _decision_ctx(), self.tmp)
        ev = threading.Event()
        err = {}

        def worker():
            try:
                coach_daemon.run(once=True, db=self.tmp, startup_probe=False, stop_event=ev,
                                 runner=lambda _p: "coached")   # NOTE: install_signal_handlers left default(=True)
            except Exception as e:
                err["e"] = repr(e)

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=3)
        self.assertFalse(t.is_alive())                # once=True returns after one pass, no signal crash
        self.assertEqual(err, {})                     # the auto-gate saved the forgetful caller

    def test_once_answers_all_pending_then_exits(self):
        ids = [coach.ask("decision", _decision_ctx(), self.tmp) for _ in range(3)]
        calls = []

        def runner(prompt):
            calls.append(prompt)
            return f"coaching #{len(calls)}"

        rc = coach_daemon.run(once=True, runner=runner, db=self.tmp)
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 3)
        for rid in ids:
            self.assertEqual(coach.poll(rid, self.tmp)["status"], "answered")

    def test_once_is_idempotent_second_pass_does_nothing(self):
        coach.ask("decision", _decision_ctx(), self.tmp)
        calls = []
        coach_daemon.run(once=True, runner=lambda p: (calls.append(p) or "x"), db=self.tmp)
        self.assertEqual(len(calls), 1)
        coach_daemon.run(once=True, runner=lambda p: (calls.append(p) or "x"), db=self.tmp)
        self.assertEqual(len(calls), 1)                    # nothing pending the second time

    def test_ask_that_waited_gets_a_timeout_inside_the_ui_budget(self):
        # a review that sat behind a leak-seed pass has less of the UI's ~103s poll left: its runner
        # timeout is capped to what remains; one with under MIN_ASK_S left is never sent to the runner.
        waited = coach.ask("review", _decision_ctx(), self.tmp)
        spent = coach.ask("review", _decision_ctx(), self.tmp)
        with coach._conn(self.tmp) as c:
            c.execute("UPDATE requests SET created=? WHERE id=?", (time.time() - 50, waited))
            c.execute("UPDATE requests SET created=? WHERE id=?", (time.time() - 90, spent))
        seen = []

        def runner(prompt, timeout=None):
            seen.append(timeout)
            return "prose"

        coach_daemon.run(once=True, runner=runner, db=self.tmp)
        self.assertEqual(len(seen), 1)                      # the 90s-old request never reached the runner
        self.assertAlmostEqual(seen[0], coach_daemon.UI_ASK_BUDGET_S - 50, delta=2)   # CLAUDE_TIMEOUT (90) is the larger
        self.assertEqual(coach.poll(spent, self.tmp)["status"], "answered")   # …but was still answered (stub)

    def test_capped_ask_failing_does_not_trip_the_breaker_but_a_near_full_one_does(self):
        # the breaker gate in run(): an attempt capped below half of CLAUDE_TIMEOUT that fails says nothing
        # about provider health, so it must not open the breaker; the same failure with a near-full budget
        # left counts. Threshold 1 makes a single counted failure open it, so the SECOND ask reaches the
        # runner only if the first did not count.
        saved = (coach_daemon.BREAKER_THRESHOLD, coach_daemon.UI_ASK_BUDGET_S, coach_daemon.CLAUDE_TIMEOUT)
        coach_daemon.BREAKER_THRESHOLD, coach_daemon.UI_ASK_BUDGET_S, coach_daemon.CLAUDE_TIMEOUT = 1, 95.0, 90.0
        try:
            budget, full = coach_daemon.UI_ASK_BUDGET_S, coach_daemon.CLAUDE_TIMEOUT
            # capped: left = full/2 - 10 (and >= MIN_ASK_S); near-full: left = the whole budget less 5s
            for age, runner_calls in ((budget - (full / 2 - 10), 2), (5, 1)):
                tmpdir = tempfile.mkdtemp()
                self.addCleanup(shutil.rmtree, tmpdir, True)
                db = Path(tmpdir) / "coach_test.db"
                ids = [coach.ask("review", _decision_ctx(), db) for _ in range(2)]
                with coach._conn(db) as c:
                    c.execute("UPDATE requests SET created=?", (time.time() - age,))
                calls = []

                def failing(prompt, timeout=None):
                    calls.append(timeout)
                    return ""                               # the provider failed

                coach_daemon.run(once=True, runner=failing, db=db, startup_probe=False)
                self.assertEqual(len(calls), runner_calls, f"age {age}s")
                for rid in ids:
                    self.assertEqual(coach.poll(rid, db)["status"], "answered")   # all answered (stubs)
        finally:
            coach_daemon.BREAKER_THRESHOLD, coach_daemon.UI_ASK_BUDGET_S, coach_daemon.CLAUDE_TIMEOUT = saved

    def test_coaching_off_serves_facts_floor_without_calling_claude(self):
        # the in-UI toggle OFF (settings.coaching_enabled → False): the worker answers from the
        # honest facts floor and NEVER invokes the claude runner — zero spend. The request still
        # leaves 'pending' (the always-answer floor), so a polling browser is never stranded.
        rid = coach.ask("decision", _decision_ctx(), self.tmp)
        calls = []
        coach_daemon.run(once=True, runner=lambda p: (calls.append(p) or "claude prose"),
                         db=self.tmp, coaching_enabled=lambda: False)
        self.assertEqual(calls, [])                         # claude was NEVER called
        poll = coach.poll(rid, self.tmp)
        self.assertEqual(poll["status"], "answered")        # but the request WAS answered…
        resp = poll["response"].lower()
        self.assertNotIn("unavailable", resp)               # …with CLEAN facts (deliberate OFF → no degraded head)
        self.assertIn("need 25%", resp)                     # …just the facts (pot odds present)

    def test_coaching_on_callable_still_calls_claude(self):
        # the toggle is live, not a one-way switch: an explicit ON callable routes to claude.
        rid = coach.ask("decision", _decision_ctx(), self.tmp)
        calls = []
        coach_daemon.run(once=True, runner=lambda p: (calls.append(p) or "coached"),
                         db=self.tmp, coaching_enabled=lambda: True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(coach.poll(rid, self.tmp)["response"], "coached")

    def test_coaching_enabled_default_none_is_always_on(self):
        # backward-compat: no coaching_enabled arg = the standalone daemon's unchanged always-on path.
        coach.ask("decision", _decision_ctx(), self.tmp)
        calls = []
        coach_daemon.run(once=True, runner=lambda p: (calls.append(p) or "x"), db=self.tmp)
        self.assertEqual(len(calls), 1)

    def test_provider_codex_routes_to_call_codex(self):
        # slice c: provider="codex" maps to call_codex per-request (no explicit runner injected).
        coach.ask("decision", _decision_ctx(), self.tmp)
        seen = []
        o_codex, o_claude = coach_daemon.call_codex, coach_daemon.call_claude
        coach_daemon.call_codex = lambda p, **k: (seen.append("codex") or "codex coached")
        coach_daemon.call_claude = lambda p, **k: (seen.append("claude") or "claude coached")
        try:
            coach_daemon.run(once=True, db=self.tmp, startup_probe=False, provider=lambda: "codex")
        finally:
            coach_daemon.call_codex, coach_daemon.call_claude = o_codex, o_claude
        self.assertEqual(seen, ["codex"])                  # routed to codex, never claude

    def test_provider_default_is_claude(self):
        # no provider arg = unchanged default → call_claude (the standalone daemon's behavior).
        coach.ask("decision", _decision_ctx(), self.tmp)
        seen = []
        o_codex, o_claude = coach_daemon.call_codex, coach_daemon.call_claude
        coach_daemon.call_codex = lambda p, **k: (seen.append("codex") or "x")
        coach_daemon.call_claude = lambda p, **k: (seen.append("claude") or "x")
        try:
            coach_daemon.run(once=True, db=self.tmp, startup_probe=False)
        finally:
            coach_daemon.call_codex, coach_daemon.call_claude = o_codex, o_claude
        self.assertEqual(seen, ["claude"])

    def test_explicit_runner_overrides_provider(self):
        # an injected runner wins over the provider mapping (the test seam stays intact).
        coach.ask("decision", _decision_ctx(), self.tmp)
        seen = []
        coach_daemon.run(once=True, db=self.tmp, startup_probe=False, provider=lambda: "codex",
                         runner=lambda p: (seen.append("injected") or "x"))
        self.assertEqual(seen, ["injected"])


class TestNoFalseNestingWarning(unittest.TestCase):
    def test_coach_started_inside_claude_code_does_not_announce_the_stub(self):
        # 0.1.2 printed "every answer will fall back to the stub" whenever CLAUDECODE was set, but
        # _claude_env strips the nesting sentinels and a gutcheck started from a Claude Code session
        # answered via claude (claude=1 stub=0). The warning told the player coaching was broken.
        import contextlib
        import io
        from gutcheck import server as srv
        saved = os.environ.get("CLAUDECODE")
        os.environ["CLAUDECODE"] = "1"
        stop = threading.Event()
        stop.set()                                   # the loop exits at once; only startup output runs
        err = io.StringIO()
        try:
            with tempfile.TemporaryDirectory() as td, contextlib.redirect_stderr(err):
                srv._run_coach(Path(td) / "coach.db", stop)
        finally:
            if saved is None:
                os.environ.pop("CLAUDECODE", None)
            else:
                os.environ["CLAUDECODE"] = saved
        self.assertIn("[coach_daemon] up", err.getvalue())          # the startup path really ran
        self.assertNotIn("fall back to the stub", err.getvalue())


class TestClaudeEnv(unittest.TestCase):
    def test_strips_nesting_and_api_key_sets_hook_suppress(self):
        saved = dict(os.environ)
        try:
            os.environ["CLAUDECODE"] = "1"
            os.environ["CLAUDE_CODE_ENTRYPOINT"] = "cli"
            os.environ["CLAUDE_CODE_SSE_PORT"] = "1234"
            os.environ["ANTHROPIC_API_KEY"] = "sk-should-be-stripped"
            os.environ["KEEPME"] = "yes"
            env = coach_daemon._claude_env()
            self.assertNotIn("CLAUDECODE", env)
            self.assertNotIn("ANTHROPIC_API_KEY", env)     # → rides the subscription, not the API
            self.assertFalse(any(k.startswith("CLAUDE_CODE_") for k in env))
            self.assertEqual(env.get("FLOW_HOOK_SUPPRESS"), "1")
            self.assertEqual(env.get("KEEPME"), "yes")     # unrelated env preserved (PATH/HOME/keychain)
        finally:
            os.environ.clear()
            os.environ.update(saved)

    def test_argv_core_shape(self):
        argv = coach_daemon._claude_argv("hello")
        self.assertIn("-p", argv)
        self.assertIn("hello", argv)
        self.assertEqual(argv[argv.index("--model") + 1], coach_daemon.COACH_MODEL)
        self.assertEqual(argv[argv.index("--output-format") + 1], "json")
        self.assertEqual(argv[argv.index("--tools") + 1], "")          # all tools disabled
        self.assertIn("--no-session-persistence", argv)                # no session litter
        self.assertNotIn("--bare", argv)                               # --bare would break subscription billing


class TestCodexRunner(unittest.TestCase):
    """call_codex's contract (slice c) — fed a canned subprocess.run, no real codex spawn."""

    def test_pipes_prompt_on_stdin_and_reads_o_file(self):
        captured = {}

        def fake_run(argv, timeout=None, input=None, **kw):
            captured["argv"], captured["input"] = argv, input
            Path(argv[argv.index("-o") + 1]).write_text("codex: value-bet bigger\n")  # codex writes -o
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        orig = coach_daemon._run_tracked
        coach_daemon._run_tracked = fake_run
        try:
            out = coach_daemon.call_codex("THE PROMPT")
        finally:
            coach_daemon._run_tracked = orig
        self.assertEqual(out, "codex: value-bet bigger")   # read from the -o file, stripped
        self.assertEqual(captured["input"], "THE PROMPT")  # piped on stdin (EOF) …
        self.assertNotIn("THE PROMPT", captured["argv"])   # … NOT a positional arg (the hang class)
        self.assertIn("exec", captured["argv"])

    def test_nonzero_exit_returns_empty(self):
        def fake_run(argv, timeout=None, input=None, **kw):
            return types.SimpleNamespace(returncode=1, stdout="", stderr="boom")

        orig = coach_daemon._run_tracked
        coach_daemon._run_tracked = fake_run
        try:
            self.assertEqual(coach_daemon.call_codex("p"), "")  # failure → '' → grounded-stub fallback
        finally:
            coach_daemon._run_tracked = orig

    def test_no_orphan_temp_left(self):
        before = set(Path(tempfile.gettempdir()).glob("fp_codex_*.txt"))

        def fake_run(argv, timeout=None, input=None, **kw):
            Path(argv[argv.index("-o") + 1]).write_text("x")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        orig = coach_daemon._run_tracked
        coach_daemon._run_tracked = fake_run
        try:
            coach_daemon.call_codex("p")
        finally:
            coach_daemon._run_tracked = orig
        after = set(Path(tempfile.gettempdir()).glob("fp_codex_*.txt"))
        self.assertEqual(after - before, set())            # the -o temp is unlinked in finally


class TestCallClaudeParsing(unittest.TestCase):
    """call_claude's PARSING (not the spawn) — feed it canned subprocess results."""

    def _run_with_fake_subprocess(self, returncode, stdout, stderr=""):
        import types
        fake = types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)
        orig = coach_daemon._run_tracked
        coach_daemon._run_tracked = lambda *a, **k: fake
        try:
            return coach_daemon.call_claude("prompt")
        finally:
            coach_daemon._run_tracked = orig

    def test_parses_json_result_field(self):
        out = self._run_with_fake_subprocess(0, json.dumps({"result": "  fold here  ", "is_error": False}))
        self.assertEqual(out, "fold here")

    def test_is_error_returns_empty(self):
        out = self._run_with_fake_subprocess(0, json.dumps({"result": "boom", "is_error": True}))
        self.assertEqual(out, "")                          # → triggers stub fallback

    def test_plain_text_stdout_used_as_is(self):
        out = self._run_with_fake_subprocess(0, "not json, just prose")
        self.assertEqual(out, "not json, just prose")

    def test_empty_stdout_returns_empty(self):
        self.assertEqual(self._run_with_fake_subprocess(0, "   "), "")

    def test_nonzero_exit_with_plaintext_is_rejected(self):
        # an auth/nesting error printed to stdout on a failure exit must NOT become coaching prose
        self.assertEqual(self._run_with_fake_subprocess(1, "Error: not logged in"), "")

    def test_nonzero_exit_with_valid_json_success_is_accepted(self):
        # a real result survives a housekeeping non-zero exit
        out = self._run_with_fake_subprocess(1, json.dumps({"result": "call", "is_error": False}))
        self.assertEqual(out, "call")

    def test_non_string_result_returns_empty_not_raise(self):
        # JSON shape drift (result is a dict) → '' rather than an AttributeError
        out = self._run_with_fake_subprocess(0, json.dumps({"result": {"x": 1}, "is_error": False}))
        self.assertEqual(out, "")


class TestOneWorkerPerRequest(unittest.TestCase):
    """spore-155, reproduced 10-17: the built-in coach and an external daemon on one coach.db each
    fired `claude -p` for the same request (a stub on PATH logged two spot invocations for one ask)."""

    def test_two_workers_on_one_channel_spend_once(self):
        db = Path(tempfile.mkdtemp()) / "coach.db"
        coach.ask("review", {"hand_no": 1, "summary": "x"}, db)
        calls, lock = [], threading.Lock()

        def slow_runner(prompt):
            with lock:
                calls.append(prompt)
            time.sleep(0.6)                      # long enough that the second worker polls mid-call
            return "coached"

        workers = [threading.Thread(target=coach_daemon.run, kwargs=dict(once=True, runner=slow_runner, db=db))
                   for _ in range(2)]
        for w in workers:
            w.start()
        for w in workers:
            w.join(10)
        self.assertEqual(len(calls), 1)          # one request, one spend
        self.assertEqual(coach.get(1, db)["status"], "answered")

    def test_a_dead_workers_claim_lapses(self):
        db = Path(tempfile.mkdtemp()) / "coach.db"
        rid = coach.ask("review", {"hand_no": 1}, db)
        self.assertTrue(coach.claim(rid, "w1", path=db))
        self.assertFalse(coach.claim(rid, "w2", path=db))                    # live claim holds
        self.assertTrue(coach.claim(rid, "w2", lease=-1.0, path=db))          # lapsed claim is taken over


class TestReapInflight(unittest.TestCase):
    """spore-155 / codex MED-1, reproduced 10-17: killing the server mid-answer left the `claude -p`
    child running, re-parented to init (ppid 1)."""

    def test_reap_terminates_the_in_flight_child(self):
        done = {}

        def call():
            done["r"] = coach_daemon._run_tracked(["sleep", "30"], timeout=60)

        t = threading.Thread(target=call)
        t.start()
        for _ in range(100):                     # wait for the child to register
            with coach_daemon._INFLIGHT_LOCK:
                procs = list(coach_daemon._INFLIGHT)
            if procs:
                break
            time.sleep(0.05)
        self.assertEqual(len(procs), 1)
        self.assertEqual(coach_daemon.reap_inflight(), 1)
        t.join(10)
        self.assertFalse(t.is_alive())
        self.assertIsNotNone(procs[0].returncode)                           # dead, not orphaned
        self.assertNotEqual(done["r"].returncode, 0)
        self.assertEqual(coach_daemon._INFLIGHT, set())
        with self.assertRaises(OSError):                                    # a call racing the shutdown must not spawn
            coach_daemon._run_tracked(["sleep", "30"], timeout=60)

    def tearDown(self):
        coach_daemon._REAPED = False


class TestBreaker(unittest.TestCase):
    """The half-open circuit breaker as a unit, with an injected clock."""

    def _breaker(self, threshold=3, cooldown=60):
        self.t = [0.0]
        return coach_daemon._Breaker(threshold, cooldown, clock=lambda: self.t[0])

    def test_closed_until_threshold_then_opens(self):
        b = self._breaker(threshold=3)
        for _ in range(2):
            self.assertTrue(b.should_call_claude())
            self.assertIsNone(b.record("stub"))           # failing, not yet at threshold
        self.assertTrue(b.should_call_claude())
        self.assertEqual(b.record("stub"), "open")         # 3rd consecutive failure opens
        self.assertFalse(b.should_call_claude())           # OPEN → serve stubs, don't call

    def test_success_resets_fail_count(self):
        b = self._breaker(threshold=3)
        b.should_call_claude(); b.record("stub")
        b.should_call_claude(); b.record("claude")         # success resets
        b.should_call_claude(); self.assertIsNone(b.record("stub"))   # back to 1, not near threshold
        self.assertTrue(b.should_call_claude())

    def test_any_provider_success_resets(self):
        # slice c: a codex success is a SUCCESS — must reset like claude (else the breaker would never
        # reset on codex and could trip spuriously). A non-claude provider source can't be "untouched".
        b = self._breaker(threshold=3)
        b.should_call_claude(); b.record("stub")
        b.should_call_claude(); self.assertIsNone(b.record("codex"))  # codex success resets the fail count
        b.should_call_claude(); self.assertIsNone(b.record("stub"))   # back to 1, not 2
        self.assertTrue(b.should_call_claude())

    def test_error_source_never_moves_breaker(self):
        b = self._breaker(threshold=2)
        for _ in range(5):
            self.assertTrue(b.should_call_claude())
            self.assertIsNone(b.record("error"))           # payload errors never open the breaker

    def test_half_open_probe_success_closes(self):
        b = self._breaker(threshold=1, cooldown=60)
        b.should_call_claude(); self.assertEqual(b.record("stub"), "open")
        self.assertFalse(b.should_call_claude())            # OPEN during cooldown
        self.t[0] = 61                                      # cooldown elapses
        self.assertTrue(b.should_call_claude())             # → HALF_OPEN probe allowed
        self.assertIsNone(b.record("claude"))               # probe succeeds → CLOSED
        self.assertTrue(b.should_call_claude())

    def test_half_open_probe_failure_reopens_immediately(self):
        b = self._breaker(threshold=3, cooldown=60)
        for _ in range(3):
            b.should_call_claude(); b.record("stub")        # opens after 3
        self.assertFalse(b.should_call_claude())
        self.t[0] = 61
        self.assertTrue(b.should_call_claude())             # HALF_OPEN
        self.assertEqual(b.record("stub"), "reopen")        # ONE failure re-opens (not another full 3)
        self.assertFalse(b.should_call_claude())

    def test_trip_cold_starts_open(self):
        b = self._breaker(threshold=3, cooldown=60)
        b.trip_cold()
        self.assertFalse(b.should_call_claude())            # OPEN from the start (probe-failed startup)
        self.t[0] = 61
        self.assertTrue(b.should_call_claude())             # HALF_OPEN after the cooldown

    def test_is_closed_is_pure_and_proven_only(self):
        # P4b-ii: the leak-seed gate reads is_closed() — True ONLY when the provider is PROVEN-good, and it
        # must NOT transition the breaker (can't steal should_call_claude's half-open recovery probe).
        b = self._breaker(threshold=1, cooldown=60)
        self.assertTrue(b.is_closed())                      # CLOSED at start → seeding allowed
        b.should_call_claude(); self.assertEqual(b.record("stub"), "open")   # → OPEN
        self.assertFalse(b.is_closed())                     # OPEN during cooldown → no seeding
        self.t[0] = 61                                      # cooldown elapses, but recovery is UNPROVEN
        self.assertFalse(b.is_closed())                     # post-cooldown-unproven → STILL no seeding (HIGH-1)
        self.assertTrue(b.should_call_claude())             # is_closed did NOT consume the half-open probe
        self.assertFalse(b.is_closed())                     # HALF_OPEN (probe pending) → still no seeding
        self.assertIsNone(b.record("claude"))               # probe succeeds → CLOSED
        self.assertTrue(b.is_closed())                      # proven-good again → seeding resumes


class TestHardening(unittest.TestCase):
    """Slice-2 hardening from the L1/L2 apparatus: crash-safety, circuit breaker, stale expiry."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "coach_test.db"

    def test_malformed_request_answered_with_safe_fallback_not_crash(self):
        # villain missing 'position' → KeyError inside the formatter, BEFORE the runner is reached.
        # The shared exposure point under both the claude and stub paths — must still answer safely.
        bad = {"hand_no": 1, "hero": {"hole": ["As", "Kd"]}, "villains": [{"name": "X"}]}
        rid = coach.ask("decision", bad, self.tmp)
        req = coach.get(rid, self.tmp)
        responded, source = coach_daemon.answer_one(req, runner=lambda _p: "should not matter",
                                                    db=self.tmp)
        self.assertTrue(responded)
        self.assertEqual(source, "error")
        self.assertEqual(coach.poll(rid, self.tmp)["status"], "answered")       # row left the queue
        self.assertEqual(coach.poll(rid, self.tmp)["response"], coach_daemon._SAFE_FALLBACK)

    def test_poison_request_does_not_crash_the_loop(self):
        coach.ask("decision", {"hand_no": 1, "hero": {}, "villains": [{"name": "X"}]}, self.tmp)
        rc = coach_daemon.run(once=True, runner=lambda _p: "x", db=self.tmp, startup_probe=False)
        self.assertEqual(rc, 0)                                                  # did not raise
        self.assertEqual(len(coach.pending(self.tmp)), 0)                        # poison didn't linger

    def test_circuit_breaker_stops_calling_claude_after_threshold(self):
        saved = coach_daemon.BREAKER_THRESHOLD
        coach_daemon.BREAKER_THRESHOLD = 3
        try:
            for _ in range(6):
                coach.ask("decision", _decision_ctx(), self.tmp)
            calls = {"n": 0}

            def failing(_p):
                calls["n"] += 1
                return ""                                                        # claude always fails

            coach_daemon.run(once=True, runner=failing, db=self.tmp, startup_probe=False)
            self.assertEqual(calls["n"], 3)                                      # breaker opened after 3
            self.assertEqual(len(coach.pending(self.tmp)), 0)                    # all 6 answered (stubs)
        finally:
            coach_daemon.BREAKER_THRESHOLD = saved

    def test_stale_decision_serves_facts_stub_without_spending_a_claude_call(self):
        saved = coach_daemon.STALE_DECISION_S
        coach_daemon.STALE_DECISION_S = -1                                       # everything is "stale"
        try:
            rid = coach.ask("decision", _decision_ctx(), self.tmp)
            calls = {"n": 0}
            coach_daemon.run(once=True, db=self.tmp, startup_probe=False,
                             runner=lambda _p: (calls.__setitem__("n", calls["n"] + 1) or "x"))
            self.assertEqual(calls["n"], 0)                                      # never spent a claude call
            resp = coach.poll(rid, self.tmp)
            self.assertEqual(resp["status"], "answered")
            # the honest facts FLOOR, not a dead "expired" placeholder — a still-polling player still
            # gets the band / equity / pot-odds facts (codex L3 HIGH-1). Coaching is ON here + the spot
            # aged, so the degraded head shows (the coach was wanted, the spot just passed).
            self.assertIn("ai coach unavailable", resp["response"].lower())
            self.assertNotIn("expired", resp["response"])
        finally:
            coach_daemon.STALE_DECISION_S = saved

    def test_review_never_expires(self):
        saved = coach_daemon.STALE_DECISION_S
        coach_daemon.STALE_DECISION_S = -1
        try:
            rid = coach.ask("review", _review_ctx(), self.tmp)
            calls = {"n": 0}
            coach_daemon.run(once=True, db=self.tmp, startup_probe=False,
                             runner=lambda _p: (calls.__setitem__("n", calls["n"] + 1) or "good review"))
            self.assertEqual(calls["n"], 1)                                      # reviews are answered
            self.assertEqual(coach.poll(rid, self.tmp)["response"], "good review")
        finally:
            coach_daemon.STALE_DECISION_S = saved

    def test_channel_read_error_does_not_crash_the_daemon(self):
        orig = coach.pending

        def boom(*_a, **_k):
            raise RuntimeError("database is locked")

        coach.pending = boom
        try:
            rc = coach_daemon.run(once=True, runner=lambda _p: "x", db=self.tmp, startup_probe=False)
            self.assertEqual(rc, 0)                                  # degraded, did not raise/crash-loop
        finally:
            coach.pending = orig

    def test_is_stale_decision_defensive_on_bad_created(self):
        # the schema enforces created NOT NULL, but defend the loop anyway — an unparseable
        # timestamp must be treated as NOT stale (answered normally), never raise in the loop.
        self.assertFalse(coach_daemon._is_stale_decision({"kind": "decision", "created": None}, 1e9))
        self.assertFalse(coach_daemon._is_stale_decision({"kind": "decision", "created": "oops"}, 1e9))
        self.assertFalse(coach_daemon._is_stale_decision({"kind": "review", "created": None}, 1e9))


def _leak_doc(name, level, ids, explanation):
    """A minimal valid 4-section anneal continuity doc carrying one leak line (mirrors the seam tests)."""
    ev = f" [evidence: {', '.join(ids)} \"{explanation}\"]" if ids else ""
    return (
        "# gutcheck-operator — Memory (v1)\n\n"
        "## State\nLeak-memory session.\n\n"
        f"## Patterns\n- ! {name} | {level}x (2026-06-24){ev}\n\n"
        "## Decisions\n[decided(rationale: \"read-based, behavior language only\", on: \"2026-06-24\")] No archetype.\n\n"
        "## Context\nReads recorded this session, grounded in the hands.\n"
    )


class TestExtractDoc(unittest.TestCase):
    """P4b-ii: _extract_doc pulls the continuity doc out of the author's response (transport clean-up)."""

    _DOC = ("# T\n\n## State\nx\n\n## Patterns\n- a | 1x (d)\n\n## Decisions\nx\n\n## Context\nx\n")

    def test_strips_a_wrapping_code_fence(self):
        self.assertEqual(coach_daemon._extract_doc(f"```markdown\n{self._DOC}\n```"), self._DOC.strip())
        self.assertEqual(coach_daemon._extract_doc(f"```\n{self._DOC}```"), self._DOC.strip())

    def test_trims_preamble_before_the_first_heading(self):
        out = coach_daemon._extract_doc(f"Here's the updated memory:\n\n{self._DOC}")
        self.assertTrue(out.startswith("# T"))

    def test_does_not_mangle_a_titleless_doc(self):
        # line-anchored: a doc that opens at "## State" (no "# " title) must keep "## State" intact, not
        # get split on the "# " substring inside it.
        titleless = "## State\nx\n\n## Patterns\n- a | 1x (d)\n"
        self.assertTrue(coach_daemon._extract_doc(titleless).startswith("## State"))

    def test_no_patterns_section_is_refused(self):
        self.assertEqual(coach_daemon._extract_doc(""), "")
        self.assertEqual(coach_daemon._extract_doc("just prose, no doc here"), "")
        self.assertEqual(coach_daemon._extract_doc("## State\nstate only, nothing to graduate"), "")

    def test_preamble_fence_and_outro(self):
        # L1 HIGH-1: the most common chatty LLM shape — preamble + ```md wrapper + trailing outro. The
        # closing fence + outro must NOT survive into the committed doc (they'd compound into ## Context).
        resp = f"Here's the updated leak memory:\n\n```markdown\n{self._DOC}\n```\n\nHope that helps!"
        out = coach_daemon._extract_doc(resp)
        self.assertEqual(out, self._DOC.strip())
        self.assertNotIn("```", out)
        self.assertNotIn("Hope that helps", out)

    def test_leading_non_doc_fence_then_real_doc(self):
        # L1 HIGH-2: a leading ```json plan block before the real doc must NOT cause the doc to be dropped.
        resp = f"```json\n{{\"plan\": \"name the leak\"}}\n```\n\n{self._DOC}"
        out = coach_daemon._extract_doc(resp)
        self.assertTrue(out.startswith("# T"))
        self.assertIn("## Patterns", out)
        self.assertNotIn("plan", out)

    def test_leading_junk_fence_with_heading_then_real_doc(self):
        # codex L3 MED-1 (shape-5): a junk fence whose content includes a markdown heading, before the
        # real doc — the anchor must skip the junk heading (its closing ``` sits above ## Patterns) and
        # take the real doc. (The whole-doc-in-```md case stays intact — see test_strips_a_wrapping_*.)
        resp = ("```\n# plan: name the leak\nsome notes\n```\n\n"
                "# Real\n\n## State\nx\n\n## Patterns\n- a | 1x (d)\n\n## Decisions\nx\n\n## Context\nx\n")
        out = coach_daemon._extract_doc(resp)
        self.assertTrue(out.startswith("# Real"))
        self.assertIn("## Patterns", out)
        self.assertNotIn("plan", out)
        self.assertNotIn("```", out)


@unittest.skipUnless(HAS_ANNEAL, _SKIP_ANNEAL)
class TestEnforceLevelPolicy(unittest.TestCase):
    """codex L3 HIGH-2: structural cap on author-asserted levels (don't trust the prompt) — a NEW leak
    starts at 1x, an EXISTING leak increments by at most one, caps DOWN only so legit graduation is kept."""

    def test_new_leak_overlevel_capped_to_1x(self):
        out = coach_daemon._enforce_level_policy(_leak_doc("brand_new", 3, ["aa11bb22"], "grounded words"), "")
        line = next(l for l in out.splitlines() if "brand_new" in l)
        self.assertIn("| 1x", line)
        self.assertNotIn("| 3x", line)

    def test_new_leak_1x_unchanged(self):
        out = coach_daemon._enforce_level_policy(_leak_doc("brand_new", 1, ["aa11bb22"], "grounded words"), "")
        self.assertIn("| 1x", next(l for l in out.splitlines() if "brand_new" in l))

    def test_existing_legit_increment_unchanged(self):
        current = _leak_doc("sizing", 1, [], "the prior explanation")
        out = coach_daemon._enforce_level_policy(_leak_doc("sizing", 2, ["aa11bb22"], "fresh words"), current)
        self.assertIn("| 2x", next(l for l in out.splitlines() if "sizing" in l))

    def test_existing_level_jump_capped(self):
        current = _leak_doc("sizing", 1, [], "the prior explanation")
        out = coach_daemon._enforce_level_policy(_leak_doc("sizing", 3, ["aa11bb22"], "fresh words"), current)
        line = next(l for l in out.splitlines() if "sizing" in l)
        self.assertIn("| 2x", line)               # 1x -> 3x jump capped to +1
        self.assertNotIn("| 3x", line)

    def test_prose_lines_untouched(self):
        doc = "# T\n\n## State\nsome prose | not a pattern\n\n## Patterns\n- a | 1x (d)\n"
        out = coach_daemon._enforce_level_policy(doc, "")
        self.assertIn("some prose | not a pattern", out)


class TestLeakSeedGate(unittest.TestCase):
    """P4b-ii: the run-loop GATE wiring — enough reads + coaching on + breaker closed → call seed_leaks.
    Uses a spy + a stub store so the gate is tested without anneal (the WRITE path is covered separately)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "coach_test.db"

    def test_loop_drives_seeding_at_threshold(self):
        seen = {"n": 0}
        orig_seed = coach_daemon.seed_leaks
        orig_thr, orig_int = coach_daemon.LEAK_SEED_THRESHOLD, coach_daemon.LEAK_SEED_MIN_INTERVAL
        coach_daemon.seed_leaks = lambda ls, runner=None: (
            seen.__setitem__("n", seen["n"] + 1) or {"seeded": True, "result": {}})
        coach_daemon.LEAK_SEED_THRESHOLD = 1
        coach_daemon.LEAK_SEED_MIN_INTERVAL = 0.0

        class _StubLeakStore:
            def new_read_count(self):
                return 5                                   # always above threshold

        stop = threading.Event()
        try:
            t = threading.Thread(target=lambda: coach_daemon.run(
                once=False, interval=0.01, runner=lambda _p: "x", db=self.tmp,
                startup_probe=False, stop_event=stop, install_signal_handlers=False,
                leak_store=_StubLeakStore()))
            t.start()
            deadline = time.time() + 5
            while seen["n"] == 0 and time.time() < deadline:
                time.sleep(0.02)
            stop.set()
            t.join(timeout=5)
            self.assertGreaterEqual(seen["n"], 1)          # the loop reached the seed driver
        finally:
            stop.set()
            coach_daemon.seed_leaks = orig_seed
            coach_daemon.LEAK_SEED_THRESHOLD = orig_thr
            coach_daemon.LEAK_SEED_MIN_INTERVAL = orig_int

    def test_no_leak_store_means_no_seeding(self):
        # isolate the `leak_store is None` guard (L1 LOW-2): a POLLING loop (once=False, so the not-once
        # guard is satisfied) with leak_store=None must never reach the seed driver. The legacy external
        # daemon CLI runs exactly this shape (it passes no leak_store).
        seen = {"n": 0}
        orig_seed = coach_daemon.seed_leaks
        orig_int = coach_daemon.LEAK_SEED_MIN_INTERVAL
        coach_daemon.seed_leaks = lambda *a, **k: seen.__setitem__("n", seen["n"] + 1)
        coach_daemon.LEAK_SEED_MIN_INTERVAL = 0.0          # interval can't be what's suppressing it
        stop = threading.Event()
        try:
            t = threading.Thread(target=lambda: coach_daemon.run(
                once=False, interval=0.01, runner=lambda _p: "x", db=self.tmp,
                startup_probe=False, stop_event=stop, install_signal_handlers=False, leak_store=None))
            t.start()
            time.sleep(0.2)                                # several idle poll passes
            stop.set()
            t.join(timeout=5)
            self.assertEqual(seen["n"], 0)                 # leak_store=None → the seed branch is inert
        finally:
            stop.set()
            coach_daemon.seed_leaks = orig_seed
            coach_daemon.LEAK_SEED_MIN_INTERVAL = orig_int


@unittest.skipUnless(HAS_ANNEAL, _SKIP_ANNEAL)
class TestLeakSeedDriver(unittest.TestCase):
    """P4b-ii: seed_leaks drives prepare -> author (injected runner) -> commit through anneal end-to-end."""

    def _ls_with_reads(self, tmp, k=3):
        ls = LeakStore(Path(tmp) / "leaks.db")
        for h in range(1, k + 1):
            ls.record_read_episode({"session_id": "s1", "hand_no": h, "n": 1, "street": "river",
                                    "gut": 66, "truth": 34.0, "lean": "value", "gap": 32,
                                    "note": f"thin value bet into a caller, hand {h}"})
        return ls

    def _ids(self, ls, k):
        eps = ls.recall_leaks(limit=k) or []
        return [e.id for e in eps[:k]]

    def test_seed_authors_and_commits(self):
        with tempfile.TemporaryDirectory() as tmp:
            ls = self._ls_with_reads(tmp, 3)
            try:
                ids = self._ids(ls, 3)
                self.assertEqual(len(ids), 3)
                doc = _leak_doc("barrel_into_the_caller", 1, ids,
                                "river value bet barreled into a caller who only calls")
                # the injected author returns the doc inside a code fence — exercises _extract_doc too.
                outcome = coach_daemon.seed_leaks(ls, runner=lambda _p: f"```markdown\n{doc}\n```")
                self.assertTrue(outcome.get("seeded"), outcome)
                self.assertIn("barrel_into_the_caller", ls.get_leaks())
            finally:
                ls.close()

    def test_seeded_leak_is_recallable_same_session(self):
        # the WHOLE POINT of mid-session seeding (P4b-ii) feeding P4b-i recall: a leak the daemon seeds
        # DURING play is recallable at a later decision point in the SAME session — the loop closes.
        with tempfile.TemporaryDirectory() as tmp:
            ls = self._ls_with_reads(tmp, 3)
            try:
                ids = self._ids(ls, 3)
                doc = _leak_doc("barrel_into_the_caller", 1, ids,
                                "river value barrel into a caller who only calls")
                self.assertTrue(coach_daemon.seed_leaks(ls, runner=lambda _p: doc).get("seeded"))
                hits = ls.recall_relevant_leaks("river value barrel into a caller", limit=2)
                self.assertTrue(any(h["name"] == "barrel_into_the_caller" for h in hits), hits)
            finally:
                ls.close()

    def test_prompt_builds_from_a_real_prepare_package(self):
        # the author prompt must compose cleanly from a REAL prepare package (episodes + doctrine + today)
        # — no KeyError on the package shape, and it carries the citable ids + the §3f behavior-language rule.
        with tempfile.TemporaryDirectory() as tmp:
            ls = self._ls_with_reads(tmp, 3)
            try:
                wrap = ls.prepare_leak_wrap()
                prompt = coach_daemon._leak_author_prompt(wrap.get("package") or {})
                self.assertIn("OBSERVED-BEHAVIOR LANGUAGE", prompt)       # the archetype-gate rule (§3f)
                self.assertIn("RECENT COMMITTED READS", prompt)
                self.assertIn(self._ids(ls, 1)[0], prompt)               # a real episode id to cite
                ls.cancel_leak_wrap()
            finally:
                ls.close()

    def test_run_loop_seeds_and_recalls_end_to_end(self):
        # L4 integration: the REAL run() loop (the full gate — idle + threshold + breaker-closed +
        # interval) drives a real seed -> commit, and the leak becomes recallable. Exercises the whole
        # system at once: the run gate, the functools.partial(runner, timeout=) wiring, seed_leaks,
        # _enforce_level_policy, anneal commit, AND two-thread store access (the test thread polls
        # get_leaks while the daemon thread seeds — P4b-ii-a thread-safety under the real loop).
        with tempfile.TemporaryDirectory() as tmp:
            ls = self._ls_with_reads(tmp, 6)
            ids = self._ids(ls, 3)
            doc = _leak_doc("barrel_into_the_caller", 1, ids,
                            "river value barrel into a caller who only calls")
            # the author runner MUST accept the timeout kwarg the gate's functools.partial passes (MED-1).
            runner = lambda prompt, timeout=None: doc
            orig_thr, orig_int = coach_daemon.LEAK_SEED_THRESHOLD, coach_daemon.LEAK_SEED_MIN_INTERVAL
            coach_daemon.LEAK_SEED_THRESHOLD, coach_daemon.LEAK_SEED_MIN_INTERVAL = 3, 0.0
            stop = threading.Event()
            try:
                t = threading.Thread(target=lambda: coach_daemon.run(
                    once=False, interval=0.01, runner=runner, db=Path(tmp) / "coach.db",
                    startup_probe=False, stop_event=stop, install_signal_handlers=False, leak_store=ls))
                t.start()
                deadline = time.time() + 8
                while time.time() < deadline and "barrel_into_the_caller" not in (ls.get_leaks() or ""):
                    time.sleep(0.05)
                stop.set()
                t.join(timeout=5)
                self.assertIn("barrel_into_the_caller", ls.get_leaks() or "")   # the loop actually seeded
                hits = ls.recall_relevant_leaks("river value barrel into a caller", limit=2)
                self.assertTrue(any(h["name"] == "barrel_into_the_caller" for h in hits), hits)
            finally:
                stop.set()
                coach_daemon.LEAK_SEED_THRESHOLD, coach_daemon.LEAK_SEED_MIN_INTERVAL = orig_thr, orig_int
                ls.close()

    def test_seed_skips_when_no_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            ls = LeakStore(Path(tmp) / "leaks.db")
            try:
                called = []
                outcome = coach_daemon.seed_leaks(ls, runner=lambda _p: (called.append(1) or "x"))
                self.assertFalse(outcome.get("seeded"))
                self.assertEqual(called, [])               # never called the author — nothing to graduate
            finally:
                ls.close()

    def test_overlevel_new_leak_capped_through_seed(self):
        # codex L3 HIGH-2 end-to-end: an author that CLAIMS 3x on a first-time leak gets it committed at
        # 1x — the structural level cap fires before commit, so an unearned graduation can't be minted.
        with tempfile.TemporaryDirectory() as tmp:
            ls = self._ls_with_reads(tmp, 3)
            try:
                ids = self._ids(ls, 3)
                doc = _leak_doc("over_leveled", 3, ids, "river value barrel into a caller who only calls")
                self.assertTrue(coach_daemon.seed_leaks(ls, runner=lambda _p: doc).get("seeded"))
                line = next(l for l in ls.get_leaks().splitlines() if "over_leveled" in l)
                self.assertIn("| 1x", line)
                self.assertNotIn("| 3x", line)
            finally:
                ls.close()

    def test_seed_recovers_an_abandoned_wrap(self):
        # L2 Finding 6: a prior pass left a wrap OPEN (its cancel also failed). The next seed must reclaim
        # it (recover_abandoned) — the daemon is the sole in-process wrapper, so a stuck wrap is its own —
        # else one swallowed cancel-failure bricks graduation for the process lifetime.
        with tempfile.TemporaryDirectory() as tmp:
            ls = self._ls_with_reads(tmp, 3)
            try:
                ls.prepare_leak_wrap()                                       # open + abandon (never commit)
                self.assertEqual(ls.prepare_leak_wrap().get("status"), "wrap_in_progress")  # confirm stuck
                ids = self._ids(ls, 3)
                doc = _leak_doc("barrel_into_the_caller", 1, ids,
                                "river value barrel into a caller who only calls")
                outcome = coach_daemon.seed_leaks(ls, runner=lambda _p: doc)
                self.assertTrue(outcome.get("seeded"), outcome)             # recovered the wrap + committed
                self.assertIn("barrel_into_the_caller", ls.get_leaks())
            finally:
                ls.close()

    def test_junk_author_cancels_the_wrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            ls = self._ls_with_reads(tmp, 3)
            try:
                # author returns junk (no "## Patterns") → seed_leaks must CANCEL the wrap, not leave it
                # open (an abandoned wrap would brick the next pass — the single-writer dead-store mode).
                outcome = coach_daemon.seed_leaks(ls, runner=lambda _p: "sorry, I can't help with that")
                self.assertFalse(outcome.get("seeded"))
                wrap = ls.prepare_leak_wrap()              # a fresh prepare is READY again (wrap released)
                self.assertEqual(wrap.get("status"), "ready")
                ls.cancel_leak_wrap()                      # clean up the probe wrap we just opened
            finally:
                ls.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
