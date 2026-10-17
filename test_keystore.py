"""gutcheck metered-worker suite — the BYO-API-key store, the call_anthropic_api worker, and the
/api/secrets HTTP surface (the third coach provider, 'anthropic_api').

Three layers, mirroring how test_settings.py covers the settings store + its route:
  - TestKeystore     — keystore.py: round-trip, the SAFE status (never the key), 0600 perms, tolerance
                       of a missing/corrupt file, the env override, and clear() semantics.
  - TestMeteredWorker — coach_daemon.call_anthropic_api with the `anthropic` SDK MOCKED (the dep is an
                       optional extra, absent in the test env): success / refusal / API-error / no-key /
                       ImportError all degrade to '' without raising, and the key is passed EXPLICITLY.
  - TestSecretsRoute — /api/secrets GET (status) + POST (set/clear) over a real in-process server, and
                       the invariant the whole design exists for: the key never appears in any payload.

    ./venv/bin/python -m pytest test_keystore.py -q
"""

import json
import os
import stat
import sys
import tempfile
import threading
import time
import types
import unittest
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

from gutcheck import coach_daemon
from gutcheck import game as G
from gutcheck import keystore
from gutcheck import paths
from gutcheck import seam as seam_mod
from gutcheck import server
from gutcheck import settings


_ENV = "GUTCHECK_ANTHROPIC_API_KEY"
_OAI_ENV = "GUTCHECK_OPENAI_API_KEY"


class _TmpDir(unittest.TestCase):
    """Isolate the data dir + the anthropic env override so no test touches ~/.gutcheck or a real key."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._prev_dir = os.environ.get(paths.ENV_VAR)
        os.environ[paths.ENV_VAR] = self.tmp
        self._prev_env = os.environ.pop(_ENV, None)              # start with no env-provided key
        self._prev_model = os.environ.pop("GUTCHECK_API_MODEL", None)   # …and no env model override

    def tearDown(self):
        if self._prev_dir is None:
            os.environ.pop(paths.ENV_VAR, None)
        else:
            os.environ[paths.ENV_VAR] = self._prev_dir
        if self._prev_env is None:
            os.environ.pop(_ENV, None)
        else:
            os.environ[_ENV] = self._prev_env
        if self._prev_model is None:
            os.environ.pop("GUTCHECK_API_MODEL", None)
        else:
            os.environ["GUTCHECK_API_MODEL"] = self._prev_model

    def _file(self) -> Path:
        return Path(self.tmp) / "secrets.json"


class TestKeystore(_TmpDir):
    def test_absent_is_unset(self):
        self.assertIsNone(keystore.get_api_key("anthropic"))
        self.assertEqual(keystore.key_status("anthropic"), {"set": False, "hint": "", "source": ""})

    def test_set_roundtrips_and_status(self):
        keystore.set_api_key("sk-ant-abcdEFGH1234", "anthropic")
        self.assertEqual(keystore.get_api_key("anthropic"), "sk-ant-abcdEFGH1234")
        self.assertEqual(keystore.key_status("anthropic"),
                         {"set": True, "hint": "…1234", "source": "file"})

    def test_openai_provider_roundtrips_independently(self):
        # the second metered vendor: openai keys store/read under their own provider slot, isolated from
        # anthropic (one secrets.json, keyed by provider — clearing one never touches the other).
        keystore.set_api_key("sk-ant-aaaa1111", "anthropic")
        keystore.set_api_key("sk-proj-oaiKEY9999", "openai")
        self.assertEqual(keystore.get_api_key("openai"), "sk-proj-oaiKEY9999")
        self.assertEqual(keystore.key_status("openai"), {"set": True, "hint": "…9999", "source": "file"})
        keystore.clear_api_key("openai")
        self.assertIsNone(keystore.get_api_key("openai"))                 # openai cleared…
        self.assertEqual(keystore.get_api_key("anthropic"), "sk-ant-aaaa1111")   # …anthropic untouched

    def test_openai_env_override(self):
        os.environ["GUTCHECK_OPENAI_API_KEY"] = "sk-proj-fromENV"
        try:
            self.assertEqual(keystore.get_api_key("openai"), "sk-proj-fromENV")
            self.assertEqual(keystore.key_status("openai")["source"], "env")
        finally:
            os.environ.pop("GUTCHECK_OPENAI_API_KEY", None)

    def test_status_never_contains_the_key(self):
        keystore.set_api_key("sk-ant-SUPERSECRETvalue", "anthropic")
        blob = json.dumps(keystore.key_status("anthropic"))
        self.assertNotIn("SUPERSECRET", blob)
        self.assertNotIn("sk-ant-SUPERSECRETvalue", blob)

    def test_file_is_owner_only_0600(self):
        keystore.set_api_key("sk-ant-perms", "anthropic")
        mode = stat.S_IMODE(os.stat(self._file()).st_mode)
        self.assertEqual(mode, 0o600, oct(mode))

    def test_short_value_hints_nothing(self):
        # a stray short value (never a real key) must not echo in full via the hint
        keystore.set_api_key("ab12", "anthropic")
        self.assertEqual(keystore.key_status("anthropic")["hint"], "")

    def test_blank_and_unknown_rejected(self):
        with self.assertRaises(ValueError):
            keystore.set_api_key("   ", "anthropic")      # blank → reject (don't store an empty key)
        with self.assertRaises(ValueError):
            keystore.set_api_key("sk-x", "groq")          # unknown provider (anthropic+openai are valid)
        with self.assertRaises(ValueError):
            keystore.clear_api_key("groq")

    def test_corrupt_and_nonobject_tolerated(self):
        paths.ensure_data_dir()
        self._file().write_text("{not json", encoding="utf-8")
        self.assertIsNone(keystore.get_api_key("anthropic"))   # malformed → no key, never raises
        self._file().write_text('["a","list"]', encoding="utf-8")
        self.assertIsNone(keystore.get_api_key("anthropic"))   # non-object → no key

    def test_env_overrides_file_and_clear_cannot_unset_env(self):
        keystore.set_api_key("sk-ant-FILEvalue", "anthropic")
        os.environ[_ENV] = "sk-ant-ENVvalue"
        self.assertEqual(keystore.get_api_key("anthropic"), "sk-ant-ENVvalue")   # env wins
        self.assertEqual(keystore.key_status("anthropic")["source"], "env")
        keystore.clear_api_key("anthropic")                                       # clears the FILE only
        self.assertEqual(keystore.get_api_key("anthropic"), "sk-ant-ENVvalue")    # env still effective
        os.environ.pop(_ENV)
        self.assertIsNone(keystore.get_api_key("anthropic"))                      # file really was cleared

    def test_blank_env_is_ignored(self):
        os.environ[_ENV] = "   "
        self.assertIsNone(keystore.get_api_key("anthropic"))   # whitespace env → treated as unset

    def test_clear_is_idempotent(self):
        keystore.clear_api_key("anthropic")                    # absent → no-op, no raise
        keystore.set_api_key("sk-ant-x123", "anthropic")
        keystore.clear_api_key("anthropic")
        self.assertIsNone(keystore.get_api_key("anthropic"))
        keystore.clear_api_key("anthropic")                    # second clear → still a no-op

    def test_set_strips_whitespace(self):
        keystore.set_api_key("  sk-ant-trim99  ", "anthropic")
        self.assertEqual(keystore.get_api_key("anthropic"), "sk-ant-trim99")

    def test_stale_temp_does_not_block_save(self):
        # L2 LOW: a crash-orphaned temp (it matches reset_data's *.tmp sweep) must NOT break a save.
        # The unique mkstemp name can't collide with it, so set_api_key just succeeds — the fix removes
        # the FileExistsError class structurally rather than catching it.
        paths.ensure_data_dir()
        (Path(self.tmp) / "secrets.orphan.tmp").write_text("stale", encoding="utf-8")
        keystore.set_api_key("sk-ant-after-orphan", "anthropic")          # must not raise
        self.assertEqual(keystore.get_api_key("anthropic"), "sk-ant-after-orphan")

    def test_overwrite_preserves_0600(self):
        # replacing an existing (even widened) secrets.json must still end owner-only — os.replace
        # carries the temp's 0600 mode onto the final inode, so a re-save can't silently widen perms.
        keystore.set_api_key("sk-ant-first", "anthropic")
        os.chmod(self._file(), 0o644)                                     # simulate a widened file
        keystore.set_api_key("sk-ant-second", "anthropic")               # re-save over it
        self.assertEqual(stat.S_IMODE(os.stat(self._file()).st_mode), 0o600)
        self.assertEqual(keystore.get_api_key("anthropic"), "sk-ant-second")


# ---- a fake `anthropic` SDK so the worker is testable without the optional dep installed ----
def _fake_anthropic(behavior, captured=None):
    """Build a stand-in `anthropic` module. ``behavior(Block, Resp)`` returns the messages.create result
    (or raises to simulate an API error). ``captured`` (a dict) records the Anthropic(**kwargs) so a test
    can assert the key is passed EXPLICITLY + max_retries=0."""
    m = types.ModuleType("anthropic")

    class _Block:
        def __init__(self, type, text=""):
            self.type = type
            self.text = text

    class _Resp:
        def __init__(self, content, stop_reason="end_turn"):
            self.content = content
            self.stop_reason = stop_reason

    class _Messages:
        def create(self, **kw):
            if captured is not None:
                captured["_create"] = kw          # record the messages.create(**kw) for dial assertions
            return behavior(_Block, _Resp)

    class Anthropic:
        def __init__(self, **kw):
            if captured is not None:
                captured.update(kw)
                # the real SDK reads ANTHROPIC_CUSTOM_HEADERS at construction — record what it would see,
                # so a test can prove the worker DENIES the ambient channel for the constructor window.
                captured["_custom_headers_env"] = os.environ.get("ANTHROPIC_CUSTOM_HEADERS")
                # likewise the credential vector: the SDK resolves ANTHROPIC_AUTH_TOKEN at construction —
                # record it so a test can prove the worker strips the ambient token for the build window.
                captured["_auth_token_env"] = os.environ.get("ANTHROPIC_AUTH_TOKEN")
                captured["_client"] = self
            self.messages = _Messages()
            self.closed = False

        def __enter__(self):                 # the worker uses `with anthropic.Anthropic(...) as client`
            return self

        def __exit__(self, *exc):
            self.closed = True
            return False

    m.Anthropic = Anthropic
    return m


class TestMeteredWorker(_TmpDir):
    def setUp(self):
        super().setUp()
        self._prev_ant_base = os.environ.pop("GUTCHECK_ANTHROPIC_BASE_URL", None)
        self._prev_ant_hdrs = os.environ.pop("ANTHROPIC_CUSTOM_HEADERS", None)
        # fresh in-flight gate per test: a prior overrun test's abandoned thread still holds the OLD lock
        # (it releases the lock it captured locally), so a new lock object isolates this test cleanly.
        coach_daemon._METERED_INFLIGHT = threading.Lock()

    def tearDown(self):
        sys.modules.pop("anthropic", None)   # never leak the fake across tests/files
        if self._prev_ant_base is not None:
            os.environ["GUTCHECK_ANTHROPIC_BASE_URL"] = self._prev_ant_base
        if self._prev_ant_hdrs is not None:
            os.environ["ANTHROPIC_CUSTOM_HEADERS"] = self._prev_ant_hdrs
        super().tearDown()

    def test_runner_routes_to_metered(self):
        self.assertIs(coach_daemon._runner_for("anthropic_api"), coach_daemon.call_anthropic_api)

    def test_no_key_degrades_without_importing_sdk(self):
        # absent key → '' and the SDK is never even imported (a poisoned import would prove it was)
        sys.modules["anthropic"] = _fake_anthropic(lambda B, R: (_ for _ in ()).throw(AssertionError("imported")))
        self.assertEqual(coach_daemon.call_anthropic_api("spot", timeout=1), "")

    def test_success_joins_text_blocks_only(self):
        keystore.set_api_key("sk-ant-key9", "anthropic")
        sys.modules["anthropic"] = _fake_anthropic(
            lambda B, R: R([B("text", "Bet bigger."), B("thinking", "x"), B("text", " She calls.")]))
        self.assertEqual(coach_daemon.call_anthropic_api("spot"), "Bet bigger. She calls.")

    def test_key_passed_explicitly_and_no_retries(self):
        keystore.set_api_key("sk-ant-EXPLICIT", "anthropic")
        cap = {}
        sys.modules["anthropic"] = _fake_anthropic(lambda B, R: R([B("text", "ok")]), captured=cap)
        coach_daemon.call_anthropic_api("spot", timeout=7)
        self.assertEqual(cap.get("api_key"), "sk-ant-EXPLICIT")   # never the ambient ANTHROPIC_API_KEY
        self.assertEqual(cap.get("base_url"), "https://api.anthropic.com")  # never the ambient ANTHROPIC_BASE_URL
        self.assertEqual(cap.get("max_retries"), 0)               # bound wall-clock to ~timeout
        self.assertEqual(cap.get("timeout"), 7)

    def test_ambient_anthropic_base_url_is_never_honored(self):
        # codex HIGH (sibling openai worker): the metered key must NEVER be shipped to whatever
        # ANTHROPIC_BASE_URL the daemon inherited. Without an explicit gutcheck opt-in, base_url is forced
        # to the official endpoint, ambient ignored.
        keystore.set_api_key("sk-ant-key9", "anthropic")
        os.environ["ANTHROPIC_BASE_URL"] = "https://evil.example.com"        # ambient — must be IGNORED
        try:
            cap = {}
            sys.modules["anthropic"] = _fake_anthropic(lambda B, R: R([B("text", "ok")]), captured=cap)
            coach_daemon.call_anthropic_api("spot")
            self.assertEqual(cap.get("base_url"), "https://api.anthropic.com")  # official default, not the ambient
        finally:
            os.environ.pop("ANTHROPIC_BASE_URL", None)

    def test_explicit_gutcheck_base_url_override_is_honored(self):
        # the legit-endpoint escape hatch (gateway / proxy): only an EXPLICIT GUTCHECK_ANTHROPIC_BASE_URL redirects.
        keystore.set_api_key("sk-ant-key9", "anthropic")
        os.environ["ANTHROPIC_BASE_URL"] = "https://evil.example.com"               # ambient still ignored…
        os.environ["GUTCHECK_ANTHROPIC_BASE_URL"] = "https://gateway.example.com"  # …explicit opt-in wins
        try:
            cap = {}
            sys.modules["anthropic"] = _fake_anthropic(lambda B, R: R([B("text", "ok")]), captured=cap)
            coach_daemon.call_anthropic_api("spot")
            self.assertEqual(cap.get("base_url"), "https://gateway.example.com")
        finally:
            os.environ.pop("ANTHROPIC_BASE_URL", None)
            os.environ.pop("GUTCHECK_ANTHROPIC_BASE_URL", None)

    def test_ambient_custom_headers_denied_during_construction(self):
        # codex (sibling openai worker): the SDK reads ANTHROPIC_CUSTOM_HEADERS at construction and spreads
        # those headers OVER the auth header — an inherited value could override x-api-key on the billed call.
        # The metered worker must DENY it for the constructor window AND restore it afterward.
        keystore.set_api_key("sk-ant-key9", "anthropic")
        os.environ["ANTHROPIC_CUSTOM_HEADERS"] = "x-api-key: evil-override"   # ambient — must be DENIED at build
        try:
            cap = {}
            sys.modules["anthropic"] = _fake_anthropic(lambda B, R: R([B("text", "ok")]), captured=cap)
            coach_daemon.call_anthropic_api("spot")
            self.assertIsNone(cap.get("_custom_headers_env"))                 # denied during construction
            self.assertEqual(os.environ.get("ANTHROPIC_CUSTOM_HEADERS"), "x-api-key: evil-override")  # restored after
        finally:
            os.environ.pop("ANTHROPIC_CUSTOM_HEADERS", None)

    def test_ambient_auth_token_denied_during_construction(self):
        # the lone metered LOW: an inherited ANTHROPIC_AUTH_TOKEN must NOT authenticate the operator's billed
        # call as a different account. In anthropic 0.112 the explicit api_key= already makes the SDK skip all
        # credential env vars ("explicit ctor args are total"), so this is belt-and-suspenders — but the worker
        # strips the var for the build window so hermeticity is STRUCTURAL, not dependent on that SDK precedence.
        keystore.set_api_key("sk-ant-key9", "anthropic")
        os.environ["ANTHROPIC_AUTH_TOKEN"] = "evil-bearer-token"   # ambient — must be DENIED at build
        try:
            cap = {}
            sys.modules["anthropic"] = _fake_anthropic(lambda B, R: R([B("text", "ok")]), captured=cap)
            coach_daemon.call_anthropic_api("spot")
            self.assertIsNone(cap.get("_auth_token_env"))                     # the ENV route: denied during construction (the real proof)
            self.assertEqual(cap.get("api_key"), "sk-ant-key9")              # only the explicit key authenticates
            self.assertIsNone(cap.get("auth_token"))                          # the KWARG route: worker never passes auth_token= (guards a future regression)
            self.assertEqual(os.environ.get("ANTHROPIC_AUTH_TOKEN"), "evil-bearer-token")  # restored after
        finally:
            os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

    def test_client_is_closed_on_success(self):
        # the new `client = Ctor(...)` / `with client:` split (the deny-block hardening) must still close
        # the transport on success — parity with the openai sibling's close tests.
        keystore.set_api_key("sk-ant-key9", "anthropic")
        cap = {}
        sys.modules["anthropic"] = _fake_anthropic(lambda B, R: R([B("text", "ok")]), captured=cap)
        coach_daemon.call_anthropic_api("spot")
        self.assertTrue(cap["_client"].closed)                       # the `with client:` closes on success

    def test_client_is_closed_on_api_error(self):
        # …and on the ERROR branch: if create() raises, the split's `with client:` must STILL close the
        # transport — otherwise an error-storm leaks an httpx client per failed call.
        keystore.set_api_key("sk-ant-key9", "anthropic")
        cap = {}
        def boom(B, R):
            raise RuntimeError("503 overloaded")
        sys.modules["anthropic"] = _fake_anthropic(boom, captured=cap)
        self.assertEqual(coach_daemon.call_anthropic_api("spot"), "")  # degraded
        self.assertTrue(cap["_client"].closed)                        # …and the transport was still closed

    def test_refusal_returns_empty(self):
        keystore.set_api_key("sk-ant-key9", "anthropic")
        sys.modules["anthropic"] = _fake_anthropic(lambda B, R: R([], stop_reason="refusal"))
        self.assertEqual(coach_daemon.call_anthropic_api("spot"), "")

    def test_max_tokens_truncation_degrades_not_served(self):
        # L1 MED: anthropic counts THINKING tokens against max_tokens, so a high-effort read can stop on
        # 'max_tokens' with a TRUNCATED, mid-sentence answer. That must NOT be served as coaching — it
        # degrades to the facts floor (the end_turn whitelist, parity with the openai finish_reason screen).
        keystore.set_api_key("sk-ant-key9", "anthropic")
        sys.modules["anthropic"] = _fake_anthropic(
            lambda B, R: R([B("text", "You should bet because the range is so wide that—")], stop_reason="max_tokens"))
        self.assertEqual(coach_daemon.call_anthropic_api("spot"), "")

    def test_max_tokens_budget_is_generous_not_2048(self):
        # the budget bump half of the L1 MED: 2048 starved adaptive thinking; the worker now sends the
        # generous _ANTHROPIC_MAX_TOKENS so a high-effort read has room to think AND answer.
        keystore.set_api_key("sk-ant-key9", "anthropic")
        cap = {}
        sys.modules["anthropic"] = _fake_anthropic(lambda B, R: R([B("text", "ok")]), captured=cap)
        coach_daemon.call_anthropic_api("spot")
        self.assertEqual(cap["_create"]["max_tokens"], coach_daemon._ANTHROPIC_MAX_TOKENS)
        self.assertGreater(coach_daemon._ANTHROPIC_MAX_TOKENS, 2048)

    def test_unexpected_terminal_fails_closed(self):
        # the whitelist's fail-closed property (L1): ANY non-end_turn terminal degrades, even one we never
        # expect for this call shape and even WITH content present — so a future SDK terminal can't leak
        # unvetted text as coaching. (None is also covered: None != "end_turn".)
        keystore.set_api_key("sk-ant-key9", "anthropic")
        for stop in ("pause_turn", "stop_sequence", None):
            sys.modules["anthropic"] = _fake_anthropic(
                lambda B, R, _s=stop: R([B("text", "unvetted text")], stop_reason=_s))
            self.assertEqual(coach_daemon.call_anthropic_api("spot"), "", f"stop_reason={stop!r} must degrade")

    def test_api_error_never_raises(self):
        keystore.set_api_key("sk-ant-key9", "anthropic")
        def boom(B, R):
            raise RuntimeError("401 authentication_error")
        sys.modules["anthropic"] = _fake_anthropic(boom)
        self.assertEqual(coach_daemon.call_anthropic_api("spot"), "")   # degraded, not crashed

    def test_missing_sdk_degrades(self):
        keystore.set_api_key("sk-ant-key9", "anthropic")
        sys.modules.pop("anthropic", None)
        import builtins
        real = builtins.__import__
        def no_anthropic(name, *a, **k):
            if name == "anthropic":
                raise ImportError("no module named anthropic")
            return real(name, *a, **k)
        builtins.__import__ = no_anthropic
        try:
            self.assertEqual(coach_daemon.call_anthropic_api("spot"), "")
        finally:
            builtins.__import__ = real

    def test_watchdog_abandons_a_call_that_overruns_the_budget(self):
        # L3 (3 substrates): anthropic's timeout= is per-operation, not wall-clock. The watchdog must
        # cap TOTAL elapsed at ~`timeout` and serve "" — never let a slow-drip response wedge the daemon.
        import threading as _t
        keystore.set_api_key("sk-ant-key9", "anthropic")

        def slow(B, R):
            _t.Event().wait(5.0)               # block well past the budget (never set → ~full 5s)
            return R([B("text", "too late")])
        sys.modules["anthropic"] = _fake_anthropic(slow)
        t0 = time.monotonic()
        out = coach_daemon.call_anthropic_api("spot", timeout=0.3)
        elapsed = time.monotonic() - t0
        self.assertEqual(out, "")                                   # abandoned → facts floor
        self.assertLess(elapsed, 2.0, f"watchdog didn't bound the call: {elapsed:.2f}s")

    def test_inflight_gate_blocks_a_second_concurrent_call(self):
        # codex MED: while one metered call is in flight (gate held), a new call must NOT start a 2nd
        # billable call — it degrades to the facts floor WITHOUT even constructing the client.
        keystore.set_api_key("sk-ant-key9", "anthropic")
        coach_daemon._METERED_INFLIGHT.acquire()                    # simulate a prior call still in flight
        try:
            cap = {}
            sys.modules["anthropic"] = _fake_anthropic(lambda B, R: R([B("text", "ok")]), captured=cap)
            self.assertEqual(coach_daemon.call_anthropic_api("spot"), "")   # facts floor
            self.assertEqual(cap, {})                                       # client never built (gate short-circuit)
        finally:
            coach_daemon._METERED_INFLIGHT.release()

    def test_inflight_gate_released_after_a_normal_call(self):
        # the gate MUST be released when the call finishes, or every call after the first would be blocked.
        keystore.set_api_key("sk-ant-key9", "anthropic")
        sys.modules["anthropic"] = _fake_anthropic(lambda B, R: R([B("text", "ok")]))
        self.assertEqual(coach_daemon.call_anthropic_api("spot"), "ok")
        self.assertTrue(coach_daemon._METERED_INFLIGHT.acquire(blocking=False))  # released → acquirable
        coach_daemon._METERED_INFLIGHT.release()

    def test_real_overrun_holds_gate_until_the_abandoned_worker_finishes(self):
        # the REAL lifecycle (codex/complement/L1), not a manually-held lock: a call that OVERRUNS is
        # abandoned but its worker keeps draining and KEEPS THE GATE — so a 2nd call short-circuits to the
        # facts floor (client never built) until the abandoned worker actually finishes and releases.
        keystore.set_api_key("sk-ant-key9", "anthropic")
        release = threading.Event()

        def slow(B, R):
            release.wait(2.0)                          # hold the worker (→ overrun) until we let it finish
            return R([B("text", "late")])
        sys.modules["anthropic"] = _fake_anthropic(slow)
        self.assertEqual(coach_daemon.call_anthropic_api("spot", timeout=0.05), "")   # overrun → abandoned
        try:
            cap = {}                                   # 2nd call while the abandoned worker still drains
            sys.modules["anthropic"] = _fake_anthropic(lambda B, R: R([B("text", "second")]), captured=cap)
            self.assertEqual(coach_daemon.call_anthropic_api("spot", timeout=0.05), "")   # gated → facts floor
            self.assertEqual(cap, {})                  # the abandoned worker held the gate → client never built
        finally:
            release.set()                              # let the abandoned worker finish + release the gate

    def test_effort_sent_for_capable_model(self):
        keystore.set_api_key("sk-ant-key9", "anthropic")
        settings.apply_patch({"api_model": "claude-opus-4-8", "api_effort": "high"})
        cap = {}
        sys.modules["anthropic"] = _fake_anthropic(lambda B, R: R([B("text", "ok")]), captured=cap)
        coach_daemon.call_anthropic_api("spot")
        cc = cap["_create"]
        self.assertEqual(cc["model"], "claude-opus-4-8")
        self.assertEqual(cc.get("output_config"), {"effort": "high"})     # the fidelity dial
        self.assertEqual(cc.get("thinking"), {"type": "adaptive"})        # paired so effort drives depth

    def test_default_low_effort_is_sent_on_capable_model(self):
        # the DEFAULT metered call (no apply_patch → opus-4-8 + low) MUST carry effort+thinking — this is
        # the path the [api] floor bump protects (a too-old SDK would reject these kwargs by default).
        keystore.set_api_key("sk-ant-key9", "anthropic")
        cap = {}
        sys.modules["anthropic"] = _fake_anthropic(lambda B, R: R([B("text", "ok")]), captured=cap)
        coach_daemon.call_anthropic_api("spot")
        cc = cap["_create"]
        self.assertEqual(cc["model"], "claude-opus-4-8")
        self.assertEqual(cc.get("output_config"), {"effort": "low"})
        self.assertEqual(cc.get("thinking"), {"type": "adaptive"})

    def test_haiku_never_gets_effort_even_when_dialed_up(self):
        keystore.set_api_key("sk-ant-key9", "anthropic")
        settings.apply_patch({"api_model": "claude-haiku-4-5", "api_effort": "high"})
        cap = {}
        sys.modules["anthropic"] = _fake_anthropic(lambda B, R: R([B("text", "ok")]), captured=cap)
        coach_daemon.call_anthropic_api("spot")
        cc = cap["_create"]
        self.assertEqual(cc["model"], "claude-haiku-4-5")
        self.assertNotIn("output_config", cc)                             # capability-gated — Haiku 400s on effort
        self.assertNotIn("thinking", cc)

    def test_xhigh_sent_for_opus(self):
        # xhigh is Opus-4.7+ only; on Opus it must reach output_config.effort unchanged.
        keystore.set_api_key("sk-ant-key9", "anthropic")
        settings.apply_patch({"api_model": "claude-opus-4-8", "api_effort": "xhigh"})
        cap = {}
        sys.modules["anthropic"] = _fake_anthropic(lambda B, R: R([B("text", "ok")]), captured=cap)
        coach_daemon.call_anthropic_api("spot")
        cc = cap["_create"]
        self.assertEqual(cc["model"], "claude-opus-4-8")
        self.assertEqual(cc.get("output_config"), {"effort": "xhigh"})     # Opus accepts the deepest level
        self.assertEqual(cc.get("thinking"), {"type": "adaptive"})

    def test_xhigh_degrades_to_high_on_sonnet(self):
        # Sonnet 4.6 takes effort but 400s on xhigh (Opus-only) — a persisted xhigh must degrade to high
        # (the deepest level Sonnet DOES support), never reach the SDK as xhigh.
        keystore.set_api_key("sk-ant-key9", "anthropic")
        settings.apply_patch({"api_model": "claude-sonnet-4-6", "api_effort": "xhigh"})
        cap = {}
        sys.modules["anthropic"] = _fake_anthropic(lambda B, R: R([B("text", "ok")]), captured=cap)
        coach_daemon.call_anthropic_api("spot")
        cc = cap["_create"]
        self.assertEqual(cc["model"], "claude-sonnet-4-6")
        self.assertEqual(cc.get("output_config"), {"effort": "high"})      # degraded from xhigh, not sent as-is
        self.assertEqual(cc.get("thinking"), {"type": "adaptive"})

    def test_xhigh_on_haiku_is_a_plain_call(self):
        # the novel degrade path on a NON-effort model: xhigh degrades to high, then the _EFFORT_MODELS gate
        # drops it entirely (Haiku ∉ _EFFORT_MODELS) → plain call, no effort/thinking, never a 400.
        keystore.set_api_key("sk-ant-key9", "anthropic")
        settings.apply_patch({"api_model": "claude-haiku-4-5", "api_effort": "xhigh"})
        cap = {}
        sys.modules["anthropic"] = _fake_anthropic(lambda B, R: R([B("text", "ok")]), captured=cap)
        coach_daemon.call_anthropic_api("spot")
        cc = cap["_create"]
        self.assertEqual(cc["model"], "claude-haiku-4-5")
        self.assertNotIn("output_config", cc)                             # degrade→high then dropped: plain call
        self.assertNotIn("thinking", cc)

    def test_xhigh_models_are_a_subset_of_effort_models(self):
        # the load-bearing invariant (now structurally derived via intersection): every xhigh-capable model
        # MUST also be effort-capable, else xhigh could be sent to a model the _EFFORT_MODELS gate excludes.
        self.assertTrue(coach_daemon._XHIGH_EFFORT_MODELS <= coach_daemon._EFFORT_MODELS)

    def test_env_overrides_model(self):
        keystore.set_api_key("sk-ant-key9", "anthropic")
        settings.apply_patch({"api_model": "claude-sonnet-4-6"})
        os.environ["GUTCHECK_API_MODEL"] = "claude-opus-4-8"
        try:
            cap = {}
            sys.modules["anthropic"] = _fake_anthropic(lambda B, R: R([B("text", "ok")]), captured=cap)
            coach_daemon.call_anthropic_api("spot")
            self.assertEqual(cap["_create"]["model"], "claude-opus-4-8")  # env override wins over the UI dial
        finally:
            os.environ.pop("GUTCHECK_API_MODEL", None)

    def test_opus_4_5_excluded_from_effort(self):
        # opus-4-5 supports output_config.effort but NOT adaptive thinking (4.6+) — pairing them would be an
        # invalid request (L3 codex). _EFFORT_MODELS excludes it, so an env-override to it gets a plain call.
        keystore.set_api_key("sk-ant-key9", "anthropic")
        os.environ["GUTCHECK_API_MODEL"] = "claude-opus-4-5"
        try:
            cap = {}
            sys.modules["anthropic"] = _fake_anthropic(lambda B, R: R([B("text", "ok")]), captured=cap)
            coach_daemon.call_anthropic_api("spot")
            cc = cap["_create"]
            self.assertEqual(cc["model"], "claude-opus-4-5")
            self.assertNotIn("output_config", cc)                        # not in the adaptive+effort set
            self.assertNotIn("thinking", cc)
        finally:
            os.environ.pop("GUTCHECK_API_MODEL", None)

    def test_error_log_is_type_name_only_no_key(self):
        # the failure log must carry the exception CLASS, never str(e) (which a future SDK could use to
        # echo a key fragment). Capture stderr and assert: key absent, class present.
        import io
        import contextlib
        keystore.set_api_key("sk-ant-SECRETLOGVALUE", "anthropic")
        def boom(B, R):
            raise RuntimeError("sk-ant-SECRETLOGVALUE leaked into the message")  # SDK that echoes the key
        sys.modules["anthropic"] = _fake_anthropic(boom)
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            self.assertEqual(coach_daemon.call_anthropic_api("spot"), "")
        log = buf.getvalue()
        self.assertIn("RuntimeError", log)                          # the class diagnoses the failure
        self.assertNotIn("SECRETLOGVALUE", log)                     # …and str(e) never reaches the log


# ---- a fake `openai` SDK so the openai_api worker is testable without the optional dep installed ----
def _fake_openai(behavior, captured=None):
    """Build a stand-in `openai` module. ``behavior(Msg, Choice, Resp)`` returns the chat.completions.create
    result (or raises to simulate an API error). ``captured`` records OpenAI(**kwargs) [api_key/timeout/
    max_retries], the create(**kwargs) under '_create', and the client instance under '_client' (so a test
    can assert .close() ran)."""
    m = types.ModuleType("openai")

    class _Message:
        def __init__(self, content="", refusal=None):
            self.content = content
            self.refusal = refusal

    class _Choice:
        def __init__(self, message, finish_reason="stop"):
            self.message = message
            self.finish_reason = finish_reason

    class _Resp:
        def __init__(self, choices):
            self.choices = choices

    class _Completions:
        def create(self, **kw):
            if captured is not None:
                captured["_create"] = kw
            return behavior(_Message, _Choice, _Resp)

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class OpenAI:
        def __init__(self, **kw):
            if captured is not None:
                captured.update(kw)
                captured["_client"] = self
                # the real SDK reads OPENAI_CUSTOM_HEADERS at construction — record what it would see.
                captured["_custom_headers_env"] = os.environ.get("OPENAI_CUSTOM_HEADERS")
            self.chat = _Chat()
            self.closed = False

        def close(self):
            self.closed = True

        def __enter__(self):                      # the worker uses `with openai.OpenAI(...) as client`
            return self

        def __exit__(self, *exc):                 # closes on BOTH the success and the exception path
            self.close()
            return False

    m.OpenAI = OpenAI
    return m


class TestOpenAIMeteredWorker(_TmpDir):
    """coach_daemon.call_openai_api with the `openai` SDK MOCKED — the SIBLING of TestMeteredWorker. Same
    contract: success / refusal / truncation / API-error / no-key / ImportError all degrade to '' without
    raising; the key is passed EXPLICITLY; reasoning_effort is sent (gpt-5 only) and never max_tokens/
    temperature; the watchdog bounds a slow call."""

    def setUp(self):
        super().setUp()
        self._prev_oai_env = os.environ.pop("GUTCHECK_OPENAI_API_KEY", None)
        self._prev_oai_model = os.environ.pop("GUTCHECK_OPENAI_MODEL", None)
        self._prev_oai_base = os.environ.pop("GUTCHECK_OPENAI_BASE_URL", None)
        self._prev_oai_hdrs = os.environ.pop("OPENAI_CUSTOM_HEADERS", None)
        # fresh in-flight gate per test (see TestMeteredWorker.setUp).
        coach_daemon._METERED_INFLIGHT = threading.Lock()

    def tearDown(self):
        sys.modules.pop("openai", None)           # never leak the fake across tests/files
        if self._prev_oai_env is not None:
            os.environ["GUTCHECK_OPENAI_API_KEY"] = self._prev_oai_env
        if self._prev_oai_model is not None:
            os.environ["GUTCHECK_OPENAI_MODEL"] = self._prev_oai_model
        if self._prev_oai_base is not None:
            os.environ["GUTCHECK_OPENAI_BASE_URL"] = self._prev_oai_base
        if self._prev_oai_hdrs is not None:
            os.environ["OPENAI_CUSTOM_HEADERS"] = self._prev_oai_hdrs
        super().tearDown()

    def test_runner_routes_to_openai(self):
        self.assertIs(coach_daemon._runner_for("openai_api"), coach_daemon.call_openai_api)

    def test_no_key_degrades_without_importing_sdk(self):
        # absent key → '' and the SDK is never imported (a poisoned import proves it)
        sys.modules["openai"] = _fake_openai(lambda M, C, R: (_ for _ in ()).throw(AssertionError("imported")))
        self.assertEqual(coach_daemon.call_openai_api("spot", timeout=1), "")

    def test_success_returns_content(self):
        keystore.set_api_key("sk-oai-key9", "openai")
        sys.modules["openai"] = _fake_openai(
            lambda M, C, R: R([C(M("  Bet bigger. She calls.  "))]))
        self.assertEqual(coach_daemon.call_openai_api("spot"), "Bet bigger. She calls.")

    def test_key_passed_explicitly_no_retries_and_dials(self):
        keystore.set_api_key("sk-oai-EXPLICIT", "openai")
        settings.apply_patch({"openai_model": "gpt-5.4", "openai_effort": "high"})
        cap = {}
        sys.modules["openai"] = _fake_openai(lambda M, C, R: R([C(M("ok"))]), captured=cap)
        coach_daemon.call_openai_api("spot", timeout=7)
        self.assertEqual(cap.get("api_key"), "sk-oai-EXPLICIT")    # never the ambient OPENAI_API_KEY
        self.assertEqual(cap.get("base_url"), "https://api.openai.com/v1")  # never the ambient OPENAI_BASE_URL
        self.assertEqual(cap.get("max_retries"), 0)               # no paid backoff
        self.assertEqual(cap.get("timeout"), 7)
        cc = cap["_create"]
        self.assertEqual(cc["model"], "gpt-5.4")
        self.assertEqual(cc.get("reasoning_effort"), "high")      # the depth dial (gpt-5 family)
        self.assertEqual(cc["messages"], [{"role": "user", "content": "spot"}])
        self.assertEqual(cc["max_completion_tokens"], coach_daemon._OPENAI_MAX_COMPLETION_TOKENS)
        self.assertNotIn("max_tokens", cc)                        # reasoning models reject the legacy kwarg
        self.assertNotIn("temperature", cc)                       # …and reject a non-default temperature

    def test_ambient_openai_base_url_is_never_honored(self):
        # codex HIGH: the metered key must NEVER be shipped to whatever OPENAI_BASE_URL the daemon inherited.
        # Without an explicit gutcheck opt-in, base_url is forced to the official endpoint, ambient ignored.
        keystore.set_api_key("sk-oai-key9", "openai")
        os.environ["OPENAI_BASE_URL"] = "https://evil.example.com/v1"        # ambient — must be IGNORED
        try:
            cap = {}
            sys.modules["openai"] = _fake_openai(lambda M, C, R: R([C(M("ok"))]), captured=cap)
            coach_daemon.call_openai_api("spot")
            self.assertEqual(cap.get("base_url"), "https://api.openai.com/v1")  # official default, not the ambient
        finally:
            os.environ.pop("OPENAI_BASE_URL", None)

    def test_explicit_gutcheck_base_url_override_is_honored(self):
        # the legit-endpoint escape hatch (Azure / proxy): only an EXPLICIT GUTCHECK_OPENAI_BASE_URL redirects.
        keystore.set_api_key("sk-oai-key9", "openai")
        os.environ["OPENAI_BASE_URL"] = "https://evil.example.com/v1"        # ambient still ignored…
        os.environ["GUTCHECK_OPENAI_BASE_URL"] = "https://azure.example.com/v1"  # …explicit opt-in wins
        try:
            cap = {}
            sys.modules["openai"] = _fake_openai(lambda M, C, R: R([C(M("ok"))]), captured=cap)
            coach_daemon.call_openai_api("spot")
            self.assertEqual(cap.get("base_url"), "https://azure.example.com/v1")
        finally:
            os.environ.pop("OPENAI_BASE_URL", None)
            os.environ.pop("GUTCHECK_OPENAI_BASE_URL", None)

    def test_ambient_custom_headers_denied_during_construction(self):
        # codex HIGH: the SDK reads OPENAI_CUSTOM_HEADERS at construction and spreads it OVER the auth header.
        # The metered worker must DENY it for the constructor window AND restore it afterward.
        keystore.set_api_key("sk-oai-key9", "openai")
        os.environ["OPENAI_CUSTOM_HEADERS"] = "Authorization: Bearer evil"   # ambient — must be DENIED at build
        try:
            cap = {}
            sys.modules["openai"] = _fake_openai(lambda M, C, R: R([C(M("ok"))]), captured=cap)
            coach_daemon.call_openai_api("spot")
            self.assertIsNone(cap.get("_custom_headers_env"))                # denied during construction
            self.assertEqual(os.environ.get("OPENAI_CUSTOM_HEADERS"), "Authorization: Bearer evil")  # restored after
        finally:
            os.environ.pop("OPENAI_CUSTOM_HEADERS", None)

    def test_default_dials_send_effort_on_gpt5(self):
        # the DEFAULT metered call (no apply_patch → gpt-5.5 + low) MUST carry reasoning_effort
        keystore.set_api_key("sk-oai-key9", "openai")
        cap = {}
        sys.modules["openai"] = _fake_openai(lambda M, C, R: R([C(M("ok"))]), captured=cap)
        coach_daemon.call_openai_api("spot")
        cc = cap["_create"]
        self.assertEqual(cc["model"], "gpt-5.5")
        self.assertEqual(cc.get("reasoning_effort"), "low")

    def test_extended_effort_range_none_and_xhigh_are_sent(self):
        # the picker grew the full gpt-5.4/5.5 range: 'none' (cheapest floor) and 'xhigh' (deepest) must
        # reach reasoning_effort. 'none' is a non-empty string so it passes the worker's truthy gate and is
        # sent EXPLICITLY (distinct from omitting the param) — never silently dropped to a default.
        keystore.set_api_key("sk-oai-key9", "openai")
        for level in ("none", "xhigh"):
            with self.subTest(level=level):
                settings.apply_patch({"openai_effort": level})
                cap = {}
                sys.modules["openai"] = _fake_openai(lambda M, C, R: R([C(M("ok"))]), captured=cap)
                coach_daemon.call_openai_api("spot")
                self.assertEqual(cap["_create"].get("reasoning_effort"), level)

    def test_non_gpt5_override_gets_plain_call(self):
        # a power-user override to a non-reasoning model must NOT send reasoning_effort (it would 400)
        keystore.set_api_key("sk-oai-key9", "openai")
        os.environ["GUTCHECK_OPENAI_MODEL"] = "gpt-4o"
        cap = {}
        sys.modules["openai"] = _fake_openai(lambda M, C, R: R([C(M("ok"))]), captured=cap)
        coach_daemon.call_openai_api("spot")
        cc = cap["_create"]
        self.assertEqual(cc["model"], "gpt-4o")                   # env override wins over the UI dial
        self.assertNotIn("reasoning_effort", cc)                  # …and a non-gpt-5 model gets a plain call

    def test_gpt5_non_offered_override_gets_plain_call(self):
        # the allowlist (not a `gpt-5` PREFIX) is the gate: a non-reasoning gpt-5 chat variant reachable via
        # the override starts with "gpt-5" but is NOT an offered reasoning model → it must get a PLAIN call,
        # never reasoning_effort (which would 400). This is the L1+L2 MED fix locked.
        keystore.set_api_key("sk-oai-key9", "openai")
        os.environ["GUTCHECK_OPENAI_MODEL"] = "gpt-5-chat-latest"
        cap = {}
        sys.modules["openai"] = _fake_openai(lambda M, C, R: R([C(M("ok"))]), captured=cap)
        coach_daemon.call_openai_api("spot")
        cc = cap["_create"]
        self.assertEqual(cc["model"], "gpt-5-chat-latest")
        self.assertNotIn("reasoning_effort", cc)                  # prefix-matched but NOT allowlisted → plain call

    def test_refusal_returns_empty(self):
        keystore.set_api_key("sk-oai-key9", "openai")
        sys.modules["openai"] = _fake_openai(lambda M, C, R: R([C(M("", refusal="I can't help with that."))]))
        self.assertEqual(coach_daemon.call_openai_api("spot"), "")

    def test_truncation_returns_empty(self):
        # finish_reason 'length' = the answer (or its reasoning) was cut — never serve a truncated coaching line
        keystore.set_api_key("sk-oai-key9", "openai")
        sys.modules["openai"] = _fake_openai(lambda M, C, R: R([C(M("half a th"), finish_reason="length")]))
        self.assertEqual(coach_daemon.call_openai_api("spot"), "")

    def test_content_filter_returns_empty(self):
        keystore.set_api_key("sk-oai-key9", "openai")
        sys.modules["openai"] = _fake_openai(lambda M, C, R: R([C(M("x"), finish_reason="content_filter")]))
        self.assertEqual(coach_daemon.call_openai_api("spot"), "")

    def test_api_error_never_raises(self):
        keystore.set_api_key("sk-oai-key9", "openai")
        def boom(M, C, R):
            raise RuntimeError("401 invalid_api_key")
        sys.modules["openai"] = _fake_openai(boom)
        self.assertEqual(coach_daemon.call_openai_api("spot"), "")    # degraded, not crashed

    def test_missing_sdk_degrades(self):
        keystore.set_api_key("sk-oai-key9", "openai")
        sys.modules.pop("openai", None)
        import builtins
        real = builtins.__import__
        def no_openai(name, *a, **k):
            if name == "openai":
                raise ImportError("no module named openai")
            return real(name, *a, **k)
        builtins.__import__ = no_openai
        try:
            self.assertEqual(coach_daemon.call_openai_api("spot"), "")
        finally:
            builtins.__import__ = real

    def test_client_is_closed_on_success(self):
        keystore.set_api_key("sk-oai-key9", "openai")
        cap = {}
        sys.modules["openai"] = _fake_openai(lambda M, C, R: R([C(M("ok"))]), captured=cap)
        coach_daemon.call_openai_api("spot")
        self.assertTrue(cap["_client"].closed)                    # the `with` closes the transport on success

    def test_client_is_closed_on_api_error(self):
        # the close path on the ERROR branch (L1 #2): if create() raises, the context manager must STILL
        # close the transport — otherwise an error-storm leaks an httpx client per failed call.
        keystore.set_api_key("sk-oai-key9", "openai")
        cap = {}
        def boom(M, C, R):
            raise RuntimeError("503 service_unavailable")
        sys.modules["openai"] = _fake_openai(boom, captured=cap)
        self.assertEqual(coach_daemon.call_openai_api("spot"), "")   # degraded
        self.assertTrue(cap["_client"].closed)                       # …and the transport was still closed

    def test_empty_choices_returns_empty(self):
        # a (shouldn't-happen) response with no choices must degrade, not IndexError
        keystore.set_api_key("sk-oai-key9", "openai")
        sys.modules["openai"] = _fake_openai(lambda M, C, R: R([]))
        self.assertEqual(coach_daemon.call_openai_api("spot"), "")

    def test_null_content_returns_empty(self):
        # finish_reason 'stop' + no refusal but content=None (a stop-with-empty-body) → '' (never serve blank)
        keystore.set_api_key("sk-oai-key9", "openai")
        sys.modules["openai"] = _fake_openai(lambda M, C, R: R([C(M(content=None))]))
        self.assertEqual(coach_daemon.call_openai_api("spot"), "")

    def test_watchdog_abandons_a_call_that_overruns_the_budget(self):
        import threading as _t
        keystore.set_api_key("sk-oai-key9", "openai")
        def slow(M, C, R):
            _t.Event().wait(5.0)                  # block well past the budget
            return R([C(M("too late"))])
        sys.modules["openai"] = _fake_openai(slow)
        t0 = time.monotonic()
        out = coach_daemon.call_openai_api("spot", timeout=0.3)
        elapsed = time.monotonic() - t0
        self.assertEqual(out, "")                                 # abandoned → facts floor
        self.assertLess(elapsed, 2.0, f"watchdog didn't bound the call: {elapsed:.2f}s")

    def test_inflight_gate_blocks_a_second_concurrent_call(self):
        # codex MED: the gate is SHARED across both workers — a call in flight blocks a 2nd billable call
        # here too, degrading to the facts floor without constructing the client.
        keystore.set_api_key("sk-oai-key9", "openai")
        coach_daemon._METERED_INFLIGHT.acquire()                  # simulate a prior call still in flight
        try:
            cap = {}
            sys.modules["openai"] = _fake_openai(lambda M, C, R: R([C(M("ok"))]), captured=cap)
            self.assertEqual(coach_daemon.call_openai_api("spot"), "")   # facts floor
            self.assertEqual(cap, {})                                     # client never built (gate short-circuit)
        finally:
            coach_daemon._METERED_INFLIGHT.release()

    def test_error_log_is_type_name_only_no_key(self):
        import io
        import contextlib
        keystore.set_api_key("sk-oai-SECRETLOGVALUE", "openai")
        def boom(M, C, R):
            raise RuntimeError("sk-oai-SECRETLOGVALUE leaked into the message")
        sys.modules["openai"] = _fake_openai(boom)
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            self.assertEqual(coach_daemon.call_openai_api("spot"), "")
        log = buf.getvalue()
        self.assertIn("RuntimeError", log)                        # the class diagnoses the failure
        self.assertNotIn("SECRETLOGVALUE", log)                   # …and str(e) never reaches the log


class TestSecretsRoute(unittest.TestCase):
    """/api/secrets GET (status) + POST (set/clear) over a REAL in-process server (the test_settings
    idiom). The headline invariant: the key goes IN over POST and never comes back in any payload."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._prev_dir = os.environ.get("GUTCHECK_DATA_DIR")
        os.environ["GUTCHECK_DATA_DIR"] = self.tmp
        self._prev_env = os.environ.pop(_ENV, None)
        self._prev_oai_env = os.environ.pop(_OAI_ENV, None)        # no env-provided openai key bleeds in
        paths.ensure_data_dir()
        d = Path(self.tmp)
        self.g = G.Game(d / "session.json", d / "career.json", d / "sessions",
                        d / "calibration.json", d / "history.db")
        server.GAME = self.g
        server.SEAM = seam_mod.Seam(self.g)
        self._prev_built_in = server.COACH_BUILT_IN
        server.COACH_BUILT_IN = True
        self.httpd = HTTPServer(("127.0.0.1", 0), server.Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        server.COACH_BUILT_IN = self._prev_built_in
        if self._prev_dir is None:
            os.environ.pop("GUTCHECK_DATA_DIR", None)
        else:
            os.environ["GUTCHECK_DATA_DIR"] = self._prev_dir
        if self._prev_env is not None:
            os.environ[_ENV] = self._prev_env
        if self._prev_oai_env is not None:
            os.environ[_OAI_ENV] = self._prev_oai_env

    def _get(self, path):
        return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}").read())

    def _post(self, path, body):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        return json.loads(urllib.request.urlopen(req).read())

    def _post_raw(self, path, body):
        """POST returning (code, parsed_body) even on a 4xx — so a rejection is assertable."""
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            r = urllib.request.urlopen(req)
            return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_status_unset_then_set_then_clear(self):
        self.assertEqual(self._get("/api/secrets")["anthropic"]["set"], False)
        out = self._post("/api/secrets", {"key": "sk-ant-roundtrip77"})
        self.assertTrue(out["anthropic"]["set"])
        self.assertEqual(out["anthropic"]["hint"], "…ip77")
        self.assertEqual(self._get("/api/secrets")["anthropic"]["set"], True)   # persisted
        cleared = self._post("/api/secrets", {"clear": True})
        self.assertFalse(cleared["anthropic"]["set"])
        self.assertEqual(self._get("/api/secrets")["anthropic"]["set"], False)

    def test_key_never_appears_in_any_payload(self):
        secret = "sk-ant-DO-NOT-LEAK-9090"
        self._post("/api/secrets", {"key": secret})
        for path in ("/api/secrets", "/api/settings"):
            self.assertNotIn("DO-NOT-LEAK", json.dumps(self._get(path)), path)
        # and it's the actual value on disk that's stored (0600), proving the round-trip is real
        self.assertEqual(keystore.get_api_key("anthropic"), secret)
        mode = stat.S_IMODE(os.stat(Path(self.tmp) / "secrets.json").st_mode)
        self.assertEqual(mode, 0o600, oct(mode))

    def test_settings_payload_carries_status(self):
        self.assertFalse(self._get("/api/settings")["api_key"]["set"])
        self._post("/api/secrets", {"key": "sk-ant-fromsettings"})
        s = self._get("/api/settings")
        self.assertTrue(s["api_key"]["set"])
        self.assertEqual(s["api_key"]["source"], "file")

    def test_blank_key_rejected(self):
        code, body = self._post_raw("/api/secrets", {"key": "   "})
        self.assertEqual(code, 400)
        self.assertIn("error", body)

    def test_missing_key_rejected(self):
        code, _ = self._post_raw("/api/secrets", {})       # no key, no clear → set_api_key(None) → 400
        self.assertEqual(code, 400)

    def test_unknown_provider_rejected(self):
        code, _ = self._post_raw("/api/secrets", {"provider": "groq", "key": "sk-x"})
        self.assertEqual(code, 400)

    def test_openai_key_roundtrips_isolated_from_anthropic(self):
        # the route is vendor-keyed: set the openai key under provider:"openai", and it lands in the openai
        # slot ONLY — the anthropic slot stays unset, and the payload carries both statuses, neither the key.
        self.assertEqual(self._get("/api/secrets")["openai"]["set"], False)
        out = self._post("/api/secrets", {"provider": "openai", "key": "sk-proj-route1234"})
        self.assertTrue(out["openai"]["set"])
        self.assertEqual(out["openai"]["hint"], "…1234")
        self.assertFalse(out["anthropic"]["set"])                          # anthropic untouched
        s = self._get("/api/settings")
        self.assertTrue(s["openai_key"]["set"])                            # folded into the settings payload
        self.assertFalse(s["api_key"]["set"])
        for path in ("/api/secrets", "/api/settings"):                     # the key never comes back
            self.assertNotIn("route1234", json.dumps(self._get(path)), path)
        cleared = self._post("/api/secrets", {"provider": "openai", "clear": True})
        self.assertFalse(cleared["openai"]["set"])


if __name__ == "__main__":
    unittest.main()
