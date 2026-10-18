"""Tests for mcp_server.py — the thin MCP proxy over the game server's seam (P2).

Uses the in-memory FastMCP path (build_server + call_tool) against a LIVE in-process game server
(GUTCHECK_URL points at it). Skipped wholesale if the optional `mcp` extra isn't installed, so the
core suite still runs on a zero-extra checkout.
"""

import asyncio
import unittest.mock
import importlib.util
import json
import os
import random
import shutil
import tempfile
import threading
import unittest
from http.server import HTTPServer
from pathlib import Path

from gutcheck import game as G
from gutcheck import seam as seam_mod
from gutcheck import server as S

HAS_MCP = importlib.util.find_spec("mcp") is not None


def _play(d: Path):
    random.seed(7)
    g = G.Game(d / "session.json", d / "career.json", d / "sessions",
               d / "calibration.json", d / "history.db")
    # sit_down draws the master seed from SystemRandom, so random.seed(7) alone never made the deals
    # repeat; pin that draw so the fixture plays the same session every run.
    with unittest.mock.patch.object(G.random, "SystemRandom", lambda: random.Random(7)):
        g.sit_down(hero="Phill")
    sid = g.s["session"]["id"]
    for _ in range(25):
        g.new_hand()
        guard = 0
        while True:
            v = g.view()
            if not v.get("in_hand") or v.get("over") or v.get("to_act") != "hero":
                break
            o = v.get("options") or {}
            if o.get("is_check"):
                g.act("check")
            elif o.get("check_or_call"):
                g.act("call")
            else:
                break
            guard += 1
            if guard > 30:
                break
    g.get_up()
    report = json.loads((d / "sessions" / f"{sid}.json").read_text())
    return g, sid, report


def _payload(res):
    """call_tool returns a content list (or a (content, structured) tuple in some versions)."""
    if hasattr(res, "structuredContent"):          # a tool that returns CallToolResult (show_table)
        return res.structuredContent
    content = res[0] if isinstance(res, tuple) else res
    return json.loads(content[0].text)


@unittest.skipUnless(HAS_MCP, "mcp SDK not installed (optional extra)")
class TestMcpProxy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = Path(tempfile.mkdtemp())
        cls.g, cls.sid, report = _play(cls.d)
        pots = report.get("biggest_pots") or report.get("notable_hands") or []
        cls.hand_no = pots[0]["hand_no"]
        # find a reviewable hand with a gradable postflop decision (for the commit_read success test)
        cls.dp = None
        _s = seam_mod.Seam(cls.g)
        for h in (report.get("biggest_pots", []) + report.get("notable_hands", [])):
            dps = [d for d in _s.get_ground_truth(cls.sid, h["hand_no"]).get("decision_points", [])
                   if d.get("equity") is not None]
            if dps:
                cls.dp = (h["hand_no"], dps[0]["n"])
                break
        S.GAME = cls.g
        S.SEAM = seam_mod.Seam(cls.g)
        cls.httpd = HTTPServer(("127.0.0.1", 0), S.Handler)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls._prev = os.environ.get("GUTCHECK_URL")
        os.environ["GUTCHECK_URL"] = f"http://127.0.0.1:{cls.port}"
        from gutcheck import mcp_server
        cls.M = mcp_server

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        if cls._prev is None:
            os.environ.pop("GUTCHECK_URL", None)
        else:
            os.environ["GUTCHECK_URL"] = cls._prev

    def _call(self, mcp, name, args):
        return _payload(asyncio.run(mcp.call_tool(name, args)))

    # --- per-transport gating: the security decision (answer-key tool is stdio/local-only) ---
    def test_ground_truth_registered_for_local_not_remote(self):
        local = {t.name for t in asyncio.run(
            self.M.build_server(expose_ground_truth=True).list_tools())}
        remote = {t.name for t in asyncio.run(
            self.M.build_server(expose_ground_truth=False).list_tools())}
        base = {"get_table_state", "get_history", "get_calibration", "get_sizing_tells", "commit_read"}
        self.assertTrue(base <= local and base <= remote)     # reads + commit on BOTH transports
        self.assertIn("get_ground_truth", local)              # answer-key: local stdio only
        self.assertNotIn("get_ground_truth", remote)          # NOT over the remote exfil leg

    # --- read tools proxy through to the live seam ---
    def test_get_table_state_proxies(self):
        d = self._call(self.M.build_server(expose_ground_truth=True), "get_table_state", {})
        self.assertIn("seated", d)

    def test_get_history_proxies(self):
        d = self._call(self.M.build_server(expose_ground_truth=True), "get_history", {"since": 0})
        self.assertEqual({"hands", "watermark", "count"}, set(d) & {"hands", "watermark", "count"})
        self.assertGreater(d["count"], 0)

    def test_reaches_game_with_a_proxy_in_the_environment(self):
        # 0.1.2: urlopen honoured http_proxy for the loopback game server, so with a proxy exported in the
        # shell that launched the MCP client every tool reported the game 'unreachable' while it was up.
        saved = {k: os.environ.get(k) for k in ("http_proxy", "HTTP_PROXY", "no_proxy", "NO_PROXY")}
        try:
            for k in ("no_proxy", "NO_PROXY"):
                os.environ.pop(k, None)
            os.environ["http_proxy"] = os.environ["HTTP_PROXY"] = "http://127.0.0.1:9"   # nothing listens
            d = self._call(self.M.build_server(expose_ground_truth=True), "get_table_state", {})
            self.assertIn("seated", d)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_get_calibration_proxies(self):
        d = self._call(self.M.build_server(expose_ground_truth=True), "get_calibration", {})
        self.assertTrue(set(d) & {"exact", "modeled"})        # the calibration shape

    def test_get_history_session_filter_proxies(self):
        mcp = self.M.build_server(expose_ground_truth=True)
        scoped = self._call(mcp, "get_history", {"session_id": self.sid})
        self.assertGreater(scoped["count"], 0)
        self.assertTrue(all(h["session_id"] == self.sid for h in scoped["hands"]))   # only this session
        ghost = self._call(mcp, "get_history", {"session_id": "no-such-session"})
        self.assertEqual(ghost["count"], 0)                   # scope isolates

    def test_get_history_defaults_to_recent_window(self):
        # the bug this guards: get_history() once dumped the WHOLE corpus (blew the host token budget).
        # The default must be the most-recent _HISTORY_DEFAULT hands (the coach's "what did I just play"),
        # not the full forward stream. The 25-hand fixture exceeds the default, so the tail is observable.
        mcp = self.M.build_server(expose_ground_truth=True)
        full = self._call(mcp, "get_history", {"latest": False, "limit": 500})   # the whole stream
        recent = self._call(mcp, "get_history", {})                              # default: recent window
        self.assertLessEqual(recent["count"], self.M._HISTORY_DEFAULT)           # bounded by the default
        self.assertEqual(recent["watermark"], full["watermark"])                 # newest id either way
        ids = [h["id"] for h in recent["hands"]]
        self.assertEqual(ids, sorted(ids))                                       # presented oldest→newest
        self.assertGreater(full["count"], self.M._HISTORY_DEFAULT)               # fixture proves the cap bites
        self.assertEqual(recent["count"], self.M._HISTORY_DEFAULT)
        self.assertEqual(ids, [h["id"] for h in full["hands"]][-recent["count"]:])   # the TAIL of the stream

    def test_get_history_clamps_and_forwards_limit_and_latest(self):
        # the tool clamps limit to _HISTORY_MAX (budget guard) and forwards latest as a query flag.
        # Spy on _get so we assert the EXACT proxied params (a 9999 request must never reach the seam).
        captured = {}
        orig = self.M._get
        def spy(path, **params):
            captured.clear(); captured.update(params)
            return {"hands": [], "watermark": 0, "count": 0}
        self.M._get = spy
        try:
            mcp = self.M.build_server(expose_ground_truth=True)
            self._call(mcp, "get_history", {"limit": 9999, "latest": True})
            self.assertEqual(captured["limit"], self.M._HISTORY_MAX)   # 9999 clamped down, never sent raw
            self.assertEqual(captured["latest"], "1")                  # bool → query string the server parses
            self._call(mcp, "get_history", {"limit": 3, "latest": False})
            self.assertEqual(captured["limit"], 3)                     # an in-range limit passes through
            self.assertEqual(captured["latest"], "0")
        finally:
            self.M._get = orig

    def test_get_sizing_tells_proxies(self):
        mcp = self.M.build_server(expose_ground_truth=True)
        d = self._call(mcp, "get_sizing_tells", {"session_id": self.sid})
        self.assertEqual(d["session_id"], self.sid)
        self.assertIsInstance(d["villains"], list)
        self.assertIn("note", d)
        for v in d["villains"]:                               # well-formed if present
            self.assertGreaterEqual(v["showdowns"], 1)
            self.assertIn("aggressive_postflop", v)
            self.assertEqual([b["size"] for b in v["by_size"]], ["small", "medium", "big", "overbet"])
        # scope isolates — a bogus session has no villains
        self.assertEqual(self._call(mcp, "get_sizing_tells", {"session_id": "ghost"})["villains"], [])

    def test_no_arg_flow_resolves_through_none_drop(self):
        # complement L3 / MED-1: the PRIMARY flow calls these tools with NO session arg → session_id=None
        # → _get must DROP it (not serialize the string "None") → the server resolves the default. If _get
        # serialized "None", get_history would query session_id='None' and return 0 rows — so a non-zero
        # all-sessions count is the load-bearing proof the None-drop works (BOTH tools share _get). (The
        # None→live-session resolution itself is unit-covered in test_sizing_tells; this fixture is got-up.)
        mcp = self.M.build_server(expose_ground_truth=True)
        self.assertGreater(self._call(mcp, "get_history", {})["count"], 0)      # None→all, not "None"→empty
        tells = self._call(mcp, "get_sizing_tells", {})                         # routes cleanly, no "None" crash
        self.assertIsInstance(tells["villains"], list)

    # --- the §3f archetype gate is FULLY structural on the coach surface (no reveal option) ---
    def test_ground_truth_always_gated_through_coach_tool(self):
        mcp = self.M.build_server(expose_ground_truth=True)
        gt = self._call(mcp, "get_ground_truth", {"session_id": self.sid, "hand_no": self.hand_no})
        self.assertTrue(gt["ok"])
        self.assertFalse(gt["archetype_revealed"])
        for v in gt["villains"]:
            self.assertNotIn("archetype", v)
            self.assertNotIn("_truth", v)
        # the coach tool exposes NO reveal_archetype param — the gate can't be flipped by the model
        tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
        props = tools["get_ground_truth"].inputSchema.get("properties") or {}
        self.assertNotIn("reveal_archetype", props)

    # --- commit_read's review path actually round-trips the POST + grades (was untested, L1 T1) ---
    def test_commit_read_review_success_proxies(self):
        if not self.dp:
            self.skipTest("no gradable postflop decision in the sampled session")
        hand_no, n = self.dp
        res = self._call(self.M.build_server(expose_ground_truth=True), "commit_read",
                         {"kind": "review", "session_id": self.sid, "hand_no": hand_no,
                          "n": n, "gut": 55, "lean": "call"})
        self.assertTrue(res.get("ok"))       # the POST reached /api/calibration/review and graded
        self.assertIn("gap", res)

    # --- commit_read routing: unsupported kind is a clean ToolError, not a silent wrong call ---
    def test_commit_read_rejects_non_review(self):
        from mcp.server.fastmcp.exceptions import ToolError
        mcp = self.M.build_server(expose_ground_truth=True)
        with self.assertRaises(ToolError):
            asyncio.run(mcp.call_tool("commit_read", {"kind": "live", "gut": 50}))

    # --- MCP Apps embedded UI (ui://): the HUMAN play-plane. The widget frames the game server
    #     SAME-ORIGIN so the app's own _same_origin/_host_ok guards still hold; the COACH keeps the
    #     gated tool plane. Local stdio only (the frame targets the operator's localhost). ---------
    def test_table_ui_surface_is_local_stdio_only(self):
        local_tools = {t.name for t in asyncio.run(
            self.M.build_server(expose_ground_truth=True).list_tools())}
        remote_tools = {t.name for t in asyncio.run(
            self.M.build_server(expose_ground_truth=False).list_tools())}
        self.assertIn("show_table", local_tools)        # the UI-linked tool: local path
        self.assertNotIn("show_table", remote_tools)    # not over the remote leg (frames localhost)
        local_res = [str(r.uri) for r in asyncio.run(
            self.M.build_server(expose_ground_truth=True).list_resources())]
        remote_res = [str(r.uri) for r in asyncio.run(
            self.M.build_server(expose_ground_truth=False).list_resources())]
        self.assertTrue(any("gutcheck/table" in u for u in local_res))
        self.assertFalse(any("gutcheck/table" in u for u in remote_res))

    def test_show_table_tool_links_the_ui_resource(self):
        # the host preloads+renders the linked resource ONLY if the tool carries _meta.ui.resourceUri
        # (nested form; the flat ui/resourceUri is deprecated). Lock the exact wire shape — list_tools
        # must propagate it (verified: MCPTool(_meta=info.meta)) or the host never renders.
        tools = {t.name: t for t in asyncio.run(
            self.M.build_server(expose_ground_truth=True).list_tools())}
        self.assertEqual(tools["show_table"].meta, {"ui": {"resourceUri": self.M._TABLE_UI_URI},
                                                    "ui/resourceUri": self.M._TABLE_UI_URI,
                                    "openai/outputTemplate": self.M._TABLE_UI_URI})

    def test_show_table_renders_via_the_gated_coach_route(self):
        # the firewall at the show_table entry: it MUST proxy the archetype-STRIPPED /api/coach/state,
        # NEVER the ungated browser /api/state. Spy the proxy to LOCK the route (codex/c-n L3: the old
        # test looped a "villains" key coach/state never returns, so the strip went unchecked) and
        # assert no villain type/archetype survives in EITHER seats surface the seam strips.
        captured = []
        orig = self.M._get
        def spy(path, **params):
            captured.append(path)
            return orig(path, **params)
        self.M._get = spy
        try:
            d = self._call(self.M.build_server(expose_ground_truth=True), "show_table", {})
        finally:
            self.M._get = orig
        self.assertIn("/api/coach/state", captured)      # the gated coach plane (archetype stripped)
        self.assertNotIn("/api/state", captured)         # never the ungated browser plane
        self.assertIn("seated", d)
        seats = list(d.get("seats") or []) + list((d.get("table") or {}).get("seats") or [])
        for s in seats:
            self.assertNotIn("type", s)                  # seam pops both → absent, never a real type
            self.assertNotIn("archetype", s)

    def test_table_ui_resource_read_carries_csp_and_frames_server(self):
        # the load-bearing render contract: read() carries mimeType ;profile=mcp-app AND
        # _meta.ui.csp.frameDomains == [the DERIVED game-server origin] (assert against _frame_origin,
        # NOT raw env — they differ on a trailing slash/path; c-n L3), else the host applies frame-src
        # 'none' (a blank iframe). The body frames that SAME origin so the inner app stays same-origin.
        base = self.M._frame_origin()
        contents = list(asyncio.run(
            self.M.build_server(expose_ground_truth=True).read_resource(self.M._TABLE_UI_URI)))
        self.assertEqual(len(contents), 1)
        c = contents[0]
        self.assertEqual(c.mime_type, "text/html;profile=mcp-app")
        self.assertEqual(c.meta, {"ui": {"csp": {"frameDomains": [base]}}})
        self.assertIn(f'src="{base}/"', c.content)       # frames the game server, same-origin

    def test_table_ui_derives_a_clean_origin_from_a_messy_url(self):
        # the framed origin is DERIVED (scheme://netloc) — a path/query/injection in GUTCHECK_URL is
        # DROPPED at the source, not merely html-escaped, so it can reach neither the iframe src nor the
        # CSP (codex MED + c-n HIGH L3). html.escape stays as defence-in-depth over an already-clean origin.
        prev = os.environ.get("GUTCHECK_URL")
        os.environ["GUTCHECK_URL"] = 'http://127.0.0.1:8777/"><script>x</script>'
        try:
            c = list(asyncio.run(
                self.M.build_server(expose_ground_truth=True).read_resource(self.M._TABLE_UI_URI)))[0]
            self.assertNotIn("<script>x</script>", c.content)            # path/script dropped, not present
            self.assertIn('src="http://127.0.0.1:8777/"', c.content)     # clean derived origin
            self.assertEqual(c.meta, {"ui": {"csp": {"frameDomains": ["http://127.0.0.1:8777"]}}})
        finally:
            if prev is None:
                os.environ.pop("GUTCHECK_URL", None)
            else:
                os.environ["GUTCHECK_URL"] = prev

    def test_table_ui_not_registered_for_a_non_http_origin(self):
        # fail-closed: a non-http(s) GUTCHECK_URL (javascript:, data:, file:, userinfo, …) means NO
        # widget at all, rather than a javascript: iframe src that html.escape would pass through
        # unchanged (codex MED + c-n HIGH L3 — escape doesn't reject a no-special-char scheme).
        for bad in ("javascript:alert(1)", "data:text/html,<script>x</script>", "file:///etc/passwd",
                    "http://user:pass@127.0.0.1:8777"):
            prev = os.environ.get("GUTCHECK_URL")
            os.environ["GUTCHECK_URL"] = bad
            try:
                self.assertIsNone(self.M._frame_origin(), bad)
                mcp = self.M.build_server(expose_ground_truth=True)
                tools = {t.name for t in asyncio.run(mcp.list_tools())}
                res = [str(r.uri) for r in asyncio.run(mcp.list_resources())]
                self.assertNotIn("show_table", tools, bad)                      # tool absent
                self.assertFalse(any("gutcheck/table" in u for u in res), bad) # resource absent
            finally:
                if prev is None:
                    os.environ.pop("GUTCHECK_URL", None)
                else:
                    os.environ["GUTCHECK_URL"] = prev

    # --- the claude.ai custom-connector test leg: remote streamable-http + the widget, READ-ONLY ----
    def test_ui_domain_matches_the_claude_docs_vector(self):
        # the worked example on claude.com/docs/connectors/building/mcp-apps/getting-started
        self.assertEqual(self.M._ui_domain_for("https://example.com/mcp"),
                         "c3d80a4ed901ee05b21755a88273b4a4.claudemcpcontent.com")
        # the hash covers the exact string: a trailing slash is a different connector URL
        self.assertNotEqual(self.M._ui_domain_for("https://example.com/mcp"),
                            self.M._ui_domain_for("https://example.com/mcp/"))

    def test_remote_ui_mode_serves_widget_readonly_with_domain(self):
        dom = self.M._ui_domain_for("https://t.example/s3cret")
        srv = self.M.build_server(expose_ground_truth=False, expose_ui=True, ui_frame=False, read_only=True,
                                  ui_domain=dom, http_path="/s3cret", allowed_hosts=["t.example"])
        tools = {t.name for t in asyncio.run(srv.list_tools())}
        self.assertIn("show_table", tools)
        self.assertNotIn("commit_read", tools)          # nothing that writes on the public leg
        self.assertNotIn("get_ground_truth", tools)     # answer key never over a remote leg
        self.assertEqual(srv.settings.streamable_http_path, "/s3cret")
        # resource meta carries the derived domain in the read contents (where the host looks) and NO csp:
        # claude.ai's sandbox refuses a nested localhost frame, so the public view frames nothing
        c = list(asyncio.run(srv.read_resource(self.M._TABLE_UI_URI)))[0]
        self.assertEqual(c.meta, {"ui": {"domain": dom}})
        body = c.content
        self.assertNotIn("<iframe", body)               # self-contained: no nested frame
        for needle in ("ui/initialize", "ui/notifications/initialized", "'ping'", "size-changed",
                       "2026-01-26", 'id="tbl"', 'id="rf"', "tools/call", "show_table",
                       "ui/notifications/tool-result", "fpToolName", "toolInfo"):
            self.assertIn(needle, body)
        self.assertNotIn("http://unpkg", body)          # no CDN dependency in the sandbox
        self.assertNotIn("innerHTML", body)             # state text is only ever set as textContent
        # the tool RESULT carries the same _meta.ui link as the tool definition
        with unittest.mock.patch.object(self.M, "_get", return_value={"ok": True}):
            res = asyncio.run(srv.call_tool("show_table", {}))
        self.assertEqual(res.meta, {"ui": {"resourceUri": self.M._TABLE_UI_URI},
                                    "ui/resourceUri": self.M._TABLE_UI_URI,
                                    "openai/outputTemplate": self.M._TABLE_UI_URI})

    def test_local_ui_mode_still_frames_the_game_with_the_panel(self):
        body = self.M._table_ui_html("http://127.0.0.1:8777")
        for needle in ("<iframe", 'allow="local-network-access"', 'id="panel"', "ui/initialize",
                       f"height:{self.M._FRAME_HEIGHT}px"):
            self.assertIn(needle, body)
        self.assertNotIn("innerHTML", body)

    def test_public_history_omits_the_replay_seed_and_assignment(self):
        fake = {"hands": [{"hand_no": 1, "seed": 99, "assign": [2, 3], "log": []}], "count": 1, "watermark": 1}
        pub = self.M.build_server(expose_ground_truth=False, read_only=True)
        loc = self.M.build_server(expose_ground_truth=True)
        with unittest.mock.patch.object(self.M, "_get", side_effect=lambda *a, **k: json.loads(json.dumps(fake))):
            hp = self._call(pub, "get_history", {})["hands"][0]
            hl = self._call(loc, "get_history", {})["hands"][0]
        self.assertNotIn("seed", hp); self.assertNotIn("assign", hp)   # public: nothing derives the deal
        self.assertEqual(hp["hand_no"], 1)
        self.assertIn("seed", hl)                                       # local stdio keeps full replay

    def test_public_url_validation(self):
        ok = self.M._public_leg_kwargs("https://T.Example/abcdefghijklmnop1234")
        self.assertEqual(ok["allowed_hosts"], ["t.example"])            # normalised hostname
        self.assertEqual(ok["http_path"], "/abcdefghijklmnop1234")
        self.assertEqual(ok["ui_domain"], self.M._ui_domain_for("https://T.Example/abcdefghijklmnop1234"))
        self.assertTrue(ok["read_only"] and ok["expose_ui"] and not ok["ui_frame"])
        self.assertEqual(self.M._public_leg_kwargs("https://t.example:8443/abcdefghijklmnop1234")["allowed_hosts"],
                         ["t.example", "t.example:8443"])
        for bad in ("https://t.example/short",                          # guessable path: no entropy
                    "https://t.example/{x}abcdefghijklmnop",            # route-template syntax
                    "https://t.example/abcdefghijklmnop1234/",          # trailing slash
                    "https://t.example/a%2Fcdefghijklmnop1234",         # percent-escape
                    "https://u:p@t.example/abcdefghijklmnop1234",       # userinfo
                    "http://t.example/abcdefghijklmnop1234",            # not https
                    "https://[broken/abcdefghijklmnop1234",             # malformed
                    "https://t.example/abcdefghijklmnop1234#"):         # fragment
            with self.assertRaises(SystemExit, msg=bad):
                self.M._public_leg_kwargs(bad)

    def test_main_rejects_ui_flag_misuse(self):
        for argv in (["--transport", "streamable-http", "--ui"],
                     ["--transport", "streamable-http", "--public-url", "https://x.example/abcdefghijklmnop12"],
                     ["--ui", "--public-url", "https://x.example/abcdefghijklmnop12"]):
            with self.assertRaises(SystemExit, msg=argv):
                self.M.main(argv)

    def test_ui_extension_declared_in_initialize_capabilities(self):
        # THE render gate (the Desktop non-render bug): a host engages the widget pipeline only if the
        # server DECLARES io.modelcontextprotocol/ui in its initialize capabilities.extensions (spec
        # 2026-01-26 / SEP-1724 — the `extensions` channel, NOT `experimental`; mcp 1.28.0 has no
        # `extensions` field so it rides as an extra via extra="allow"). Absent it the host degrades to
        # text-only. Advertised on the local UI path; NOT on the remote leg (which serves no UI).
        loc_srv = self.M.build_server(expose_ground_truth=True)
        loc = loc_srv._mcp_server.create_initialization_options().capabilities.model_dump(exclude_none=True)
        self.assertEqual((loc.get("extensions") or {}).get("io.modelcontextprotocol/ui"), {})
        rem = self.M.build_server(expose_ground_truth=False)._mcp_server \
                  .create_initialization_options().capabilities.model_dump(exclude_none=True)
        self.assertNotIn("io.modelcontextprotocol/ui", (rem.get("extensions") or {}))
        # idempotent: re-advertising must not TypeError on an already-present `extensions` nor clobber
        # it (codex L3 — the merge-style rebuild, vs an extensions= kwarg collision at startup).
        self.M._advertise_ui_extension(loc_srv)
        caps2 = loc_srv._mcp_server.create_initialization_options().capabilities.model_dump(exclude_none=True)
        self.assertEqual((caps2.get("extensions") or {}).get("io.modelcontextprotocol/ui"), {})


class TestSessionIdIsOpaque(unittest.TestCase):
    """The master seed rebuilds every villain hole card, every future deck and the §3f archetype roster,
    so no coach-facing string may carry it. Reproduced against a scratch game (Diogenes 10-17 MEDIUM)."""

    def _seated(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        g = G.Game(d / "session.json", d / "career.json", d / "sessions",
                   d / "calibration.json", d / "history.db")
        g.sit_down(hero="Phill")
        return g

    def test_master_seed_is_too_wide_to_search_from_the_visible_names(self):
        seeds = [self._seated().s["master_seed"] for _ in range(5)]
        self.assertGreater(max(seeds), 2**40)       # a 31-bit seed never passes this
        self.assertLess(max(seeds) + 10**6 * 101, 2**53)     # exact as a JSON double on a Node host

    def test_session_id_does_not_carry_the_master_seed(self):
        g = self._seated()
        seed = str(g.s["master_seed"])
        self.assertNotIn(seed, g.s["session"]["id"])

    def test_coach_state_does_not_carry_the_master_seed(self):
        g = self._seated()
        g.new_hand()
        blob = json.dumps(seam_mod.Seam(g).get_coach_table_state())
        self.assertNotIn(str(g.s["master_seed"]), blob)

    def test_the_roster_is_not_rebuildable_from_the_session_id(self):
        g = self._seated()
        want = [(p["name"], p["type"]) for p in g.s["players"][1:]]
        seed = g.s["master_seed"]
        sid = g.s["session"]["id"]
        for form in (str(seed), format(seed, "x"), format(seed, "X")):
            self.assertNotIn(form, sid)
        for tok in sid.split("-")[1:]:
            for base in (10, 16):
                try:
                    n = int(tok, base)
                except ValueError:
                    continue
                got = [(v["name"], v["type"]) for v in g._generate_villains(random.Random(n))]
                self.assertNotEqual(got, want)


@unittest.skipUnless(HAS_MCP, "mcp SDK not installed (optional extra)")
class TestMcpHistoryStrip(unittest.TestCase):
    def test_every_non_stdio_leg_omits_seed_and_assign(self):
        from gutcheck import mcp_server as M
        fake = {"hands": [{"hand_no": 1, "seed": 99, "assign": [2, 3], "log": []}], "count": 1, "watermark": 1}
        legs = {"public read-only": dict(expose_ground_truth=False, read_only=True),
                "plain streamable-http": dict(expose_ground_truth=False, read_only=False)}
        for name, kw in legs.items():
            srv = M.build_server(**kw)
            with unittest.mock.patch.object(M, "_get", side_effect=lambda *a, **k: json.loads(json.dumps(fake))):
                h = _payload(asyncio.run(srv.call_tool("get_history", {})))["hands"][0]
            self.assertNotIn("seed", h, name)
            self.assertNotIn("assign", h, name)
        loc = M.build_server(expose_ground_truth=True)       # local stdio is the trusted leg: it keeps replay
        with unittest.mock.patch.object(M, "_get", side_effect=lambda *a, **k: json.loads(json.dumps(fake))):
            h = _payload(asyncio.run(loc.call_tool("get_history", {})))["hands"][0]
        self.assertIn("seed", h)


@unittest.skipUnless(HAS_MCP, "mcp SDK not installed (optional extra)")
class TestMcpErrorHandling(unittest.TestCase):
    def test_backend_unreachable_is_a_toolerror(self):
        from mcp.server.fastmcp.exceptions import ToolError
        from gutcheck import mcp_server
        prev = os.environ.get("GUTCHECK_URL")
        os.environ["GUTCHECK_URL"] = "http://127.0.0.1:9"   # discard port — connection refused
        try:
            mcp = mcp_server.build_server(expose_ground_truth=True)
            with self.assertRaises(ToolError):
                asyncio.run(mcp.call_tool("get_table_state", {}))
        finally:
            if prev is None:
                os.environ.pop("GUTCHECK_URL", None)
            else:
                os.environ["GUTCHECK_URL"] = prev


@unittest.skipUnless(HAS_MCP, "mcp SDK not installed (optional extra)")
class TestMcpHttpSecurity(unittest.TestCase):
    def test_main_refuses_non_loopback_http_host(self):
        # the SDK only auto-protects a loopback bind; a non-loopback --host ships no Origin/auth defense,
        # so it must be REFUSED until the remote hardening lands. [L2 H1 / complement L3]
        from gutcheck import mcp_server
        with self.assertRaises(SystemExit):
            mcp_server.main(["--transport", "streamable-http", "--host", "100.64.0.1"])

    def test_constructor_host_reaches_the_bind(self):
        # the --host guard is only meaningful if FastMCP binds the constructor host (verified via lsof:
        # streamable-http binds 127.0.0.1, not 0.0.0.0). Lock the contract against an SDK regression.
        from gutcheck import mcp_server
        m = mcp_server.build_server(expose_ground_truth=False, host="127.0.0.1", port=8791)
        self.assertEqual(m.settings.host, "127.0.0.1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
