"""A minimal MCP Apps HOST simulator: loads the widget HTML a gutcheck-mcp serves inside an iframe, plays the
host side of the handshake (answers ui/initialize, sends tool-result, answers tools/call) and reports what the
View did. It is the run that caught the unguarded-localStorage crash and proved the handshake + Refresh path.

    python tools/mcp_apps_host_sim.py [--framed ORIGIN] [--state-url http://127.0.0.1:8777/api/coach/state] [--shot out.png]

Default exercises the SELF-CONTAINED public view (origin=None); --framed http://127.0.0.1:8777 the local framed
mode (needs that game server running). Needs `pip install playwright` in a SEPARATE venv and Google Chrome
(channel="chrome"); it is NOT a gutcheck dependency. It does not model claude.ai's sandbox CSP (claude.ai refuses a
nested localhost frame), only the postMessage protocol.
"""
import argparse, html, json, sys, urllib.request
from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser(); ap.add_argument("--framed", default=None)
ap.add_argument("--state-url", default="http://127.0.0.1:8777/api/coach/state"); ap.add_argument("--shot", default=None)
a = ap.parse_args()
from gutcheck import mcp_server as m          # run with the gutcheck venv's PYTHONPATH, or from the repo root
widget = m._table_ui_html(a.framed)
state = json.load(urllib.request.urlopen(a.state_url))
host = """<!doctype html><body style='margin:0;background:#222'><iframe id=v style='width:900px;height:900px;border:0' srcdoc="%s"></iframe>
<script>
window.LOG=[]; const L=(x)=>window.LOG.push(x); const v=document.getElementById('v'); const STATE=%s;
window.addEventListener('message',e=>{ if(e.source!==v.contentWindow) return; const m=e.data; L(m.method||('resp '+m.id));
  if(m.method==='ui/initialize') v.contentWindow.postMessage({jsonrpc:'2.0',id:m.id,result:{protocolVersion:'2026-01-26',hostInfo:{name:'FakeHost',version:'0'},hostCapabilities:{serverTools:{}},hostContext:{}}},'*');
  if(m.method==='ui/notifications/initialized') v.contentWindow.postMessage({jsonrpc:'2.0',method:'ui/notifications/tool-result',params:{content:[],structuredContent:STATE}},'*');
  if(m.method==='tools/call') v.contentWindow.postMessage({jsonrpc:'2.0',id:m.id,result:{content:[],structuredContent:STATE}},'*');
});
</script></body>""" % (html.escape(widget, quote=True), json.dumps(state))
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True, args=["--mute-audio"])
    try:
        pg = b.new_page(viewport={"width": 920, "height": 920}); errs = []
        pg.on("pageerror", lambda e: errs.append(str(e))); pg.set_content(host); pg.wait_for_timeout(2000)
        fr = pg.frame_locator("#v")
        print("handshake strip:", fr.locator("#hs").inner_text())
        if a.framed is None:
            fr.locator("#rf").click(); pg.wait_for_timeout(600)
            print("after Refresh:", fr.locator("#hs").inner_text()); print("info:", fr.locator(".info").inner_text())
        else:
            print("panel:", fr.locator("#panel").inner_text())
        print("view->host messages:", pg.evaluate("window.LOG")); print("page errors:", errs)
        if a.shot: pg.screenshot(path=a.shot)
    finally:
        b.close()
sys.exit(1 if errs else 0)
