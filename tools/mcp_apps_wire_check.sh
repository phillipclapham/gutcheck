#!/bin/bash
# MCP Apps wire check: what claude.ai / any host sees from a gutcheck-mcp streamable-http endpoint.
#   tools/mcp_apps_wire_check.sh <MCP URL> [Host header]
# Runs initialize, tools/list (show_table's _meta.ui link), resources/list + resources/read of
# ui://gutcheck/table (mimeType, _meta.ui.domain/csp), and tools/call show_table. Run it from a machine
# OUTSIDE your own (e.g. `ssh host "bash -s -- URL" < tools/mcp_apps_wire_check.sh`) to test the tunnel.
U="$1"; HH="$2"
H=(-H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream'); [ -n "$HH" ] && H+=(-H "Host: $HH")
post(){ curl -sS -X POST "$U" "${H[@]}" ${SID:+-H "mcp-session-id: $SID"} -d "$1" | sed -n 's/^data: //p'; }
SID=$(curl -sS -D - -o /dev/null -X POST "$U" "${H[@]}" -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{"extensions":{"io.modelcontextprotocol/ui":{"mimeTypes":["text/html;profile=mcp-app"]}}},"clientInfo":{"name":"wire","version":"1"}}}' | tr -d '\r' | awk -F': ' 'tolower($1)=="mcp-session-id"{print $2}')
echo "session: ${SID:+yes}"
curl -sS -o /dev/null -X POST "$U" "${H[@]}" -H "mcp-session-id: $SID" -d '{"jsonrpc":"2.0","method":"notifications/initialized"}'
post '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' | python3 -c "
import sys,json; d=json.loads(sys.stdin.read()); ts=d['result']['tools']; print('tools:',[t['name'] for t in ts]); st=[t for t in ts if t['name']=='show_table'][0]; print('show_table _meta:',st.get('_meta'))"
post '{"jsonrpc":"2.0","id":3,"method":"resources/list","params":{}}' | python3 -c "
import sys,json; d=json.loads(sys.stdin.read()); print('resources:',[(r['uri'],r.get('mimeType'),r.get('_meta')) for r in d['result']['resources']])"
post '{"jsonrpc":"2.0","id":4,"method":"resources/read","params":{"uri":"ui://gutcheck/table"}}' | python3 -c "
import sys,json; d=json.loads(sys.stdin.read()); c=d['result']['contents'][0]; print('read mime:',c['mimeType'],'| _meta:',c.get('_meta')); t=c['text']; print('html bytes',len(t),'| has initialize:', 'ui/initialize' in t, '| has ping:', \"'ping'\" in t)"
post '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"show_table","arguments":{}}}' | python3 -c "
import sys,json; d=json.loads(sys.stdin.read()); r=d['result']; print('call keys:',sorted(r.keys()),'| _meta:',r.get('_meta'),'| structured has hand_no:', 'hand_no' in json.dumps(r.get('structuredContent')))"
