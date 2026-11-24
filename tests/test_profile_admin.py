#!/usr/bin/env python3
"""bench/profile_prefill.py sends the SGLang admin key as a Bearer token to /start_profile and /stop_profile ONLY (SGLang
2026-11: those are AuthLevel.ADMIN_FORCE, HTTP 403 without --admin-api-key, 401 without the header; e5 B300 hit the 403).
python3 -I tests/test_profile_admin.py"""
import http.server
import pathlib
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import profile_prefill as PP  # noqa: E402

seen = {}


class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        seen[self.path] = self.headers.get("Authorization")
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *a):
        pass


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_address[1]}"
PP.ADMIN_KEY = "k123"
for path in ("/start_profile", "/v1/chat/completions", "/stop_profile"):
    PP.post(base, path, {})
PP.ADMIN_KEY = ""
PP.post(base, "/start_profile", {})
srv.shutdown()
bad = []
if seen.get("/v1/chat/completions") is not None:
    bad.append("inference call carried the admin key")
if seen.get("/stop_profile") != "Bearer k123":
    bad.append(f"/stop_profile header {seen.get('/stop_profile')!r}")
if seen.get("/start_profile") is not None:          # the last call, with no key set, must send no header
    bad.append(f"/start_profile without a key sent {seen.get('/start_profile')!r}")
print("FAIL\n  " + "\n  ".join(bad) if bad else "OK: admin Bearer key on the profiler endpoints only, none when unset")
sys.exit(1 if bad else 0)
