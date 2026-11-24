"""stub_health.py: a placeholder "server" on 127.0.0.1:8000 so run.py can run a GPU job (AITER tuning) as a step while NO
engine holds GPU memory. ik7: with SGLang up, the MI355X had 0 bytes free and every tuning benchmark OOMed.

Answers 200 to GET /health (run.py's readiness probe) and a minimal chat completion to POST (run.py's warmup request),
and exits once /root/or/out/STUB_DONE exists (the tune step touches it), freeing the port for the next server.
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

DONE = "/root/or/out/STUB_DONE"


class H(BaseHTTPRequestHandler):
    def _send(self, code, body):
        b = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):  # noqa: N802
        self._send(200, {"status": "stub"})

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(n)
        self._send(200, {"choices": [{"index": 0, "message": {"role": "assistant", "content": "stub"}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 0, "completion_tokens": 0}})

    def log_message(self, *a):
        pass


def watchdog(srv):
    while not os.path.exists(DONE):
        time.sleep(2)
    srv.shutdown()


if __name__ == "__main__":
    print("[stub_health] serving 127.0.0.1:8000 until", DONE, "exists (sglang.launch_server stub)", flush=True)
    s = HTTPServer(("127.0.0.1", 8000), H)
    threading.Thread(target=watchdog, args=(s,), daemon=True).start()
    s.serve_forever()
    print("[stub_health] done", flush=True)
