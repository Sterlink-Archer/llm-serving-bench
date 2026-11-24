#!/usr/bin/env python3
"""Mock OpenAI-compatible server for dry-running bench/traffic.py and bench/matrix.py with no GPU (stdlib).
It behaves like a serving engine in the ways the harness must measure:
  * tokens: prompt_tokens = chars / --cpt + 8 template tokens; streamed output, one token per chunk;
  * prefix cache: 64-char blocks, hashed cumulatively across requests (like vLLM's block hashing); cached_tokens is
    reported in usage.prompt_tokens_details and counted in vllm:prefix_cache_{queries,hits}_total;
  * prefill: TTFT = 15 ms + uncached prompt tokens / --prefill-tps (cached tokens are free);
  * decode: per-token delay = --itl x max(1, active_streams / --knee), so aggregate throughput saturates at --knee streams;
  * /health and Prometheus /metrics with vLLM's counter names.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

L = threading.Lock()
ST = dict(gen=0, prompt=0, q=0, hits=0, active=0, req=0)
BLOCKS = set()
A = None


def prompt_text(req):
    if "messages" in req:
        return "".join(f"<{m['role']}>{m['content'] if isinstance(m['content'], str) else json.dumps(m['content'])}" for m in req["messages"])
    return req.get("prompt", "")


def cache_lookup_insert(text):
    h = hashlib.sha256()
    cached, hit_open = 0, True
    keys = []
    for k in range(0, len(text) - len(text) % 64, 64):
        h.update(text[k:k + 64].encode())
        d = h.digest()
        keys.append(d)
        if hit_open and d in BLOCKS:
            cached = k + 64
        else:
            hit_open = False
    BLOCKS.update(keys)
    return cached


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, b"")
        if self.path == "/metrics":
            with L:
                t = (f"vllm:generation_tokens_total{{m=\"x\"}} {ST['gen']}\nvllm:prompt_tokens_total{{m=\"x\"}} {ST['prompt']}\n"
                     f"vllm:prefix_cache_queries_total{{m=\"x\"}} {ST['q']}\nvllm:prefix_cache_hits_total{{m=\"x\"}} {ST['hits']}\n"
                     f"vllm:request_success_total{{m=\"x\"}} {ST['req']}\nvllm:num_preemptions_total{{m=\"x\"}} 0\n")
            return self._send(200, t.encode(), "text/plain")
        self._send(404, b"{}")

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        text = prompt_text(req)
        with L:
            cached_chars = cache_lookup_insert(text) if not A.no_cache else 0
        pt = int(len(text) / A.cpt) + 8
        ct = min(pt, int(cached_chars / A.cpt) // 16 * 16)
        with L:
            ST["prompt"] += pt
            ST["q"] += pt
            ST["hits"] += ct
        mt = int(req.get("max_tokens") or 16)
        n_out = mt if req.get("ignore_eos") else max(1, min(mt, A.natural_len))
        if not req.get("stream"):
            time.sleep(0.015 + (pt - ct) / A.prefill_tps + n_out * A.itl)
            ans = "Answer: B" if "Options:" in text else "Answer: 42"
            with L:
                ST["gen"] += n_out
                ST["req"] += 1
            return self._send(200, json.dumps(dict(choices=[dict(message=dict(content=ans), finish_reason="stop")],
                                                   usage=dict(prompt_tokens=pt, completion_tokens=n_out,
                                                              prompt_tokens_details=dict(cached_tokens=ct)))).encode())
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def chunk(o):
            d = b"data: " + (o if isinstance(o, bytes) else json.dumps(o).encode()) + b"\n\n"
            self.wfile.write(f"{len(d):x}\r\n".encode() + d + b"\r\n")
            self.wfile.flush()
        usage = lambda n: dict(prompt_tokens=pt, completion_tokens=n, prompt_tokens_details=dict(cached_tokens=ct))  # noqa: E731
        time.sleep(0.015 + (pt - ct) / A.prefill_tps)
        with L:
            ST["active"] += 1
        try:
            for k in range(1, n_out + 1):
                with L:
                    ST["gen"] += 1
                    act = ST["active"]
                chunk(dict(choices=[dict(delta=dict(content=" w"), finish_reason=("length" if k == n_out and req.get("ignore_eos") else
                                                                                     "stop" if k == n_out else None))], usage=usage(k)))
                if k < n_out:
                    time.sleep(A.itl * max(1.0, act / A.knee))
            chunk(dict(choices=[], usage=usage(n_out)))
            chunk(b"[DONE]")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            with L:
                ST["req"] += 1
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with L:
                ST["active"] -= 1


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18100)
    ap.add_argument("--cpt", type=float, default=4.0)
    ap.add_argument("--itl", type=float, default=0.004)
    ap.add_argument("--knee", type=float, default=32)
    ap.add_argument("--prefill-tps", type=float, default=200000)
    ap.add_argument("--natural-len", type=int, default=120)
    ap.add_argument("--no-cache", action="store_true")
    A = ap.parse_args()
    ThreadingHTTPServer.request_queue_size = 2048     # the default 5 resets connections under a 64-client burst
    srv = ThreadingHTTPServer(("127.0.0.1", A.port), H)
    srv.daemon_threads = True
    srv.serve_forever()


if __name__ == "__main__":
    main()
