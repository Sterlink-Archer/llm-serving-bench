#!/usr/bin/env python3
"""bench/traffic.py on replies whose first token carries no text (Inkling 2 Mini swallows a special token) and on a reply
with no text at all (2026-11-22: summarize crashed comparing None with the window).  python3 -I tests/test_traffic_notext.py
A local SSE server sends: reply A = an empty-content chunk with usage completion_tokens=1 at ~0.2 s, then text at ~0.4 s and
~0.5 s; reply B = only a finish chunk with no text."""
import argparse
import asyncio
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import traffic as T  # noqa: E402


def ev(obj):
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


async def handle(reader, writer):
    req = b""
    while b"\r\n\r\n" not in req:
        req += await reader.read(65536)
    head, _, body = req.partition(b"\r\n\r\n")
    n = int([ln.split(b":")[1] for ln in head.split(b"\r\n") if ln.lower().startswith(b"content-length")][0])
    while len(body) < n:
        body += await reader.read(65536)
    kind = json.loads(body)["messages"][0]["content"]
    writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
    u = lambda c: dict(prompt_tokens=10, completion_tokens=c, total_tokens=10 + c)  # noqa: E731
    if kind == "A":
        writer.write(ev(dict(choices=[dict(index=0, delta=dict(role="assistant", content=""))], usage=u(0))))
        await asyncio.sleep(0.2)
        writer.write(ev(dict(choices=[dict(index=0, delta=dict(content=""))], usage=u(1))))
        await asyncio.sleep(0.2)
        writer.write(ev(dict(choices=[dict(index=0, delta=dict(content="Hello"))], usage=u(2))))
        await asyncio.sleep(0.1)
        writer.write(ev(dict(choices=[dict(index=0, delta=dict(content=" there"), finish_reason="stop")], usage=u(3))))
    else:
        await asyncio.sleep(0.1)
        writer.write(ev(dict(choices=[dict(index=0, delta=dict(content=""), finish_reason="stop")], usage=u(1))))
    writer.write(ev(dict(choices=[], usage=u(3 if kind == "A" else 1))) + b"data: [DONE]\n\n")
    await writer.drain()
    writer.close()


async def run(token_basis):
    srv = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]
    reqs = []
    async with srv:
        for i, kind in enumerate("AB"):
            r = T.Req(i, dict(cls="x", max_tokens=8), None)
            body = dict(model="m", messages=[dict(role="user", content=kind)], stream=True)
            await T.send("127.0.0.1", port, "/v1/chat/completions", body, r, 10, token_basis=token_basis)
            reqs.append(r)
    return reqs


bad = []
a_text, _ = asyncio.run(run(False))
a_tok, b_tok = asyncio.run(run(True))
# text basis (every Gemma run): first token = first text chunk (~0.4 s); the token time is recorded beside it
if not (a_text.t_first and 0.33 < a_text.t_first - a_text.t_send < 0.6):
    bad.append(f"text basis TTFT {a_text.t_first and a_text.t_first - a_text.t_send}")
if not (a_text.t_tok and 0.13 < a_text.t_tok - a_text.t_send < 0.33):
    bad.append(f"t_tok must be the usage-token chunk (~0.2 s): {a_text.t_tok and a_text.t_tok - a_text.t_send}")
# token basis (Inkling): first token = the usage chunk at ~0.2 s
if not (a_tok.t_first and 0.13 < a_tok.t_first - a_tok.t_send < 0.33):
    bad.append(f"token basis TTFT {a_tok.t_first and a_tok.t_first - a_tok.t_send}")
# summarize must not crash on a finished reply with no text (text basis: B has no t_first/t_last)
for basis in ("text", "token"):
    reqs = asyncio.run(run(basis == "token"))
    a = T.finish_args(T.parser().parse_args(["--model", "m", "--out", "/dev/null", "--duration", "5", "--warmup", "0",
                                             "--ttft-basis", basis]))
    t0 = min(r.t_send for r in reqs) - 0.01
    try:
        s = T.summarize(a, {}, {}, reqs, t0, time.perf_counter() + 1, time.perf_counter() + 1, None, None, 1)
    except Exception as e:  # noqa: BLE001
        bad.append(f"summarize ({basis} basis) crashed: {e!r}")
        continue
    want_missing = 1 if basis == "text" else 0
    if s["summary"]["completed_no_first_token"] != want_missing or s["summary"]["ttft_basis"] != basis:
        bad.append(f"{basis}: completed_no_first_token {s['summary']['completed_no_first_token']} (want {want_missing})")
    if s["requests"][0]["ttft_token"] is None:
        bad.append(f"{basis}: per-request ttft_token missing")
print("FAIL\n  " + "\n  ".join(bad) if bad else "OK: traffic first-token basis (text / token) and no-text replies summarize")
sys.exit(1 if bad else 0)
