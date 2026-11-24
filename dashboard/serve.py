#!/usr/bin/env python3
"""Serve the built dashboard locally (loopback only).

  python3 -I dashboard/serve.py [--port 8765] [--demo] [--build] [--watch 60]
"""
import argparse
import functools
import http.server
import pathlib
import subprocess
import sys
import threading
import time

DASH = pathlib.Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--demo", action="store_true", help="serve dashboard/site-demo (mock data)")
    ap.add_argument("--build", action="store_true", help="rebuild before serving")
    ap.add_argument("--watch", type=int, default=0, metavar="SEC", help="rebuild every SEC seconds while serving (live runs)")
    a = ap.parse_args()
    if a.build:
        subprocess.run([sys.executable, "-I", str(DASH / "build.py")] + (["--demo"] if a.demo else []), check=True)
    root = DASH / ("site-demo" if a.demo else "site")
    if not (root / "index.html").exists():
        sys.exit(f"{root} not built: python3 -I dashboard/build.py{' --demo' if a.demo else ''}")
    if a.watch:
        def loop():
            while True:
                time.sleep(a.watch)
                r = subprocess.run([sys.executable, "-I", str(DASH / "build.py")] + (["--demo"] if a.demo else []), capture_output=True, text=True)
                print(time.strftime("%H:%M:%SZ", time.gmtime()), (r.stdout or r.stderr).strip().splitlines()[0][:160] if (r.stdout or r.stderr) else r.returncode, flush=True)
        threading.Thread(target=loop, daemon=True).start()
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    with http.server.ThreadingHTTPServer(("127.0.0.1", a.port), h) as srv:
        print(f"serving {root} at http://127.0.0.1:{a.port}/  (Ctrl-C to stop)", flush=True)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
