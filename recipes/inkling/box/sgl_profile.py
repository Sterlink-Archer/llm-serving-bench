"""sgl_profile.py: run bench/profile_prefill.py (the build session's prefill profiler, unchanged) against an SGLang server
whose /start_profile and /stop_profile are admin endpoints. This SGLang build (0.0.0.dev1+inkling2) rejects them with
HTTP 403 unless the server was started with --admin-api-key and the request carries that key as a Bearer token
(srt/utils/auth.py, AuthLevel.ADMIN_FORCE). The key is a throwaway local token from the server's env (IK_ADMIN_KEY),
never a credential. This wrapper adds the header to those two calls only, then hands over to profile_prefill.main().

  IK_ADMIN_KEY=... python3 sgl_profile.py --model inkling --dir /root/or/out/prof_x --n 4 --tokens 10000 --out ...json
"""
import os
import sys
import urllib.request

sys.path.insert(0, "/root/or/bench")
# run.py exports a server's env to the server job only, not to its steps, so the step falls back to the same throwaway
# local token the configs pass to --admin-api-key (ik7 base: HTTP 401 without it).
KEY = os.environ.get("IK_ADMIN_KEY") or "local-profiler-token"
_orig = urllib.request.urlopen


def _patched(req, *a, **k):
    url = req.full_url if isinstance(req, urllib.request.Request) else str(req)
    if KEY and ("/start_profile" in url or "/stop_profile" in url):
        if not isinstance(req, urllib.request.Request):
            req = urllib.request.Request(url, data=k.pop("data", None))
        req.add_header("Authorization", "Bearer " + KEY)
    return _orig(req, *a, **k)


urllib.request.urlopen = _patched
import profile_prefill  # noqa: E402

sys.argv = ["profile_prefill.py"] + sys.argv[1:]
sys.exit(profile_prefill.main())
