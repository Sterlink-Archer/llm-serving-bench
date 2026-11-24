"""Which protocol dirs in a run dir belong to which server: ONE rule, used by analysis/screen_report.py and
analysis/journal.py.

A protocol dir belongs to server S when
  1. run.json records it as one of S's steps (servers[S].steps[<dir name>], a bench/protocol.py command), whatever the
     dir is called -- 2026-11-22: Inkling's ikC_b300_mixed_dspark (dirs protocol_b300_*) and ikC_pro6000_mixed_dspark_rep_*
     (protocol_rep_B_text) were silently missing from claims.json because only `protocol_<server>*` was globbed; or
  2. its name starts with protocol_<S> and no longer server name claims it ('mx' never takes 'mx_recipe's dirs), for dirs
     added outside run.py's step list (e.g. a live extension pulled into the run dir).
unattributed() lists the protocol dirs with a summary that no server claims, so a report can say so instead of dropping
them.
"""
from __future__ import annotations

import pathlib


def _is_protocol_step(name, rec):
    return name.startswith("protocol") or "bench/protocol.py" in str((rec or {}).get("cmd", ""))


def protocol_dirs(rd, server, servers, run):
    rd = pathlib.Path(rd)
    names = sorted(set(servers) | {server}, key=len, reverse=True)
    recorded = {}
    for s in names:
        for st, rec in (((run or {}).get("servers") or {}).get(s, {}).get("steps") or {}).items():
            if _is_protocol_step(st, rec):
                recorded[st] = s
    out = []
    for p in sorted(rd.glob("protocol_*")):
        if not p.is_dir() or not (p / "protocol_summary.json").exists():
            continue
        owner = recorded.get(p.name) or next((s for s in names if p.name.startswith(f"protocol_{s}")), None)
        if owner == server:
            out.append(p)
    return out


def unattributed(rd, servers, run):
    rd = pathlib.Path(rd)
    claimed = {p for s in servers for p in protocol_dirs(rd, s, servers, run)}
    return [p for p in sorted(rd.glob("protocol_*")) if p.is_dir() and (p / "protocol_summary.json").exists() and p not in claimed]
