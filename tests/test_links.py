#!/usr/bin/env python3
"""Every relative link and image (markdown ![..](..), <img src>, <source srcset>) in every tracked Markdown file resolves: the target file exists, and a #anchor matches a heading
(GitHub's slug rule). Maks 2026-11-22: a lean README that is mostly links, so a dead link is a broken page.
python3 -I tests/test_links.py"""
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)|(?:src|srcset)=\"([^\"\s]+)\"")


def slug(h):
    s = h.strip().lower()
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


def anchors(md):
    out, fence = set(), False
    for ln in md.read_text(errors="replace").splitlines():
        if ln.lstrip().startswith("```"):
            fence = not fence
        elif not fence and ln.startswith("#"):
            out.add(slug(ln.lstrip("#")))
    return out


files = [ROOT / p for p in subprocess.run(["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True).stdout.split()]
files += [ROOT / "README.md"] if (ROOT / "README.md") not in files else []
bad = []
for md in files:
    if not md.exists() or "research/raw" in str(md):
        continue
    fence = False
    for i, ln in enumerate(md.read_text(errors="replace").splitlines(), 1):
        if ln.lstrip().startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        for t in (m[0] or m[1] for m in LINK.findall(ln)):
            if re.match(r"^[a-z]+:", t) or t.startswith("mailto:"):
                continue
            path, _, anc = t.partition("#")
            tgt = (md.parent / path).resolve() if path else md
            if not tgt.exists():
                bad.append(f"{md.relative_to(ROOT)}:{i}: {t} (no such file)")
            elif anc and tgt.suffix == ".md" and anc.lower() not in anchors(tgt):
                bad.append(f"{md.relative_to(ROOT)}:{i}: {t} (no heading '{anc}')")
print("FAIL\n  " + "\n  ".join(bad[:40]) + (f"\n  ... {len(bad)} in all" if len(bad) > 40 else "") if bad
      else f"OK: every relative link resolves in {len(files)} Markdown files")
sys.exit(1 if bad else 0)
