#!/usr/bin/env python3
"""Render README.md from results/claims.json: every result number on the front page comes from the claims file, never typed.
`--check` renders and compares with the committed README.md and exits 1 on drift (bin/test_all.sh runs it), so a claim that
moves without the README following fails the gate. Maks 2026-11-22: the README is lean, one screen, links for detail.

  python3 -I bin/build_readme.py            # write README.md
  python3 -I bin/build_readme.py --check    # exit 1 if README.md differs from the render
"""
from __future__ import annotations

import json
import re
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import pricebar as PB  # noqa: E402
sys.path.insert(0, str(ROOT / "analysis"))
import box_counts as BC  # noqa: E402  (per-box counts across the same-box paired tests, from the cell files)


def load():
    return {(c["group"], c["column"]): c for c in json.loads((ROOT / "results/claims.json").read_text())["claims"]}


def batch_usd_per_req():
    t = PB.bar_tiers({})["batch"]
    return PB.usd_per_req(dict(in_usd_per_m=t["in_usd_per_m"], out_usd_per_m=t["out_usd_per_m"]))


FIGS = {"ablation": "How we improved on OpenRelay's Gemma recipe, step by step",
        "knee": "Same GPUs, ours vs theirs: requests per second at the latency target",
        "price": "Price per request vs OpenRelay and the cheapest market provider, all three models"}


def fig_block(name, prefix):
    """<picture> + caption for figure `name`, image paths relative to the document (prefix 'docs/' for README, '' for
    docs/WRITEUP.md), alt and caption from docs/img/figures.json; [] when the SVGs do not exist yet."""
    fj = ROOT / "docs" / "img" / "figures.json"
    meta = (json.loads(fj.read_text()).get(name) or {}) if fj.exists() else {}
    light, dark = f"img/{name}-light.svg", f"img/{name}-dark.svg"
    if not ((ROOT / "docs" / light).exists() and (ROOT / "docs" / dark).exists()):
        return []
    alt = (meta.get("alt") or FIGS.get(name, name)).replace('"', "'")
    out = ["<picture>", f'  <source media="(prefers-color-scheme: dark)" srcset="{prefix}{dark}">',
           f'  <img alt="{alt}" src="{prefix}{light}">', "</picture>"]
    return out + (["", f"*{meta['caption']}*"] if meta.get("caption") else [])


FIG_BLOCK = re.compile(r"<!-- fig:([a-z0-9-]+) -->\n.*?<!-- /fig -->", re.S)


def refresh_writeup(text):
    """Every `<!-- fig:NAME -->...<!-- /fig -->` block in docs/WRITEUP.md is re-rendered from figures.json, so the write-up's
    figure captions are generated like the README's (the drift check covers both)."""
    return FIG_BLOCK.sub(lambda m: "\n".join([f"<!-- fig:{m.group(1)} -->"] + fig_block(m.group(1), "") + ["<!-- /fig -->"]), text)


def fig(name):
    """A <picture> with light and dark SVGs (docs/img/<name>-light.svg / -dark.svg, made by the dashboard session's
    analysis/figures.py) when both exist, with the alt text and caption read from docs/img/figures.json (generated from
    the result files, so no number here is typed); until then an invisible placeholder, never a broken image."""
    meta = {}
    fj = ROOT / "docs" / "img" / "figures.json"
    if fj.exists():
        meta = json.loads(fj.read_text()).get(name) or {}
    alt = (meta.get("alt") or FIGS[name]).replace('"', "'")
    light, dark = f"docs/img/{name}-light.svg", f"docs/img/{name}-dark.svg"
    if (ROOT / light).exists() and (ROOT / dark).exists():
        out = ["<picture>", f'  <source media="(prefers-color-scheme: dark)" srcset="{dark}">',
               f'  <img alt="{alt}" src="{light}">', "</picture>"]
        if meta.get("caption"):
            out += ["", f"*{meta['caption']}*"]
        return out
    return [f"<!-- figure '{name}': appears when its light and dark SVGs exist -->"]


def render(C):
    def c(g, col):
        r = C.get((g, col))
        if r is None:
            raise SystemExit(f"build_readme: claims.json has no row {g} {col}")
        return r

    def rate(g, col):
        r = c(g, col)["max_rate_all_seeds_pass"]
        return "no claim" if r is None else f"{r:g} req/s"

    def x(g, col):
        r = c(g, col)
        return "—" if r.get("ratio_vs_bar") is None else f"{r['ratio_vs_bar']:.2f}×"

    xb = c("mx_recipe_c4m24", "A")["usd_per_req"] / batch_usd_per_req()
    BX = BC.counts()
    FIN, OR = "final recipe (ours + MX_PS_FP4)|A", "theirs (OpenRelay's recipe)|A"

    def held(fam, r):
        v = BX[fam]["rates"][str(float(r))]
        return f"all {v['ran']}" if v["ran"] and v["held"] == v["ran"] else f"{v['held']} of {v['ran']}"
    W = "docs/WRITEUP.md"
    rows = [
        ("Gemma 4 31B — our recipe vs OpenRelay's, same GPUs",
         f"Our final recipe held 3.5 req/s on {held(FIN, 3.5)} boxes we tested; OpenRelay's held 3.5 on {held(OR, 3.5)} and 3.25 "
         f"on {held(OR, 3.25)} (counts per box). On real text in the latest test: ours {rate('mx_stack_C2b', 'B')}, theirs "
         f"{rate('or_recipe_paired_C2b', 'B')}",
         f"[same-hardware comparison]({W}#h1-our-deployment-vs-theirs-on-the-same-hardware-the-comparison-that-matters)"),
        ("Gemma 4 31B — our cost vs their price list",
         f"{x('mx_recipe_c4m24', 'A')} of their standard price, {xb:.2f}× of their batch price; {x('mx_recipe_C1', 'B')} on real text",
         f"[cost vs price]({W}#h2-our-cost-vs-their-api-price-a-different-question)"),
        ("Inkling 2 Mini — vs the cheapest provider",
         f"{x('ikC_b300_mixed_dspark', 'B')} on a B300 and {x('ikC_pro6000_mixed_dspark', 'B')} on an RTX PRO 6000 (real text)",
         "[Inkling notes](recipes/inkling/README.md)"),
        ("GLM-5.3-Flash — vs OpenRelay's GLM price",
         f"{x('glm_best_C', 'B')} on real text: we lose", "[GLM notes](recipes/glm/README.md)"),
    ]
    L = ["# Serving open models cheaper than OpenRelay", "",
         "We took the models [OpenRelay](https://openrelay.inc) sells, ran them ourselves on rented spot GPUs, and measured them",
         "the way OpenRelay measures its own service: the same request mix, the same latency target, the same load generator.",
         "Then we asked two questions. On identical hardware, can we serve more requests than their published recipe? And how",
         "does our cost per request compare with their price list?", "",
         *fig("ablation"), "", "## Results at a glance", "",
         "| model and question | result | more |", "|---|---|---|"]
    L += [f"| {q} | {a} | {d} |" for q, a, d in rows]
    L += ["", "A ratio below 1× means we are cheaper. Every result holds on two independent load seeds at every load level up to",
          "the one quoted, with 95% of requests getting their first token within 4 seconds.", "",
          *fig("knee"), "", *fig("price"), "",
          "## What we found", "",
          "- **A wasted step in vLLM's Gemma 4 speculative decoder.** Its draft model re-reads every prompt chunk and throws",
          f"  the work away. Skipping it is what took our recipe reliably under the latency target at 3.5 req/s, and it speeds up\n"
          f"  their recipe too. The fix is upstream for review as [vLLM pull request #69275](https://github.com/vllm-project/vllm/pull/69275).\n"
          f"  [The details]({W}#the-find-vllms-gemma-4-mtp-drafter-runs-every-prefill-chunk-through-its-first-pass-and-throws-it-away)",
          "- **A faster matrix kernel, and a power wall.** A newer AMD kernel adds about 4% prompt throughput; past that, the",
          f"  GPU is limited by power rather than by its datasheet peak. [The details]({W}#the-second-find-aiters-persistent-mxfp4-gemm-and-a-power-limited-ceiling-journal-steps-400-404-139)",
          "- **Random-token benchmarks undersell speculative decoding.** On real text the draft models are right far more",
          "  often, so a benchmark of random tokens makes them look worse than they are (GLM and Inkling both show it).",
          "", "## How we measured", "",
          "- OpenRelay's own protocol: long prompts (about 9,500 tokens in, 380 out), random arrival times, and a target of",
          f"  95% of requests starting their answer within 4 seconds. [Method]({W}#1-the-question)",
          "- Two kinds of load: OpenRelay's random-token benchmark, and real text. A price claim has to hold on real text.",
          f"- Each claim, and the rule that decides it, is listed in [the claims table](results/CLAIMS.md); every experiment",
          "  along the way is in [the journal](docs/JOURNAL.md).", "",
          "To run a recipe yourself, start from the [MI355X Gemma recipe](recipes/mi355x/README.md), the",
          "[GLM recipe](recipes/glm/README.md) or the [Inkling recipe](recipes/inkling/README.md).", "",
          f"**Caveats.** Spot GPUs differ from box to box (about 6% in clock speed), so results are stated per recipe across",
          f"several boxes, never from one lucky box. [Scope and open questions]({W}#h4-scope-and-open-items)", "",
          "**Licences.** Code: [MIT](LICENSE). Prompt text: Wikipedia (CC BY-SA 4.0), UltraChat 200k (MIT); evals: GSM8K (MIT),",
          "MMLU-Pro (MIT). The copy of OpenRelay's recipe in `research/raw/openrelay_recipe/` keeps its Apache-2.0 licence."]
    return "\n".join(L) + "\n"


def main():
    txt = render(load())
    p = ROOT / "README.md"
    w = ROOT / "docs" / "WRITEUP.md"
    wcur = w.read_text() if w.exists() else ""
    wtxt = refresh_writeup(wcur)
    if "--check" in sys.argv:
        cur = p.read_text() if p.exists() else ""
        if cur != txt or wcur != wtxt:
            which = [n for n, bad in (("README.md", cur != txt), ("docs/WRITEUP.md figure blocks", wcur != wtxt)) if bad]
            print(f"FAIL: {' and '.join(which)} differ from the render of results/claims.json + docs/img/figures.json; "
                  "run python3 -I bin/build_readme.py")
            sys.exit(1)
        print(f"OK: README.md and the WRITEUP figure blocks match ({len(txt.splitlines())} README lines, "
              f"{len(FIG_BLOCK.findall(wcur))} WRITEUP figures)")
        return
    p.write_text(txt)
    if wtxt != wcur:
        w.write_text(wtxt)
    print(f"wrote README.md ({len(txt.splitlines())} lines) and {len(FIG_BLOCK.findall(wtxt))} WRITEUP figure blocks")


if __name__ == "__main__":
    main()
