#!/usr/bin/env python3
"""analysis/quality.py on synthetic eval files with known answers: deltas, discordant counts, agreement, MTP pair.
  python3 -I tests/test_quality.py"""
import json
import pathlib
import shutil
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import quality as Q  # noqa: E402

tmp = pathlib.Path(tempfile.mkdtemp(prefix="or_q_"))
(tmp / "data" / "eval").mkdir(parents=True)
shutil.copy(ROOT / "data" / "eval" / "agree_400.jsonl", tmp / "data" / "eval" / "agree_400.jsonl")
Q.ROOT = tmp
agree_ids = [json.loads(l)["id"] for l in open(ROOT / "data" / "eval" / "agree_400.jsonl")][:10]


def write(run, tag, servers, gsm_right, agree_texts, rep2=None):
    rd = tmp / "results" / run / "20261122T080000Z"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "config.json").write_text(json.dumps({"snapshot": {"gpu": "H200"}, "servers": servers}))
    rows = [dict(task="gsm8kor", id=f"g{i}", gold="#### 7", text="#### 7" if ok else "#### 8") for i, ok in enumerate(gsm_right)]
    (rd / f"q_gsm8kor250_{tag}.json").write_text(json.dumps(dict(rows=rows)))
    arows = [dict(task="agree", id=i, gold=None, text=t) for i, t in zip(agree_ids, agree_texts)]
    (rd / f"q_agree400_{tag}.json").write_text(json.dumps(dict(rows=arows)))
    if rep2:
        (rd / f"q_agree400_{tag}_rep2.json").write_text(json.dumps(dict(rows=[dict(task="agree", id=i, gold=None, text=t) for i, t in zip(agree_ids, rep2)])))


ref_txt = [f"answer {i}" for i in range(10)]
write("q1", "bf16_ref", [{"name": "bf16_ref", "weights": [{"repo": "google/gemma-4-31B-it"}], "launch": "vllm serve --kv-cache-dtype auto"}],
      [True] * 8 + [False] * 2, ref_txt, rep2=ref_txt[:9] + ["answer X"])
write("q1", "fp8dyn", [{"name": "fp8dyn", "weights": [{"repo": "RedHatAI/gemma-4-31B-it-FP8-dynamic"}], "launch": "vllm serve --kv-cache-dtype fp8"}],
      [True] * 6 + [False, False, True, False], ref_txt[:5] + [f"answer {i}!" for i in range(5, 10)])
srv = [{"name": "or_recipe", "weights": [{"repo": "amd/gemma-4-31B-it-MXFP4"}], "launch": "vllm serve --kv-cache-dtype fp8 --speculative-config '{\"num_speculative_tokens\":2}'"},
       {"name": "or_recipe_nomtp", "weights": [{"repo": "amd/gemma-4-31B-it-MXFP4"}], "launch": "vllm serve --kv-cache-dtype fp8"}]
write("q2", "or_recipe", srv, [True] * 10, ref_txt)
write("q2", "or_recipe_nomtp", srv, [True] * 10, ref_txt[:8] + ["z", "z"])
sys.argv = ["quality.py", "--md", str(tmp / "Q.md")]
Q.main()
d = json.load(open(tmp / "results" / "quality_rows.json"))
by = {r["config"]: r for r in d["configs"]}
bad = []
f = by["q1:fp8dyn"]
if (f["gsm8kor250"]["b_lost"], f["gsm8kor250"]["c_gained"]) != (2, 1) or abs(f["gsm8kor250"]["delta_vs_ref"] - (-0.1)) > 1e-9:
    bad.append(f"fp8 paired counts {f['gsm8kor250']}")
if abs(f["agree_vs_ref"]["exact_match"] - 0.5) > 1e-9 or f["weights"].startswith("FP8") is False or f["kv_cache"] != "FP8":
    bad.append(f"fp8 agreement/meta {f['agree_vs_ref']['exact_match']} {f['weights']} {f['kv_cache']}")
if abs(by["q1:bf16_ref"]["agree_self_repeat"]["exact_match"] - 0.9) > 1e-9 or not by["q1:bf16_ref"]["is_reference"]:
    bad.append("bf16 self-repeat / reference flag")
if not d["mtp_lossless"] or abs(d["mtp_lossless"][0]["agree"]["exact_match"] - 0.8) > 1e-9 or by["q2:or_recipe"]["mtp"] != 2:
    bad.append(f"mtp pair {d['mtp_lossless']}")
print("FAIL\n  " + "\n  ".join(bad) if bad else "OK: quality deltas, paired counts, agreement, self-repeat and MTP pair")
sys.exit(1 if bad else 0)
