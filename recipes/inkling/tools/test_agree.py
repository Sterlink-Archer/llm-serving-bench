#!/usr/bin/env python3
"""Guard for tools/agree.py (2026-11-22: the first version pooled GSM8K and MMLU-Pro before applying the GSM8K-only 0.90
floor, so every config failed the gate at ~0.88 pooled). Synthetic evalrun outputs; no network.

  python3 -I recipes/inkling/tools/test_agree.py
"""
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import agree  # noqa: E402


def evalfile(d, name, gsm_ok, mmlu_ok, trunc=0):
    rows = []
    for i in range(100):
        ok = i < gsm_ok
        rows.append(dict(task="gsm8kor", id=f"g{i}", gold="#### 7", text=f"step\n#### {7 if ok else 8}", reasoning="r",
                         finish="length" if i >= 100 - trunc else "stop"))
    for i in range(100):
        ok = i < mmlu_ok
        rows.append(dict(task="mmlupro", id=f"m{i}", gold="A", text=f"Answer: {'A' if ok else 'B'}", reasoning="r", finish="stop"))
    p = pathlib.Path(d) / name
    p.write_text(json.dumps(dict(split="t", rows=rows)))
    return str(p)


def main():
    with tempfile.TemporaryDirectory() as d:
        ref = evalfile(d, "ref.json", 95, 77)
        good = evalfile(d, "good.json", 96, 78)       # GSM8K 0.96 >= 0.90, pooled 0.87 < 0.90: must PASS
        r = agree.compare(ref, good)
        assert r["tasks"]["gsm8kor"]["acc_cfg"] == 0.96 and r["pooled"]["acc_cfg"] == 0.87, r
        assert r["gate_pass"], "GSM8K-only floor must not be applied to the pooled accuracy"
        worse = evalfile(d, "worse.json", 80, 60)     # far worse than the reference on both tasks: must FAIL
        r2 = agree.compare(ref, worse)
        assert not r2["gate_pass"] and r2["pooled"]["mcnemar_p"] < 0.05, r2
        trunc = evalfile(d, "trunc.json", 95, 77, trunc=5)
        assert agree.compare(ref, trunc)["tasks"]["gsm8kor"]["truncated_cfg"] == 5
    print("test_agree: ok")


if __name__ == "__main__":
    main()
