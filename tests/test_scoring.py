#!/usr/bin/env python3
"""Known-answer tests for bench/scoring.py, including known-WRONG answers, so a scorer that marks everything correct
fails here (added after a 25/25 MMLU-Pro smoke result was questioned; that result was audited and is real).
  python3 -I tests/test_scoring.py"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "bench"))
import scoring as S  # noqa: E402

cases = [
    ("mmlupro", "reasoning...\nAnswer: G", "G", True),
    ("mmlupro", "reasoning...\nAnswer: G", "B", False),
    ("mmlupro", "I think (C) is plausible but Answer: D", "C", False),
    ("mmlupro", "the answer is (J)", "J", True),
    ("mmlupro", "no letter given at all", "A", False),
    ("gsm8k", "so 5*3=15\nAnswer: 15", "15", True),
    ("gsm8k", "so 5*3=15\nAnswer: 16", "15", False),
    ("gsm8k", "Answer: 1,250", "1250", True),
    ("gsm8kor", "steps\n#### 72", "Natalia sold...\n#### 72", True),
    ("gsm8kor", "steps\n#### 70", "Natalia sold...\n#### 72", False),
    ("gsm8kor", "the total is $72.", "...\n#### 72", True),
    ("gsm8kor", "", "...\n#### 72", False),
]
bad = []
for task, text, gold, want in cases:
    got = S.is_correct(task, S.extract(task, text), gold)
    if got != want:
        bad.append((task, text, gold, want, got))
print("FAIL", bad) if bad else print(f"scoring OK: {len(cases)} cases, {sum(1 for c in cases if not c[3])} known-wrong answers rejected")
sys.exit(1 if bad else 0)
