"""Prompt templates and answer extraction for the fixed eval set. Shared by bench/evalrun.py (in the sandbox) and
analysis/quality.py (on the PC, which re-scores from the raw saved outputs), so a scoring change re-scores every run."""
from __future__ import annotations

import re

LETTERS = "ABCDEFGHIJ"


def gsm8k_prompt(item: dict) -> str:
    return ("Solve the following math problem. Show your reasoning briefly, then give the final answer on the last line "
            "in the form 'Answer: <number>'.\n\n" + item["question"])


def mmlupro_prompt(item: dict) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(item["options"]))
    return (f"The following is a multiple choice question about {item['category']}. Think step by step briefly, then "
            f"finish with a last line of the form 'Answer: X' where X is the letter of the correct option.\n\n"
            f"Question: {item['question']}\n\nOptions:\n{opts}")


def gsm8kor_prompt(item: dict) -> str:
    """OpenRelay's gate prompt, verbatim from inference-recipes @6767b4d gemma-4-31b-mi355x/bench/gsm8k.py."""
    return item["question"] + "\nSolve step by step, then give the final answer as a plain number on the last line in the form: #### <number>"


def agree_prompt(it):
    """greedy-agreement set (data/eval/agree_400.jsonl): the prompt is stored whole; nothing is scored, outputs are compared
    across configs by analysis/quality.py."""
    return it["prompt"]


PROMPTS = {"gsm8k": gsm8k_prompt, "mmlupro": mmlupro_prompt, "gsm8kor": gsm8kor_prompt, "agree": agree_prompt}


def _or_num(s):
    """OpenRelay's num(): last number after removing ',' and '$'."""
    m = re.findall(r"-?\d+(?:\.\d+)?", (s or "").replace(",", "").replace("$", ""))
    return float(m[-1]) if m else None

_num = r"-?\$?\d[\d,]*(?:\.\d+)?"


def _norm_num(s: str):
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    return int(v) if v == int(v) else round(v, 6)


def extract(task: str, text: str):
    text = text or ""
    if task == "gsm8k":
        m = re.findall(r"Answer:\s*\**\s*(" + _num + ")", text, flags=re.I)
        if m:
            return _norm_num(m[-1])
        m = re.findall(_num, text)
        return _norm_num(m[-1]) if m else None
    if task == "gsm8kor":   # OpenRelay's scoring: '#### <n>' if present, else the last number in the reply
        m = re.search(r"####\s*([-\d.,$]+)", text)
        return _or_num(m.group(1)) if m else _or_num(text)
    if task == "agree":
        return None
    if task == "mmlupro":
        m = re.findall(r"Answer:\s*\**\s*\(?([A-J])\)?\b", text)
        if m:
            return m[-1]
        m = re.findall(r"answer is\s*\(?([A-J])\)?\b", text, flags=re.I)
        return m[-1] if m else None
    raise KeyError(task)


def is_correct(task: str, pred, gold) -> bool:
    if pred is None:
        return False
    if task == "gsm8kor":
        return abs(pred - _or_num(str(gold).split("####")[-1])) < 1e-6
    if task == "gsm8k":
        return _norm_num(str(gold)) == pred
    return str(pred).upper() == str(gold).upper()
