#!/usr/bin/env python3
"""Test for mkcfg.check_confirm_ladders, the confirm-ladder guard (lead, 2026-11-22 after C2).

C2's committed configs, with the claim rates expected at launch, must be refused, for exactly the two gaps that cost the C2b
re-run: or_recipe column A (3.25/3.5, no rung below 3.25) and mx_stack column B (3.5/3.75, no rung below 3.5). The C2b
configs must pass. A confirm server with no declared expectation must be refused. write() must raise for a non-legacy
confirm that breaks the rule.

  python3 -I recipes/mi355x/tools/test_ladder_guard.py      # exit 0 = pass
"""
import copy
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mkcfg  # noqa: E402

# what we expected to claim when C2 launched (README / journal 122, 131): theirs A 3.25 (C1) and B 3.0 (claims.json);
# ours-today A 3.5, B 3.25; the stack A 3.5 (mxG2), B 3.5.
EXPECT_C2 = {"or_recipe": {"A": 3.25, "B": 3.0}, "mx_recipe": {"A": 3.5, "B": 3.25}, "mx_stack": {"A": 3.5, "B": 3.5}}


def c2_as_launched():
    out = []
    for f in sorted((HERE.parent / "configs").glob("mxC2_s*_r*.json")):
        c = json.loads(f.read_text())
        for sv in c["servers"]:
            sv["expect"] = EXPECT_C2[sv["name"]]
        out.append(c)
    assert len(out) == 4, out
    return out


def main():
    fails = []
    # 1. every C2 config is refused, and the two named gaps appear on the boxes that ran those columns
    seen = set()
    for c in c2_as_launched():
        bad = mkcfg.check_confirm_ladders(c)
        if not bad:
            fails.append(f"{c['label']}: C2 config NOT refused")
        for b in bad:
            seen.add(b.split(":")[0])
    for need in ("or_recipe column A", "mx_stack column B"):
        if need not in seen:
            fails.append(f"C2 refusal does not name '{need}' (got {sorted(seen)})")
    # 2. C2b passes
    for c in mkcfg.stack_confirm_b():
        bad = mkcfg.check_confirm_ladders(c)
        if bad:
            fails.append(f"{c['label']}: C2b refused: {bad}")
    # 3. a confirm server with no expectation is refused
    c = copy.deepcopy(mkcfg.stack_confirm_b()[0])
    c["servers"][0].pop("expect")
    if not any("no expected claim rate" in b for b in mkcfg.check_confirm_ladders(c)):
        fails.append("server without expect not refused")
    # 4. write() raises for a non-legacy confirm that breaks the rule (relabelled C2 config), and never writes the file
    c = c2_as_launched()[0]
    c["label"] = "mxZZ_guardtest"
    try:
        mkcfg.write(c)
        fails.append("write() accepted a ladder without a lower rung")
    except SystemExit as e:
        if "REFUSED" not in str(e):
            fails.append(f"write() raised the wrong error: {e}")
    if (mkcfg.OUT / "mxZZ_guardtest.json").exists():
        fails.append("refused config was written")
    # 5. non-confirm stages are not checked
    c = copy.deepcopy(c2_as_launched()[0])
    c["stage"] = "mx-lever"
    if mkcfg.check_confirm_ladders(c):
        fails.append("guard fired on a non-confirm stage")
    if fails:
        print("FAIL\n  " + "\n  ".join(fails))
        sys.exit(1)
    print(f"ok: 4 C2 configs refused ({sorted(seen)}); 2 C2b configs pass; missing-expect and write() refusals hold")


if __name__ == "__main__":
    main()
