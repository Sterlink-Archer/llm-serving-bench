"""The $/token bar a config is priced against: ONE implementation, used by run.py, analysis/screen_report.py,
analysis/cost.py and analysis/journal.py.

A config picks its bar with AT MOST ONE of:
  - `price_model`: an OpenRelay model id (default "openrelay/gemma-4-31b"), priced at that model's OpenRelay list price from
    the newest research/openrelay_prices_*.json, e.g. "openrelay/glm-5.3-flash" for phase 3;
  - `price_bar`: {name, in_usd_per_m, out_usd_per_m, cached_usd_per_m?, source} for a model OpenRelay does not list (e.g. phase
    2's OpenRouter PrimeIntellect price for Inkling 2 Mini).
Setting both is an error. `is_openrelay_gemma` is True only for the default bar: the OpenRelay-vs-Gemma fields
(ratio_vs_openrelay_sync, openrelay_sync_usd_per_req, the OpenRouter Gemma comparison) are filled only then."""
from __future__ import annotations

import glob
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OPENRELAY_MODEL = "openrelay/gemma-4-31b"


def openrelay_prices(path=None):
    p = path or sorted(glob.glob(str(ROOT / "research/openrelay_prices_*.json")))[-1]
    return json.loads(pathlib.Path(p).read_text()), pathlib.Path(p).name


def openrelay_tiers(model=OPENRELAY_MODEL, prices=None):
    d, f = (prices, (prices or {}).get("saved_file") or "openrelay_prices (given)") if prices else openrelay_prices()
    if model not in d["models"]:
        raise KeyError(f"price_model {model!r} is not in research/{f}; listed: {sorted(d['models'])}")
    return d["models"][model]["tiers"], pathlib.Path(str(f)).name


def openrelay_sync(model=OPENRELAY_MODEL):
    t, f = openrelay_tiers(model)
    return t["Sync"], f


def price_model(cfg=None):
    return (cfg or {}).get("price_model") or OPENRELAY_MODEL


def price_bar(cfg=None):
    cfg = cfg or {}
    pb = cfg.get("price_bar")
    if pb and cfg.get("price_model"):
        raise ValueError(f"config {cfg.get('label')!r} sets both price_bar and price_model: pick one")
    if pb:
        return dict(pb, model=None, is_openrelay=False, is_openrelay_gemma=False)
    m = price_model(cfg)
    t, f = openrelay_sync(m)
    return dict(name=f"OpenRelay sync {m}", model=m, in_usd_per_m=t["in_usd_per_m"], out_usd_per_m=t["out_usd_per_m"],
                cached_usd_per_m=t.get("cached_in_usd_per_m"), source=f"research/{f}", is_openrelay=True,
                is_openrelay_gemma=(m == OPENRELAY_MODEL))


def bar_tiers(cfg=None, prices=None):
    """Sync (and Batch, when the source lists one) tiers in the openrelay_prices shape, for per-cell pricing in cost.py."""
    cfg = cfg or {}
    if cfg.get("price_bar"):
        b = price_bar(cfg)          # also refuses a config that sets both
        sync = dict(in_usd_per_m=b["in_usd_per_m"], out_usd_per_m=b["out_usd_per_m"], cached_in_usd_per_m=b.get("cached_usd_per_m"))
        return dict(name=b["name"], model=None, sync=sync, batch=None, is_openrelay_gemma=False)
    m = price_model(cfg)
    t, _ = openrelay_tiers(m, prices)
    return dict(name=f"OpenRelay sync {m}", model=m, sync=t["Sync"], batch=t.get("Batch"), is_openrelay_gemma=(m == OPENRELAY_MODEL))


def usd_per_req(bar, in_tok=9500, out_tok=380):
    return (in_tok * bar["in_usd_per_m"] + out_tok * bar["out_usd_per_m"]) / 1e6
