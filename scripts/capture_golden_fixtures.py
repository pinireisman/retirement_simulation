#!/usr/bin/env python3
"""Capture golden fixtures for the single-portfolio Monte Carlo engine.

Run this script once to generate bit-exact reference outputs under
tests/fixtures/golden/.  The fixtures are then verified by
tests/test_golden_single_mode.py.

Usage:
    python scripts/capture_golden_fixtures.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

# Ensure repo root and tests/ are on the path so we can import our modules.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_repo_root, "tests"))

from engine.simulation import run_simulation
import golden_scenarios


def _fixture_dir() -> str:
    return os.path.join(_repo_root, "tests", "fixtures", "golden")


def _save_arrays(name: str, result: dict) -> None:
    """Save the core arrays to an .npz file."""
    path = os.path.join(_fixture_dir(), f"{name}.npz")
    kwargs: dict[str, np.ndarray] = {
        "bal_over_time": result["bal_over_time"],
        "prop_over_time": result["prop_over_time"],
        "final_portfolio": result["final_portfolio"],
        "final_property": result["final_property"],
        "estate_final": result["estate_final"],
        "ruined": result["ruined"],
    }
    # Guardrail scenarios have multiplier percentiles.
    gp = result.get("guardrail_mult_percentiles")
    if gp is not None:
        kwargs["mult_p10"] = gp["p10"]
        kwargs["mult_p50"] = gp["p50"]
        kwargs["mult_p90"] = gp["p90"]
    np.savez_compressed(path, **kwargs)
    print(f"  {path}")


def _save_meta(name: str, result: dict, params) -> None:
    """Save the summary + scalar metadata to a .meta.json file."""
    meta = {
        "summary": result["summary"],
        "mean_withdraw_rate": result["mean_withdraw_rate"].tolist(),
        "max_withdraw_rate": result["max_withdraw_rate"].tolist(),
        "mean_withdraw_value": result["mean_withdraw_value"].tolist(),
        "numpy_version": np.__version__,
        "n_paths": params.n_paths,
        "random_seed": params.random_seed,
        "annual": params.annual,
    }
    path = os.path.join(_fixture_dir(), f"{name}.meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  {path}")


def main() -> None:
    os.makedirs(_fixture_dir(), exist_ok=True)

    for name, builder in golden_scenarios.SCENARIOS.items():
        params, guardrails = builder()
        result = run_simulation(params, guardrails=guardrails)
        _save_arrays(name, result)
        _save_meta(name, result, params)


if __name__ == "__main__":
    main()
