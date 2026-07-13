"""Runtime/memory benchmark: single-portfolio vs two-bucket strategy, at the
shapes PRD two_bucket_retirement_strategy §15 acceptance criterion #13 asks
for ("runtime stays within 2x baseline for the same simulation shape").

Run: .venv/bin/python scripts/benchmark_two_bucket.py
"""

import dataclasses
import os
import resource
import sys
import time

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_repo_root, "tests"))

from engine.simulation import run_simulation
from engine.withdrawal_strategies import ReserveConfig, WithdrawalStrategyConfig
import golden_scenarios


def _peak_rss_mb():
    # ru_maxrss is KB on Linux, bytes on macOS.
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024


def _bench(label, params):
    # ponytail: ru_maxrss is a process-lifetime high-water mark, not a
    # per-call delta, so the "delta" printed here is a lower bound once a
    # larger run has already run in-process. Good enough for a rough
    # single-vs-two_bucket comparison; a clean per-run number needs a
    # subprocess per measurement.
    rss_before = _peak_rss_mb()
    t0 = time.perf_counter()
    run_simulation(params, guardrails=None)
    dt = time.perf_counter() - t0
    rss_after = _peak_rss_mb()
    print(f"{label:32s} {dt:6.2f}s   peak RSS {rss_after:8.1f} MB "
          f"(delta this run: {rss_after - rss_before:+7.1f} MB)")
    return dt


def main():
    wcfg = WithdrawalStrategyConfig(
        type="two_bucket",
        reserve=ReserveConfig(target_years=4.0, refill_trigger_years=3.0),
    )
    for annual, shape in [(True, "annual"), (False, "monthly")]:
        base = golden_scenarios._base_params(random_seed=42, annual=annual, n_paths=10_000, end_age=95)
        two_bucket = dataclasses.replace(base, withdrawal_strategy=wcfg)
        t_single = _bench(f"single, {shape}, 10k paths", base)
        t_two = _bench(f"two_bucket, {shape}, 10k paths", two_bucket)
        print(f"  -> ratio {t_two / t_single:.2f}x\n")


if __name__ == "__main__":
    main()
