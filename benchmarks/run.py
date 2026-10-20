"""Local benchmark: jaxley_ephys vs vendored Allen on a cached specimen (CPU).

    uv run python benchmarks/run.py           # compare with benchmarks/baseline.json
    uv run python benchmarks/run.py --save    # store the results as the new baseline

"first" is the first call (incl. compilation); other times are medians after it.
Needs the test data (see tests/load_testdata.py).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from allen_ephys import ephys_extractor as efex  # noqa: E402
from allen_ephys import ephys_features as ft  # noqa: E402
from jaxley_ephys import (  # noqa: E402
    detect_spikes,
    extract_cell_features,
    spike_features,
    sweep_features,
)
from tests.utils.allen_dataset import cached_specimen_ids  # noqa: E402
from tests.utils.allen_reference import KINDS, allen_sweeps  # noqa: E402
from tests.utils.synthetic import random_trace  # noqa: E402

BASELINE = Path(__file__).with_name("baseline.json")


def timed(fn, repeat: int = 1) -> float:
    times = []
    for _ in range(repeat):
        tic = time.perf_counter()
        fn()
        times.append(time.perf_counter() - tic)
    return statistics.median(times)


def jax_sweep(s, kind: str) -> None:
    sp = detect_spikes(
        s.t, s.v, s.i, s.start, s.end, dv_cutoff=s.dv_cutoff, thresh_frac=s.thresh_frac
    )
    jax.block_until_ready((spike_features(sp), sweep_features(sp, kind)))


def allen_sweep(s) -> None:
    ext = efex.EphysSweepFeatureExtractor(
        s.t,
        s.v,
        s.i,
        start=s.start,
        end=s.end,
        dv_cutoff=s.dv_cutoff,
        thresh_frac=s.thresh_frac,
    )
    ext.process_spikes()
    for name in ("v_baseline", "tau", "sag", "peak_deflect"):
        ext.sweep_feature(name)


def allen_cell(sweeps) -> None:
    """``cell_extractor_for_nwb`` + ``process`` without reading the NWB file."""

    def sweepset(kind, **kwargs):
        ss = sweeps[kind]
        return efex.EphysSweepSetFeatureExtractor(
            [s.t for s in ss],
            [s.v for s in ss],
            [s.i for s in ss],
            start=[s.start for s in ss],
            end=[s.end for s in ss],
            **kwargs,
        )

    short = sweeps["short_square"]
    cutoff, frac = ft.estimate_adjusted_detection_parameters(
        [s.v for s in short],
        [s.t for s in short],
        efex.SHORT_SQUARES_WINDOW_START,
        efex.SHORT_SQUARES_WINDOW_END,
    )
    short_ext = sweepset("short_square", dv_cutoff=cutoff, thresh_frac=max(frac, 0.1))
    efex.EphysCellFeatureExtractor(
        sweepset("ramp"), short_ext, sweepset("long_square")
    ).process()


def run() -> dict[str, dict[str, float]]:
    _, sweeps = allen_sweeps(cached_specimen_ids()[0])
    results = {}
    for kind in KINDS:
        s = sweeps[kind][-1]  # highest amplitude: the most spikes
        first = timed(lambda: jax_sweep(s, kind))
        results[f"{kind} sweep ({s.t.size} samples)"] = {
            "jax_first": first,
            "jax": timed(lambda: jax_sweep(s, kind), 3),
            "allen": timed(lambda: allen_sweep(s), 1 if kind == "ramp" else 3),
        }

    ls = sweeps["long_square"]
    results[f"{len(ls)} long squares, loop (per sweep)"] = {
        "jax": timed(lambda: [jax_sweep(s, "long_square") for s in ls]) / len(ls),
        "allen": timed(lambda: [allen_sweep(s) for s in ls]) / len(ls),
    }

    # The long squares share their time axis and window: keep those out of the
    # batch, so that detect_spikes can skip the samples after the window (no outer
    # jit, which would make XLA constant-fold everything computed from t).
    t, start, end = ls[0].t, ls[0].start, ls[0].end
    v, i = (jnp.stack([getattr(s, k) for s in ls]) for k in "vi")

    @jax.vmap
    def batch(v, i):
        sp = detect_spikes(t, v, i, start, end)
        return spike_features(sp), sweep_features(sp, "long_square")

    def run_batch():
        jax.block_until_ready(batch(v, i))

    first = timed(run_batch)
    results[f"{len(ls)} long squares, vmap (per sweep)"] = {
        "jax_first": first / len(ls),
        "jax": timed(run_batch, 3) / len(ls),
        "allen": results[f"{len(ls)} long squares, loop (per sweep)"]["allen"],
    }

    results |= synthetic_batch()

    as_tuples = {k: [(s.t, s.v, s.i, s.start, s.end) for s in sweeps[k]] for k in KINDS}

    def jax_cell():
        out = extract_cell_features(*(as_tuples[k] for k in KINDS))
        jax.block_until_ready(out["cell"])

    first = timed(jax_cell)
    results["whole cell"] = {
        "jax_first": first,
        "jax": timed(jax_cell),
        "allen": timed(lambda: allen_cell(sweeps)),
    }
    return results


def synthetic_batch(n: int = 128) -> dict[str, dict[str, float]]:
    """Many short traces of equal length, as from simulations (per trace)."""
    rng = np.random.default_rng(0)
    traces = [random_trace(rng) for _ in range(n)]
    t, v, i, start, end = (
        jnp.asarray(np.stack([tr[k] for tr in traces])) for k in range(5)
    )

    @jax.jit
    @jax.vmap
    def batch(t, v, i, start, end):
        sp = detect_spikes(t, v, i, start, end, max_spikes=64)
        return spike_features(sp), sweep_features(sp)

    def run_batch():
        jax.block_until_ready(batch(t, v, i, start, end))

    def loop():
        for t_k, v_k, i_k, start_k, end_k in traces:
            sp = detect_spikes(t_k, v_k, i_k, start_k, end_k, max_spikes=64)
            jax.block_until_ready((spike_features(sp), sweep_features(sp)))

    def allen():
        for t_k, v_k, i_k, start_k, end_k in traces:
            ext = efex.EphysSweepFeatureExtractor(
                t_k, v_k, i_k, start=start_k, end=end_k
            )
            try:
                ext.process_spikes()
                for name in ("v_baseline", "tau", "sag", "peak_deflect"):
                    ext.sweep_feature(name)
            except Exception:  # Allen fails on some synthetic traces
                pass

    first = timed(run_batch)
    allen_time = timed(allen) / n
    name = f"{n} synthetic traces ({t.shape[1]} samples)"
    loop_first = timed(loop)
    return {
        f"{name}, loop": {
            "jax_first": loop_first / n,
            "jax": timed(loop) / n,
            "allen": allen_time,
        },
        f"{name}, vmap": {
            "jax_first": first / n,
            "jax": timed(run_batch, 3) / n,
            "allen": allen_time,
        },
    }


def report(results: dict, baseline: dict | None) -> None:
    print(f"{'':42s} {'first':>8s} {'jax':>8s} {'allen':>8s} {'speedup':>8s}  baseline")
    for name, r in results.items():
        first = f"{r['jax_first']:8.2f}" if "jax_first" in r else " " * 8
        base = ""
        if baseline and name in baseline:
            base = f"{baseline[name]['jax']:8.3f} ({baseline[name]['jax'] / r['jax']:.1f}x)"
        print(
            f"{name:42s} {first} {r['jax']:8.3f} {r['allen']:8.3f} "
            f"{r['allen'] / r['jax']:7.1f}x  {base}"
        )
    print("times in s; speedup = allen / jax (after compilation)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save", action="store_true", help="store as the new baseline")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    results = run()
    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else None
    report(results, baseline)
    if args.save:
        BASELINE.write_text(json.dumps(results, indent=1))
        print(f"saved {BASELINE}")
