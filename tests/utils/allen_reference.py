"""Vendored-Allen reference outputs for the parity tests.

Allen is slow, so its outputs are computed once per cached specimen
(``uv run python tests/build_reference.py``) and stored as JSON in
``tests/testdata/allen_reference/``. For every long-square, short-square and ramp
sweep the reference holds the analysis window and detection parameters that
Allen's ``cell_extractor_for_nwb`` uses, the full spike table, all sweep
features and the burst and pause metrics; for long squares also the delay metrics
and the double-exponential fit behind them. Per cell it holds the ``EphysCellFeatureExtractor`` outputs, the hero
sweep, baseline and first-spike means of ``extract_cell_features``, Allen's
adjusted short-square detection parameters, and these cell features under their
Allen DB names (``record``).

Allen runs on float64 copies of the NWB traces (NWB data is float32), which is
exactly what the JAX side receives, see ``allen_sweeps``. Known, intended
differences between jaxley_ephys and this reference are documented in
``allen_ephys/ALLEN_EPHYS_FIXES.md`` (section 6).
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

from tests.utils.allen_dataset import load_specimen_case

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = REPO_ROOT / "tests" / "testdata" / "allen_reference"
# Bump when the contents of the reference files change.
FORMAT_VERSION = 4
KINDS = ("long_square", "short_square", "ramp")
SWEEP_FEATURES = (
    "adapt",
    "latency",
    "isi_cv",
    "mean_isi",
    "median_isi",
    "first_isi",
    "avg_rate",
    "v_baseline",
    "tau",
    "sag",
    "peak_deflect",
    "peak_deflect_index",
    "stim_amp",
)


# First-spike features in the Allen DB, with suffixes _long_square, _short_square
# and _ramp.
DB_SPIKE_FEATURES = (
    "fast_trough_t",
    "fast_trough_v",
    "slow_trough_t",
    "slow_trough_v",
    "threshold_t",
    "threshold_v",
    "threshold_i",
    "peak_t",
    "peak_v",
    "trough_t",
    "trough_v",
    "upstroke_downstroke_ratio",
)


def source_hash() -> str:
    """Hash of the vendored Allen code and ``FORMAT_VERSION``, to detect stale references."""
    paths = sorted((REPO_ROOT / "allen_ephys").glob("*.py"))
    digest = hashlib.sha256(b"".join(path.read_bytes() for path in paths))
    digest.update(str(FORMAT_VERSION).encode())
    return digest.hexdigest()[:16]


def reference_path(specimen_id: int) -> Path:
    return REFERENCE_DIR / f"specimen_{specimen_id}.json"


def allen_sweeps(specimen_id: int):
    """The specimen's sweeps, set up as by Allen's ``cell_extractor_for_nwb``.

    Returns ``(cell, {kind: [EphysSweepFeatureExtractor, ...]})`` with float64
    traces. Each extractor carries its sweep's analysis window (``start``,
    ``end``) and detection parameters (``dv_cutoff``, ``thresh_frac``, which
    Allen adjusts for short squares).
    """
    cell = load_specimen_case(specimen_id).build_cell_extractor_for_nwb()
    sweeps = {
        "long_square": cell._long_squares_ext.sweeps(),
        "short_square": cell._short_squares_ext.sweeps(),
        "ramp": cell._ramps_ext.sweeps(),
    }
    for kind_sweeps in sweeps.values():
        for sweep in kind_sweeps:
            sweep.t, sweep.v, sweep.i = (
                np.asarray(x, dtype=np.float64) for x in (sweep.t, sweep.v, sweep.i)
            )
    return cell, sweeps


def _float(value) -> float:
    return float("nan") if value is None else float(value)


def _sweep_record(sweep, kind: str) -> dict:
    df = sweep._spikes_df
    features = {
        name: _float(sweep.sweep_feature(name, allow_missing=True))
        for name in SWEEP_FEATURES
        if name not in ("peak_deflect", "peak_deflect_index")
    }
    deflect_v, deflect_index = sweep.sweep_feature("peak_deflect")
    features["peak_deflect"] = float(deflect_v)
    features["peak_deflect_index"] = float(deflect_index)
    patterns = {
        "burst": [float(x) for x in sweep.burst_metrics()],
        "pause": [float(x) for x in sweep.pause_metrics()],
    }
    if kind == "long_square":  # delay metrics are meant for (and slow on) others
        patterns |= _delay_record(sweep)
    return {
        "sweep_number": int(sweep.id),
        "kind": kind,
        "start": float(sweep.start),
        "end": float(sweep.end),
        "filter": float(sweep.filter),
        "dv_cutoff": float(sweep.dv_cutoff),
        "thresh_frac": float(sweep.thresh_frac),
        "n_spikes": len(df),
        "spikes": {col: df[col].tolist() for col in df.columns} if len(df) else {},
        "features": features,
        "patterns": patterns,
    }


def _delay_record(sweep) -> dict:
    """Allen's delay metrics and the double-exponential fit behind them: its
    parameters ``(A1, tau1, A2, tau2)``, SSE, and the fit window's length and first
    sample interval (s). ``delay_fit`` is None without a fit and ``"failed"`` where
    ``curve_fit`` failed and Allen fell back to a straight-line fit of log(y0 - v).
    """
    import scipy.optimize

    from allen_ephys import ephys_features as ft

    fits = []

    def recording_curve_fit(f, x, y, **kwargs):
        try:
            params, cov = scipy.optimize.curve_fit(f, x, y, **kwargs)
        except RuntimeError:
            fits.append("failed")
            raise
        sse = float(np.sum((f(x, *params) - y) ** 2))
        fits.append(
            {
                "params": params.tolist(),
                "sse": sse,
                "span": float(x[-1]),
                "first": float(x[1]),
            }
        )
        return params, cov

    with mock.patch.object(ft, "curve_fit", recording_curve_fit):
        delay = sweep.delay_metrics()
    return {"delay": [float(x) for x in delay], "delay_fit": fits[0] if fits else None}


def _mean_spike_0(sweeps) -> dict[str, float]:
    """Mean first-spike features over the spiking sweeps (Allen's ``mean_spike_0``)."""
    from allen_ephys.extract_cell_features import mean_features_spike_zero

    return {
        name: float(value) for name, value in mean_features_spike_zero(sweeps).items()
    }


def _hero_index(long_sweeps, rheobase_i: float) -> int:
    """Smallest-amplitude spiking long square in (rheobase + 39, rheobase + 61) pA.

    Same selection as inside Allen's ``extract_cell_features`` (which cannot be
    called on its own); it equals the Allen DB's thumbnail sweep on all cached
    specimens.
    """
    hero, hero_amp = -1, np.inf
    for k, sweep in enumerate(long_sweeps):
        amp = sweep.sweep_feature("stim_amp")
        spiking = sweep.sweep_feature("avg_rate") > 0 and len(sweep.spikes()) > 0
        if spiking and rheobase_i + 39.0 < amp < rheobase_i + 61.0 and amp < hero_amp:
            hero, hero_amp = k, amp
    return hero


def _cell_record(cell, sweeps) -> dict:
    from allen_ephys import ephys_extractor as efex
    from allen_ephys import ephys_features as ft

    features = cell.cell_features()
    long_sq, short_sq = features["long_squares"], features["short_squares"]
    keys = (
        "rheobase_i",
        "fi_fit_slope",
        "sag",
        "vm_for_sag",
        "input_resistance",
        "tau",
    )
    long_record = {key: _float(long_sq[key]) for key in keys}
    # extract_cell_features overrides the extractor's baseline with this mean.
    long_record["v_baseline"] = float(
        np.mean([sweep.sweep_feature("v_baseline") for sweep in sweeps["long_square"]])
    )
    long_record["rheobase_index"] = int(long_sq["rheobase_extractor_index"])
    long_record["hero_index"] = _hero_index(
        sweeps["long_square"], long_sq["rheobase_i"]
    )

    short_sweeps = sweeps["short_square"]
    cutoff, thresh_frac = ft.estimate_adjusted_detection_parameters(
        [sweep.v for sweep in short_sweeps],
        [sweep.t for sweep in short_sweeps],
        efex.SHORT_SQUARES_WINDOW_START,
        efex.SHORT_SQUARES_WINDOW_END,
    )
    common = short_sq["common_amp_sweeps"]
    ramps = features["ramps"]["spiking_sweeps"]
    short_record = {
        "stimulus_amplitude": _float(short_sq["stimulus_amplitude"]),
        "common_amp_sweeps": [int(sweep.id) for sweep in common],
        "mean_spike_0": _mean_spike_0(common),
        "dv_cutoff": float(cutoff),
        "thresh_frac": max(float(thresh_frac), 0.1),
    }
    ramp_record = {
        "spiking_sweeps": [int(sweep.id) for sweep in ramps],
        "mean_spike_0": _mean_spike_0(ramps),
    }
    hero_index = long_record["hero_index"]
    hero = sweeps["long_square"][hero_index] if hero_index >= 0 else None
    return {
        "long_square": long_record,
        "short_square": short_record,
        "ramp": ramp_record,
        "record": _db_record(
            long_record, long_sq["rheobase_sweep"], hero, short_record, ramp_record
        ),
    }


def _db_record(long_record, rheobase, hero, short_record, ramp_record) -> dict:
    """Cell features under their Allen Cell Types Database names: hero-sweep
    features, long-square features, the rheobase sweep's first spike and the
    first-spike means of short squares and ramps (no DB-matching heuristics)."""

    def hero_feature(name):
        return float("nan") if hero is None else _float(hero.sweep_feature(name))

    record = {
        "adaptation": hero_feature("adapt"),
        "avg_isi": hero_feature("mean_isi"),
        "latency": hero_feature("latency"),
        "f_i_curve_slope": long_record["fi_fit_slope"],
        "ri": long_record["input_resistance"],
        "sag": long_record["sag"],
        "tau": long_record["tau"],
        "vm_for_sag": long_record["vm_for_sag"],
        "vrest": long_record["v_baseline"],
    }
    first_spike = rheobase.spikes()[0]
    for name in DB_SPIKE_FEATURES:
        record[f"{name}_long_square"] = _float(first_spike[name])
        record[f"{name}_short_square"] = short_record["mean_spike_0"][name]
        record[f"{name}_ramp"] = ramp_record["mean_spike_0"][name]
    record["threshold_i_long_square"] = long_record["rheobase_i"]
    # As in Allen's current pipeline (ipfx); see ALLEN_EPHYS_FIXES.md, section 4.
    record["threshold_i_short_square"] = short_record["stimulus_amplitude"]
    return record


def build_reference(specimen_id: int) -> dict:
    """Run vendored Allen on all sweeps of a cached specimen."""
    from allen_ephys import ephys_extractor as efex

    cell, sweeps = allen_sweeps(specimen_id)
    with warnings.catch_warnings():
        # Allen warns about failed tau fits and empty slices; both yield NaN.
        warnings.simplefilter("ignore")
        cell.process()
        # Allen only sets the short-square amplitude calculator on the
        # common-amplitude sweeps; the sweep-level reference uses it everywhere.
        for sweep in sweeps["short_square"]:
            sweep.set_stimulus_amplitude_calculator(efex._short_step_stim_amp)
        records = {
            kind: [_sweep_record(sweep, kind) for sweep in kind_sweeps]
            for kind, kind_sweeps in sweeps.items()
        }
        cell_record = _cell_record(cell, sweeps)
    return {
        "specimen_id": specimen_id,
        "source_hash": source_hash(),
        "sweeps": records,
        "cell": cell_record,
    }


def write_reference(specimen_id: int) -> Path:
    path = reference_path(specimen_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_reference(specimen_id), allow_nan=True))
    return path


def is_current(specimen_id: int) -> bool:
    path = reference_path(specimen_id)
    return (
        path.exists() and json.loads(path.read_text())["source_hash"] == source_hash()
    )


def load_reference(specimen_id: int) -> dict:
    """Load a specimen's reference; fails with an actionable message if missing or stale."""
    path = reference_path(specimen_id)
    if not path.exists():
        raise FileNotFoundError(
            f"No Allen reference for specimen {specimen_id} ({path}). "
            "Run: uv run python tests/build_reference.py"
        )
    reference = json.loads(path.read_text())
    if reference["source_hash"] != source_hash():
        raise RuntimeError(
            f"Allen reference for specimen {specimen_id} is stale (allen_ephys or "
            "tests/utils/allen_reference.py changed). Run: "
            "uv run python tests/build_reference.py"
        )
    return reference
