"""Cell-level features, named as in the Allen Cell Types Database.

``cell_features`` collects the sweepset features of the three stimulus types
(see ``sweepset.py``). ``extract_cell_features`` runs the whole pipeline on the
sweeps of a cell, like Allen's ``extract_cell_features`` with
``cell_extractor_for_nwb``.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import Array

from .spikes import detect_spikes, spike_features
from .sweep import sweep_features
from .sweepset import (
    SHORT_SQUARE_WINDOW,
    SUBTHRESHOLD_MIN_AMP,
    Features,
    long_square_features,
    ramp_features,
    short_square_detection_params,
    short_square_features,
)

# Analysis windows of ``cell_extractor_for_nwb`` for Allen Cell Types data (s):
# long squares run from 1.02 to 2.02 s, ramps from 1.02 s to the end of the
# sweep's experiment epoch; short squares use the epoch as their window.
LONG_SQUARE_WINDOW = (1.02, 2.02)
RAMP_START = 1.02
# First-spike features of the database (with suffixes _long_square, _short_square
# and _ramp); threshold_i_long_square is the rheobase.
SPIKE_FEATURES = (
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

# A sweep: t (s), v (mV), i (pA) and the analysis window start, end (s).
Sweep = tuple[Array, Array, Array | None, float, float]


def cell_features(
    long_squares: Features,
    short_squares: Features | None = None,
    ramps: Features | None = None,
) -> dict[str, Array]:
    """Cell features with Allen Cell Types Database names from sweepset features.

    Takes the outputs of ``long_square_features``, ``short_square_features`` and
    ``ramp_features``; ``None`` for a missing stimulus type gives NaN for its
    features. Returns scalars:

    - from the hero sweep: ``adaptation``, ``avg_isi`` (ms), ``latency`` (s);
    - ``f_i_curve_slope``, ``ri`` (input resistance), ``sag``, ``tau``,
      ``vm_for_sag``, ``vrest`` (baseline voltage);
    - first-spike features: ``<feature>_long_square`` from the rheobase sweep
      (``threshold_i_long_square`` is the rheobase), ``<feature>_short_square``
      and ``<feature>_ramp`` averaged over sweeps (``mean_spike_0``).

    The database's sweep ids (``rheobase_sweep_id``, ``thumbnail_sweep_id``)
    correspond to ``rheobase_index`` and ``hero_index`` of the long squares.
    """
    hero = long_squares["hero_sweep"]
    out = {
        "adaptation": hero["adapt"],
        "avg_isi": hero["mean_isi"],
        "latency": hero["latency"],
        "f_i_curve_slope": long_squares["fi_fit_slope"],
        "ri": long_squares["input_resistance"],
        "sag": long_squares["sag"],
        "tau": long_squares["tau"],
        "vm_for_sag": long_squares["vm_for_sag"],
        "vrest": long_squares["v_baseline"],
    }
    nan = jnp.full_like(long_squares["rheobase_i"], jnp.nan)
    spike = long_squares["rheobase_spike"]
    for name in SPIKE_FEATURES:
        out[f"{name}_long_square"] = spike.get(name, nan)
    out["threshold_i_long_square"] = long_squares["rheobase_i"]
    for suffix, features in (("short_square", short_squares), ("ramp", ramps)):
        means = {} if features is None else features["mean_spike_0"]
        for name in SPIKE_FEATURES:
            out[f"{name}_{suffix}"] = means.get(name, nan)
    return out


def _stack(dicts: list[dict]) -> dict:
    return jax.tree.map(lambda *x: jnp.stack(x), *dicts)


def _sweepset(sweeps: Sequence[Sweep], stimulus: str, **detection) -> tuple:
    """Stacked sweep features and first-spike features of a set of sweeps."""
    per_sweep, first = [], []
    for t, v, i, start, end in sweeps:
        sp = detect_spikes(t, v, i, start, end, max_spikes=None, **detection)
        per_sweep.append(sweep_features(sp, stimulus))
        first.append({name: x[0] for name, x in spike_features(sp).items()})
    return _stack(per_sweep), _stack(first)


def extract_cell_features(
    long_squares: Sequence[Sweep],
    short_squares: Sequence[Sweep] = (),
    ramps: Sequence[Sweep] = (),
    *,
    subthresh_min_amp: float = SUBTHRESHOLD_MIN_AMP,
    short_square_window: tuple[float, float] = SHORT_SQUARE_WINDOW,
) -> dict[str, Features]:
    """Spike, sweep, sweepset and cell features of one cell.

    Each argument lists the sweeps of one stimulus type as ``(t, v, i, start,
    end)``: ``[T]`` arrays in s, mV and pA (lengths may differ) and the analysis
    window in s. For Allen Cell Types data, Allen uses ``LONG_SQUARE_WINDOW`` for
    long squares and starts ramps at ``RAMP_START``.

    As in Allen's ``cell_extractor_for_nwb``, short-square spikes are detected
    with ``short_square_detection_params``; all other sweeps use the defaults of
    ``detect_spikes``. Returns a dict with:

    - ``long_squares``, ``short_squares``, ``ramps``: the sweepset features, plus
      the stacked ``[S]`` sweep features (``sweeps``) and first-spike features
      (``first_spikes``); ``None`` if there are no sweeps of that type. The short
      squares also hold the ``dv_cutoff`` and ``thresh_frac`` used;
    - ``cell``: the output of ``cell_features``.

    Runs eagerly (one compiled detection per sweep length).
    """
    if not long_squares:
        raise ValueError("extract_cell_features needs at least one long square")
    sweeps, first = _sweepset(long_squares, "long_square")
    long_out = long_square_features(sweeps, first, subthresh_min_amp=subthresh_min_amp)
    out = {"long_squares": long_out | {"sweeps": sweeps, "first_spikes": first}}

    out["short_squares"] = None
    if short_squares:
        dv_cutoff, thresh_frac = short_square_detection_params(
            [s[0] for s in short_squares],
            [s[1] for s in short_squares],
            short_square_window,
        )
        sweeps, first = _sweepset(
            short_squares, "short_square", dv_cutoff=dv_cutoff, thresh_frac=thresh_frac
        )
        out["short_squares"] = short_square_features(sweeps, first) | {
            "sweeps": sweeps,
            "first_spikes": first,
            "dv_cutoff": dv_cutoff,
            "thresh_frac": thresh_frac,
        }

    out["ramps"] = None
    if ramps:
        sweeps, first = _sweepset(ramps, "ramp")
        out["ramps"] = ramp_features(sweeps, first) | {
            "sweeps": sweeps,
            "first_spikes": first,
        }
    out["cell"] = cell_features(out["long_squares"], out["short_squares"], out["ramps"])
    return out
