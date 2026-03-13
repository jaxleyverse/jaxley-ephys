from __future__ import annotations

import pytest

from jaxley_ephys.features import compute_sweepset_features


def _long_square_sweepset(specimen_case):
    records = specimen_case.records(kind="long_square")
    sweepset = specimen_case.make_sweep_set(records)
    sweepset.process_spikes()
    return sweepset, records


def _sweepset_arrays(specimen_case, records):
    t_set = []
    v_set = []
    i_set = []
    start_set = []
    end_set = []
    for record in records:
        t, v, i, start, end = specimen_case.trace_for(record.sweep_number)
        t_set.append(t)
        v_set.append(v)
        i_set.append(i)
        start_set.append(start)
        end_set.append(end)
    return t_set, v_set, i_set, start_set, end_set


def test_jax_sweepset_features_temporarily_disabled(specimen_case):
    _, records = _long_square_sweepset(specimen_case)
    t_set, v_set, i_set, start_set, end_set = _sweepset_arrays(specimen_case, records)
    with pytest.raises(NotImplementedError, match="temporarily disabled"):
        compute_sweepset_features(t_set, v_set, i_set, start_set, end_set)
