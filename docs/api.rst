API reference
=============

Spike detection
---------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   jaxley_ephys.detect_spikes
   jaxley_ephys.Spikes
   jaxley_ephys.calculate_dvdt
   jaxley_ephys.find_time_index

Spike features
--------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   jaxley_ephys.spike_features

Sweep features
--------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   jaxley_ephys.sweep_features
   jaxley_ephys.sweep.avg_rate
   jaxley_ephys.sweep.latency
   jaxley_ephys.sweep.adapt
   jaxley_ephys.sweep.isi_cv
   jaxley_ephys.sweep.mean_isi
   jaxley_ephys.sweep.median_isi
   jaxley_ephys.sweep.first_isi
   jaxley_ephys.sweep.v_baseline
   jaxley_ephys.sweep.voltage_deflection
   jaxley_ephys.sweep.peak_deflect
   jaxley_ephys.sweep.peak_deflect_index
   jaxley_ephys.sweep.min_deflect
   jaxley_ephys.sweep.min_deflect_i
   jaxley_ephys.sweep.tau
   jaxley_ephys.sweep.sag
   jaxley_ephys.sweep.step_amplitude
   jaxley_ephys.sweep.short_step_amplitude

Firing patterns
---------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   jaxley_ephys.pause_metrics
   jaxley_ephys.burst_metrics
   jaxley_ephys.delay_metrics

Sweepset features
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   jaxley_ephys.long_square_features
   jaxley_ephys.short_square_features
   jaxley_ephys.ramp_features
   jaxley_ephys.short_square_detection_params
   jaxley_ephys.sweepset.mean_first_spike

Cell features
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   jaxley_ephys.cell_features
   jaxley_ephys.extract_cell_features
