# Gradients

Every feature can be differentiated with `jax.grad`, e.g. with respect to the
parameters of a model whose simulated voltage is analysed (see the
[fitting example](examples/06_fitting)). The gradients are finite everywhere,
also where a feature is undefined (NaN), and the values stay Allen's: features
whose values change in steps get the gradients of smooth stand-ins.

| Features | Gradient |
| --- | --- |
| Voltages and dV/dt at spike events (`threshold_v`, `peak_v`, `upstroke`, `downstroke`, `upstroke_downstroke_ratio`, ...), `v_baseline`, `peak_deflect`, `min_deflect`, `sag`, input resistance | exact |
| `tau`, `delay_tau` | exact gradient of the fitted optimum (implicit differentiation) |
| Times of spike events (`threshold_t`, `peak_t`, `trough_t`, `upstroke_t`, `downstroke_t`, `fast_trough_t`, `adp_t`, `slow_trough_t`), `width`, and what is computed from them: `latency`, the interspike-interval features, `adapt`, `delay_ratio`, `max_burstiness`, `pause_frac` | of the times between samples: where the defining level is crossed (linear interpolation) or the vertex of a parabola through the extremum |
| Spike counts: `avg_rate` and the f-I slope | of a soft count: upward crossings of `min_peak`, counted through a sigmoid of width `count_width` (mV), which also reaches models that do not spike yet |
| Rheobase | of a soft minimum over the long squares, weighted by soft spike counts |
| `n_bursts`, `n_pauses`; which intervals are bursts or pauses; indices, `isi_type`, `clipped` | none: they change in steps |
| Currents (`*_i`, `stim_amp`) | with respect to `i` only |

Implicit differentiation, the soft counts and the times between samples add work
only when differentiating; the forward computation is unchanged.

## Features in a loss

- **Undefined features.** A NaN in a loss makes every gradient NaN. Features are
  NaN where they are undefined, e.g. the interspike intervals with fewer than two
  spikes or `tau` of a depolarizing sweep. `nan_to=0.0` (in `spike_features`,
  `sweep_features` and the sweepset and cell functions) replaces them; the
  gradients of such entries are zero.
- **Smooth event times.** With `detect_spikes(..., interpolate=True)`, the event
  times themselves are the times between samples (within a sample of Allen's),
  so that losses on them change smoothly. Without it, the values stay Allen's
  sample times with the gradients of the times between samples; for features
  that depend on sub-sample differences, such as the coefficient of variation or
  the adaptation of regular firing, these gradients can differ markedly from
  those of the interpolated values. `interpolate=True` is the consistent choice
  for fitting them.
- **Steps remain.** Spike counts, and with them the features that depend on
  which spikes exist, change in steps; the soft gradients point in the right
  direction, but a loss on them is only piecewise smooth. The fitting example
  therefore lowers the step size over time and keeps the best parameters.
