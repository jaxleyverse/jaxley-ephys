# Relation to the AllenSDK

jaxley-ephys reimplements the electrophysiology feature extraction of the
[AllenSDK](https://github.com/AllenInstitute/AllenSDK) (`ephys_features`,
`EphysSweepFeatureExtractor`, `EphysCellFeatureExtractor` and
`extract_cell_features`), which the Allen Institute no longer maintains. The
repository contains a maintained copy of that code, `allen_ephys`, fixed for
current versions of NumPy, SciPy and pandas and for a few edge cases. It keeps its
original license, the Allen Institute Software License (BSD-2-Clause with an
additional clause that does not permit redistribution for commercial purposes).

jaxley-ephys is tested against `allen_ephys` on every sweep of eight cells of the
Allen Cell Types Database: spike tables, sweep features and cell features agree to
floating-point precision (relative 1e-9), apart from the documented differences
below. The largest one is the delay metric (`delay_metrics`,
`sweep_features(..., delays=True)`): where Allen's optimizer misses a better
double-exponential fit, jaxley-ephys uses it, which changes about a third of the
long squares, some by a lot (section 6).

```{include} ../allen_ephys/ALLEN_EPHYS_FIXES.md
:heading-offset: 1
```
