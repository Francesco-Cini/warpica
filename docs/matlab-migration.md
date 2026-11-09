# Moving from MATLAB WarpFactory

`warpica` follows the structure of WarpFactory, so MATLAB scripts translate
almost line by line. This page lists what changes. Intentional numerical
differences are in [reference-validation.md](reference-validation.md).

## Names

Function names are snake_case versions of the MATLAB names.

| MATLAB | Python |
| --- | --- |
| `metricGet_Minkowski` | `metric_get_minkowski` |
| `metricGet_Alcubierre`, `metricGet_AlcubierreComoving` | `metric_get_alcubierre`, `metric_get_alcubierre_comoving` |
| `metricGet_Lentz`, `metricGet_LentzComoving` | `metric_get_lentz`, `metric_get_lentz_comoving` |
| `metricGet_ModifiedTime`, `metricGet_ModifiedTimeComoving` | `metric_get_modified_time`, `metric_get_modified_time_comoving` |
| `metricGet_VanDenBroeck` | `metric_get_van_den_broeck` |
| `metricGet_Schwarzschild` | `metric_get_schwarzschild` |
| `metricGet_WarpShellComoving` | `metric_get_warp_shell_comoving` |
| `getEnergyTensor` | `get_energy_tensor` |
| `evalMetric` | `eval_metric` |
| `getEnergyConditions` | `get_energy_conditions` |
| `getScalars` | `get_scalars` |
| `getMomentumFlowLines` | `get_momentum_flow_lines` |
| `changeTensorIndex` | `change_tensor_index` |
| `doFrameTransfer` | `do_frame_transfer` |
| `plotTensor`, `plotThreePlusOne` | `plot_tensor`, `plot_three_plus_one` |

Rows from `metricGet_*` through `getScalars` and `doFrameTransfer` use names that
appear in this repository's MATLAB reference scripts. The remaining rows follow
the same renaming rule and were not checked against the upstream source. All
Python names import from `warpica.metrics`, `warpica.solver`, `warpica.analyser`
and `warpica.visualiser`.

## Conventions that differ

| Topic | MATLAB | Python |
| --- | --- | --- |
| Indices | one-based | zero-based in computation (`tensor[0][0]` is tt) |
| Plot slice arguments | one-based | still **one-based**, for compatibility |
| Components | cell array `tensor{mu,nu}` | nested lists `tensor[mu][nu]` |
| Structs | `metric.tensor` | dicts: `metric["tensor"]` |
| Arguments | positional | Python arguments; `diff_order` and `gpu` are keywords, e.g. `get_energy_tensor(metric, diff_order="fourth", gpu=None)` |
| Errors | `error(...)` and warnings | `ValueError` only; plain warnings only from `verify_tensor` |
| Plot output | opens a figure | returns `(fig, axes)`; you call `show()` or `savefig()` |
| GPU | `gpuArray` | `gpu=None` (CPU, stable), `"cupy"`, `"torch-directml"` (experimental) |

## Behaviour that differs on purpose

Seven deliberate numerical divergences, plus a few stricter checks, are described
and justified in [reference-validation.md](reference-validation.md). In short:

- Eulerian components are lowered with the Minkowski metric.
- Energy-condition speed sampling is strictly timelike.
- The dominant-energy flux is future directed.
- Expansion and shear use the real grid spacing.
- Modified Time and Van Den Broeck use the comoving `g00`.
- The warp shell interpolates linearly.
- Second-order stress-energy uses the exact `c` and `G`, so it is about 3.9e-5 smaller than
  MATLAB's rounded constants give.

The stricter checks: non-Cartesian `coords` fail verification instead of
warning, and a grid axis of 2 to 4 samples (fourth order) or 2 (second order)
raises `ValueError`, where upstream silently returns zero derivatives.

Because of these, a script that reproduces a published WarpFactory number may
differ here in exactly those places. Check the validation page before treating a
difference as a bug in either implementation.
