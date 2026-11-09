# Second-order solver

Ported from [WarpFactory secondOrder](https://github.com/NerdsWithAttitudes/WarpFactory/tree/03b10cb02e73998af28db87201a43f2fa0e30319/Solver/utils/secondOrder),
commit `03b10cb02e73998af28db87201a43f2fa0e30319`.
The upstream MIT copyright and permission notice is retained in the root LICENSE.

- Tensor components are indexed `[mu][nu]`; each component has axes `(t, x, y, z)`.
- First derivatives and unmixed second derivatives copy the nearest interior
  derivative to the boundary. Mixed second derivatives leave boundary faces zero.
- Axes shorter than three samples produce zero derivatives, matching upstream.
- `ricci_t_2` converts time derivatives using `c()`; Python time axis is zero.
- `ricci_t_mem_2` preserves the upstream memory variant's different convention:
  its first coordinate is `ct`, so its first grid spacing is a length.
- `ein_e_2` preserves the upstream rounded constants and unit formula;
  `units` has three entries (length, mass, time), with SI defaults `[1, 1, 1]`.
- Determinants and inverses work pointwise for nested component lists or arrays.
- Array allocation follows the input backend; DirectML hardware checks cover both orders; CUDA checks require an NVIDIA machine (see docs/gpu.md).

Use `get_energy_tensor(metric, "second")` for the public solver entry point.
