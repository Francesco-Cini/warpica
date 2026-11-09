# How it works

This page follows one metric through the library, from a grid of numbers to an
energy-condition map, and names the file that does each step. Read it before
opening the source: the code is a close port of the MATLAB original and uses
physics notation, so a map helps. The symbols are in the [glossary](#glossary).

## The pipeline

```text
metric_get_*            a metric g_mu_nu sampled on a (t, x, y, z) grid
      |
get_energy_tensor       solver: derivatives -> curvature -> Einstein tensor -> T^mu_nu (SI)
      |
eval_metric             analyser: runs the steps below and collects the results
      |-- get_energy_conditions   null / weak / strong / dominant maps
      |-- get_scalars             expansion, shear, vorticity
```

### 1. The metric

A generator such as `metric_get_alcubierre` fills sixteen NumPy arrays, one per
component `g[mu][nu]`, each shaped `(t, x, y, z)`. Internally the 3+1 builder
(`three_plus_one_builder`) assembles them from a lapse, a shift and a spatial
metric. See [concepts](concepts.md) for the dictionary layout.

### 2. The solver: from metric to stress-energy

`get_energy_tensor` (in `warpica/solver/`) applies Einstein's equation
`G_mu_nu = (8 pi G / c^4) T_mu_nu` in reverse: it knows the geometry and finds the
matter that would be needed to produce it. It calls `met_2_den` (fourth order) or
`met_2_den_2` (second order), which chain these helpers in `solver/utils/`:

| Step | Function | What it computes |
| --- | --- | --- |
| 1 | `c4_inv` | the inverse metric `g^mu_nu`, cell by cell, from explicit formulas |
| 2 | `take_finite_difference_1`, `_2` | first and mixed second derivatives of every `g_mu_nu` |
| 3 | `ricci_t` | the Ricci tensor `R_mu_nu` from those derivatives |
| 4 | `ricci_s` | the Ricci scalar `R = g^mu_nu R_mu_nu` |
| 5 | `ein_t` | the Einstein tensor `G_mu_nu = R_mu_nu - R g_mu_nu / 2` |
| 6 | `ein_e` | `T^mu_nu = c^4 / (8 pi G) * G_alpha_beta g^alpha_mu g^beta_nu` (SI units) |

`ricci_t_2` and the other `*_2` files in `solver/utils/second_order/` are the
second-order versions of the same steps.

Details worth knowing:

- **Finite differences.** Fourth order uses the five-point central stencil with
  weights `(1, -8, 0, 8, -1) / 12h`; second order uses three points. The first
  derivative needs at least 5 (or 3) samples along the axis. Near the grid edge
  the stencil does not fit, so the edge cells copy the nearest fully computed
  value. Treat results within two cells (fourth order) of the boundary as
  unreliable and keep the interesting region away from the edges. Mixed second derivatives are not copied at all: their edge cells are exactly 0.
- **Time is stored in seconds, curvature is computed in `ct`.** Time derivatives
  are divided by `c` (and by `c^2` for a second time derivative) so all four
  coordinates have units of length.
- **Symmetry saves work.** `g_mu_nu` and `R_mu_nu` are symmetric, so only the
  upper triangle is computed and copied.
- **Devices.** With `gpu=` set, steps 1 to 5 run on the device and the result is
  copied back; the SI scaling in step 6 always runs on the CPU in float64 because
  `c^4 / (8 pi G)` is too large for float32.

### 3. The analyser: from stress-energy to answers

`eval_metric` first calls `get_energy_tensor`, then:

1. **Move to the observers' frame** (`do_frame_transfer`). Energy conditions are
   statements about what an observer measures. For each grid cell,
   `get_eulerian_transformation_matrix` builds an orthonormal tetrad `M` whose
   time leg is the Eulerian observer: the one at rest relative to the constant-time
   slices, with 4-velocity `n = (1/alpha, -beta^i/alpha)`. In that frame the metric is
   exactly Minkowski, `M^T g M = eta`, so every later step raises and lowers
   indices with `eta` instead of the real metric.
2. **Sample observers** (`generate_uniform_field`). A finite set of directions
   `num_angular_vec` is spread evenly over a sphere. Null conditions use light-like
   vectors `(1, n)`; weak and strong conditions also sample `num_time_vec` speeds
   `v` between 0 and 1 (never exactly 1), giving timelike vectors `(1, v n)`
   scaled so the four components have unit
   Euclidean length. That fixes the overall scale of the reported values, not their sign.
3. **Evaluate** (`get_energy_conditions`) for each sampled vector `u` at every
   cell, with `T` the Eulerian stress-energy:

   | Condition | Quantity tested | Violated when |
   | --- | --- | --- |
   | Null | `T_mu_nu k^mu k^nu` | negative |
   | Weak | `T_mu_nu u^mu u^nu` (energy density seen by `u`) | negative |
   | Strong | `(T_mu_nu - T eta_mu_nu / 2) u^mu u^nu` | negative |
   | Dominant | the energy flux `-T^mu_nu u^nu` must be timelike or null and future directed | its size (sign flipped) is negative |

   The map keeps the **worst** value over all sampled vectors at each cell: the
   minimum for the first three, and for the dominant condition the maximum before
   the sign flip, so that negative always means "violated". `return_vec=True`
   also returns every sampled value.
4. **Scalars** (`get_scalars`). It takes the covariant derivative of the observer
   4-velocity `u` (`cov_div`), projects it onto the space orthogonal to `u`, and
   splits it into a trace (the **expansion**, how fast volumes grow), a
   trace-free symmetric part (the **shear**, deformation without volume change)
   and an antisymmetric part (the **vorticity**, rotation).

## Where the numerical choices live

| Choice | Where | Notes |
| --- | --- | --- |
| Finite-difference order and edges | `solver/utils/take_finite_difference_*.py` | see above |
| Minimum grid size per axis | `solver/utils/verify_grid_size.py` | raises `ValueError` |
| Cartesian-only check | `solver/verify_tensor.py` | non-Cartesian `coords` fail |
| Direction and speed sampling | `analyser/utils/generate_uniform_field.py` | strictly timelike speeds |
| Observer frame | `analyser/utils/get_eulerian_transformation_matrix.py` | closed-form tetrad |
| Differences from MATLAB | [reference-validation.md](reference-validation.md) | seven deliberate ones |

## Glossary

Indices `mu, nu, alpha, beta` run over 0 to 3 (t, x, y, z); `i, j, k` over the
spatial 1 to 3 unless a derivative axis is meant. In code they are zero-based list
positions: `g[1][2]` is `g_xy`.

| Symbol | Name in code | Meaning |
| --- | --- | --- |
| `g_mu_nu` / `gl` | `gl`, `metric["tensor"]` | covariant metric (indices down) |
| `g^mu_nu` / `gu` | `gu`, `g_u` | inverse metric (indices up) |
| `eta` | Minkowski metric | `diag(-1, 1, 1, 1)`, signature (-, +, +, +) |
| `alpha` | `alpha` | lapse: proper time elapsed per coordinate time for an observer at rest relative to the slices |
| `beta^i` | `beta_up`, `array_beta` | shift vector: how the slice's coordinates are dragged between slices |
| `gamma_ij` | spatial metric | the 3x3 metric of one time slice |
| `delta` | `delta`, `metric["scaling"]` | grid spacings `[dt, dx, dy, dz]` |
| `Gamma` | Christoffel symbol | metric derivatives packed into connection coefficients (`get_christoffel_sym`) |
| `R_mu_nu`, `R` | `R_munu`, `R` | Ricci tensor and Ricci scalar |
| `G_mu_nu` / `E` | `E` | Einstein tensor (the code calls it `E`) |
| `T^mu_nu` | `energy_tensor` | stress-energy tensor |
| `u`, `n` | `u_up`, `u_down` | observer 4-velocity, indices up or down |
| `M` | `M` | tetrad matrix from the coordinate frame to the observer's orthonormal frame |
| `theta`, `sigma`, `omega` | `theta`, `shear`, `omega` | expansion, shear and vorticity tensors |
| `P` | `P`, `P_mix` | projector onto space orthogonal to `u` |
| `cov_div` | `cov_div` | covariant derivative |
| `ct` basis | | time measured as `c * t`, in metres |
