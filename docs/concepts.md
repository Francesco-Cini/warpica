# Concepts and conventions

The few rules that explain most surprises when using `warplab`.

## Metric and tensor dictionaries

A metric or stress-energy tensor is a plain `dict`:

| Key | Meaning |
| --- | --- |
| `type` | `"metric"` or `"Stress-Energy"` |
| `name` | free-text label |
| `coords` | `"cartesian"`; any other value is rejected by the solver and analyser |
| `index` | `"covariant"`, `"contravariant"`, `"mixedupdown"` or `"mixeddownup"` |
| `date` | free-text creation date |
| `scaling` | grid spacings `[dt, dx, dy, dz]` |
| `params` | the generator's arguments (metrics from built-in generators only) |
| `tensor` | the components |

`tensor[mu][nu]` is a NumPy array shaped `(time, x, y, z)`, so the `g_01`
component is `metric["tensor"][0][1]`. A single `(4, 4, t, x, y, z)` array is
also accepted as input. Malformed input raises `ValueError` when you pass it to
a solver or analyser function.

## Axis order

Every field uses `(t, x, y, z)`. Spacetime index 0 is time, 1 to 3 are x, y, z.

## Units

SI throughout: time spacing in seconds, space spacing in metres, stress-energy
in SI. Velocities such as the Alcubierre `v` are fractions of c. `warplab.units`
provides `c()` and `G()`. Internally the solver works in the `ct` basis, so it
converts time derivatives for you.

## Where the grid points are

Generators place sample `i` (counting from zero) at `(i + 1) * spacing -
world_centre`, WarpFactory's convention. For a bubble in the middle of an
`N`-sample axis use `world_centre = (N + 1) / 2 * spacing`.

## Zero- versus one-based indices

- **Computation is zero-based**, like all Python: `tensor[0][0]` is time-time and
  array positions start at 0.
- **Plot slice arguments are one-based**, kept from WarpFactory: in
  `plot_tensor(metric, sliced_planes=[1, 4], slice_locations=[1, 5])`, axis
  numbers 1 to 4 mean t, x, y, z and slice positions count from 1.

## Grid size limits

Derivatives need neighbours. Along each axis, one sample is allowed and means no
variation along that axis (a snapshot). Fourth order needs 5 or more samples and
second order 3 or more; an axis of 2 to 4 samples (fourth order) or 2 (second
order) raises `ValueError`. A single time slice therefore cannot capture the
time dependence of a moving bubble.

## Energy conditions are sampled

The null, weak, strong and dominant maps test a finite set of directions and
observer velocities (`num_angular_vec`, `num_time_vec`). A non-negative result
means "no violation found among those samples", not a proof. Values are
positive when satisfied and negative when violated; `keep_positive=False` clips
positive entries to zero so only violations show.

## `smooth_factor` counts samples

In `metric_get_warp_shell_comoving`, `smooth_factor` is the width of the moving
average applied to the shell's density and pressure profiles **in radial
samples**, not metres. A value strictly between 0 and 1 is read as a fraction of
the radial samples instead. The window is made odd and shrinks near the ends.
Refining the grid therefore changes the physical smoothing length unless you
scale `smooth_factor` with it.

## Stable and experimental parts

Only names exported by `warplab.metrics`, `solver`, `analyser`, `units` and
`visualiser` are public; see [api](api.md) and the [policy](policy.md). The CPU
backend is stable; `gpu="cupy"` and `gpu="torch-directml"` are experimental.
