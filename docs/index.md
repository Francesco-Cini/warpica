# warpica

Numerical spacetime metrics, stress-energy tensors and sampled energy-condition
analysis in Python. `warpica` is a port of
[WarpFactory](https://github.com/NerdsWithAttitudes/WarpFactory), the MATLAB
toolkit for analysing warp-drive geometries.

!!! note "Credit to WarpFactory"
    The physics, the algorithms and the structure of this package come from
    WarpFactory, created by Christopher Helmerich and Jared Fuchs, with
    contributions from Alexey Bobrick, Luke Sellers, Brandon Melcher, Justin Feng
    and Gianni Martire. This is an independent port, not an official WarpFactory
    release. If you use it in published work, please cite the
    [WarpFactory paper](https://doi.org/10.1088/1361-6382/ad2e42)
    ([arXiv:2404.03095](https://arxiv.org/abs/2404.03095)).

## Where to start

- [Quickstart](quickstart.md): from an install to an energy-condition result.
- [Concepts](concepts.md): tensor layout, units and indexing.
- [How it works](how-it-works.md): the pipeline, with a glossary of the symbols.
- [Moving from MATLAB WarpFactory](matlab-migration.md).
- [API reference](reference.md), generated from the docstrings.

## Stability

The CPU path is stable and is what the 1.x compatibility promise covers. The CUDA
(CuPy) and DirectML backends are experimental. See the [policy](policy.md) and
[GPU validation](gpu.md). Results are checked against MATLAB WarpFactory, and the
places where this port deliberately differs are listed in
[MATLAB comparisons](reference-validation.md).

The source, issue tracker and changelog are on
[GitHub](https://github.com/Francesco-Cini/warpica). warpica is MIT licensed.
