# Versioning, deprecation and support policy

This policy takes effect with the 1.0 release. Until then the package is in
development and anything may change; the [changelog](https://github.com/Francesco-Cini/warpica/blob/main/CHANGELOG.md) records it.

## What the version number promises

`warpica` uses [semantic versioning](https://semver.org). For a release
`MAJOR.MINOR.PATCH`:

- **PATCH** releases fix bugs and change nothing you can rely on.
- **MINOR** releases add features and may deprecate things. Code that worked on
  `1.x` keeps working on `1.y` for `y > x`.
- **MAJOR** releases may remove deprecated things or change behaviour.

## What is covered

Covered: the names in `__all__` of `warpica.metrics`, `solver`, `analyser`,
`units` and `visualiser` (and their top-level re-exports): their signatures,
return layout and documented errors, on the **CPU path**.

Not covered, and free to change in any release:

- Internal modules: everything under a `utils` directory, `warpica.gpu`,
  `warpica._typing` and deep module paths.
- The **experimental GPU backends**, `gpu="cupy"` and `gpu="torch-directml"`,
  and the `cuda` and `directml` extras.
- Exact floating-point output. Results may change in the last digits between
  releases, and across NumPy versions and platforms.
- Type annotations of internal helpers.

## Numerical corrections

A fix to a wrong physical result is a bug fix, not a breaking change, even though
numbers change. Such fixes are listed prominently in the changelog. The
deliberate differences from upstream WarpFactory are a stable part of the
contract and are listed in [reference-validation.md](reference-validation.md);
changing one of them needs a MAJOR release.

## Deprecation

A public name or behaviour is deprecated in a MINOR release by emitting a
`DeprecationWarning` that names the replacement, and by a changelog entry. It
stays for at least two further MINOR releases and at least six months, and is
removed only in a MAJOR release.

## Supported versions

- **Python:** every CPython version that has not reached end of life, from 3.11.
  Dropping a version that reached end of life is done in a MINOR release.
- **NumPy, array-api-compat, Matplotlib:** the floors declared in `pyproject.toml`
  are the oldest versions the full test suite passes on. Raising a floor is
  announced in the changelog.
- **Fixes:** only the latest `1.x` release receives bug and security fixes.

## Reporting problems

See [SECURITY.md](https://github.com/Francesco-Cini/warpica/blob/main/SECURITY.md) for vulnerabilities and
[CONTRIBUTING.md](https://github.com/Francesco-Cini/warpica/blob/main/CONTRIBUTING.md) for bugs and proposals.
