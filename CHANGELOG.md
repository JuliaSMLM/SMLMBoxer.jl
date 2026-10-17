# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and versions follow
Julia's rule for 0.Y.Z: raising Z breaks nothing, raising Y is an interface break. Releases before
0.3.1 are described in the [GitHub release notes](https://github.com/JuliaSMLM/SMLMBoxer.jl/releases).

## [Unreleased]

## [0.3.1] - 2026-10-17

### Fixed
- `backend=:auto` now filters on the CPU when it falls back to the CPU. In 0.3.0 the fallback still
  ran the DoG convolution on the GPU, for both the standard and the sCMOS variance-weighted filter,
  so on a busy GPU `getboxes` threw an out-of-memory error instead of returning CPU results, and on
  an idle GPU it used the device it had given up on. A regression test checks that the fallback
  allocates no GPU memory and returns the same boxes as `backend=:cpu` (#18).
- The package loads on Julia 1.13: `[compat]` now allows CUDA 6 and cuDNN 6. CUDA 5.9.4, the newest
  version 0.3.0 allowed, fails to precompile on Julia 1.13 (#16).
