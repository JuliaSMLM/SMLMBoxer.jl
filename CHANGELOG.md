# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and versions follow
Julia's rule for 0.Y.Z: raising Z breaks nothing, raising Y is an interface break. Releases before
0.3.1 are described in the [GitHub release notes](https://github.com/JuliaSMLM/SMLMBoxer.jl/releases).

## [Unreleased]

## [0.3.2] - 2026-10-25

### Fixed
- `getboxes` threw `NVMLError: Not Supported (code 3)` for both `backend=:auto` and
  `backend=:gpu` on the NVIDIA DGX Spark GB10, whose driver does not report memory through
  NVML. When NVML cannot report a GPU's memory (not supported, no NVML library, or a Tegra
  board), SMLMBoxer now asks CUDA for that GPU's free memory and skips the NVML contention
  check for it; `recommend_batch_size` does the same. On the GB10 both backends now run on
  the GPU. Under `backend=:auto`, any other NVML or CUDA error from the GPU query warns and
  falls back to the CPU; under `backend=:gpu` it is thrown. A regression test, run on
  machines with a GPU, simulates both NVML failures in a separate Julia process and checks
  the backend chosen and the boxes against `backend=:cpu` (#24).

## [0.3.1] - 2026-10-17

### Changed
- `[compat]` allows CUDA 6 and cuDNN 6 alongside CUDA 4-5 and cuDNN 1.4.6 and later (#16).

### Fixed
- `backend=:auto` now filters on the CPU when it falls back to the CPU. In 0.3.0 the fallback still
  ran the DoG convolution on the GPU, for both the standard and the sCMOS variance-weighted filter,
  so on a busy GPU `getboxes` threw an out-of-memory error instead of returning CPU results, and on
  an idle GPU it used the device it had given up on. A regression test, run on machines with a GPU,
  checks that the fallback allocates no GPU memory and returns the same boxes as `backend=:cpu`
  (#18).
- Documentation: the Home and Examples pages unpack the return value as `(roi_batch, info)`;
  `SingleROI` is documented without a `camera` field, which it never had; `overlap` is described as
  the local-maximum window it sets, not a bound on box overlap; the photon-to-ADU factor is
  QE / gain (gain in e⁻/ADU), not QE × gain; pixel centres convert to µm as
  (k - 0.5) × pixel_size; and every Examples block runs against SMLMData 0.7, SMLMSim 0.7 and
  CUDA 6.
