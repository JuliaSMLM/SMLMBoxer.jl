# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and versions follow
Julia's rule for 0.Y.Z: raising Z breaks nothing, raising Y is an interface break. Releases before
0.3.1 are described in the [GitHub release notes](https://github.com/JuliaSMLM/SMLMBoxer.jl/releases).

## [Unreleased]

### Fixed
- With an `SCMOSCamera`, PSF-aware detection (`psf_sigma` with `min_photons`) now works in
  photons per pixel: each pixel becomes `(adu - offset) * gain / qe`, weighted by the
  variance `(readnoise / qe)^2`, and the threshold needs no gain conversion. 0.4.0 converted
  `min_photons` with the camera's mean QE / gain, so with a gain map the threshold in photons
  varied by pixel: at a pixel whose gain is 10% above the map's mean, `min_photons = 500`
  meant 550 photons. Boxes are still cut from the ADU image. The direct interface
  (`sigma_small`, `sigma_large`, `minval`) still filters the image as given. On this path the
  offset must be finite.
- An sCMOS readnoise, gain or QE that is zero, negative or not finite (scalar, or any pixel of
  a map) raises `ArgumentError` naming the field and the number of pixels. A zero readnoise
  previously made the filter return NaN: no boxes for a scalar 0, and none within the large
  Gaussian's window of a zero pixel. For a noise-free readout, use an IdealCamera with the
  image in photons.
- A NaN or Inf pixel in a float image raises `ArgumentError` with the count and the first
  (row, col, frame), also under `:auto`; it previously removed or shifted detections nearby.
  Integer images skip the check. On the GPU the check runs on the device: about 0.4 ms per
  1000 frames of 256 x 256 (0.5% of the call), with nothing left in the memory pool.
- `min_photons` without `psf_sigma` raises `ArgumentError`; it was silently ignored, leaving a
  threshold of 0 that made every local maximum a box. Its keyword default is now `nothing`;
  with `psf_sigma` and no `min_photons`, 500 is used and stored in `BoxerConfig`.
- An image smaller than `boxsize` in either dimension, or an array of more than 3 dimensions,
  raises `ArgumentError` before filtering (previously `BoundsError` or `ErrorException`). A
  stack of zero frames returns an empty ROIBatch instead of a `DivideError`.
- `BoxerConfig` and `getboxes` validate their parameters and raise `ArgumentError` naming the
  field: sigmas must be finite with `0 < sigma_small < sigma_large`, `psf_sigma` and
  `min_photons` finite and positive, `minval` and `overlap` finite (`overlap >= 0`),
  `boxsize >= 1`, `backend` one of `:cpu`, `:gpu`, `:auto`, and timeouts `>= 0`. Swapped
  sigmas previously returned rings of false boxes, and `psf_sigma = 0` returned no boxes.
- `psf_sigma` with a camera whose pixels are not square, or not all the same size, raises
  `ArgumentError` (relative tolerance 1e-4); the PSF width was converted with the x pixel size
  only. Such cameras can use `sigma_small` and `sigma_large` in pixels. The pixel size is now
  each axis's span over its pixel count, not its first pixel's width: a Float32 camera of
  0.1 µm pixels whose edges start at 100,000 µm got 0.1016 µm from that width. Pixel widths may differ
  by the rounding of the stored edges beyond the tolerance, so a uniform 2048-pixel camera
  stored in Float32 passes.
- sCMOS read noise, gain and QE stored at low precision (Float16) no longer overflow the noise
  variance, which is now computed in Float64 on both paths: with read noise 20 and QE 0.05, the
  photon variance of 160,000 was infinite and a bright spot gave no box.

## [0.4.0] - 2026-11-02

### Breaking
- `min_photons` now uses the actual DoG filter response to a pixel-integrated Gaussian,
  replacing the empirical 0.65 factor. A noise-free spot centered on a pixel just reaches
  the threshold at that photon count. At a PSF sigma of 1.35 pixels (0.135 µm on
  0.1 µm pixels), a pixel-corner spot needs about 1.1× as many photons; at sigma
  1 pixel it needs about 1.2×, and at 0.7 pixel about 1.5×. The standard and sCMOS
  paths use their own kernels. The same setting gives a lower filter threshold than
  before, so it finds more boxes. At a PSF sigma of 1.35 px, multiplying `min_photons` by 1.19 (IdealCamera) or
  1.12 (sCMOS) gives 0.3.4's threshold (0.65 / 0.547 and 0.65 / 0.581).
- Images must match their camera's (rows, columns), or raise `DimensionMismatch`.
  A crop passed with its full camera previously returned boxes with coordinates silently
  wrong by the crop's offset.
- sCMOS variance weighting now uses `(readnoise / gain)^2` in ADU², including maps.
  This fixes the units: the image is in ADU, so the variance must be in ADU².

### Fixed
- Increasing `overlap` now permits closer neighbors according to the max-pool window.
  Maxima sharing that window have equal intensities; the earlier detection is kept.
- Even box sizes use the same clamped-corner rule as odd sizes; box sizes 8 and 10
  could previously raise `DimensionMismatch`. Odd-size crops remain unchanged.
- After clamping, keep only the brightest filtered detection per (corner, frame);
  0.3.4 could return duplicate boxes.
- 0.3.4's known issue is fixed: under CUDA.jl 6.5.0, a `getboxes` call now returns
  its GPU memory to the driver.
- The chunked `recommend_batch_size` example offsets frame indices by the chunk start.
  The lab caller already tracks this offset; no new keyword is needed.

### Changed
- The release after each successful GPU call no longer runs a full garbage collection.
  Owned GPU arrays are explicitly freed within each batch, also on errors. From CUDA.jl
  6.2, when `reclaim()` started running a full GC, release uses blocking synchronization,
  cache purging and pool trimming; CUDA.jl 5, 6.0 and 6.1 retain their GC-free reclaim.
  A warmed-up 256×256×10 GPU call on the RTX 4090 falls from about 57 ms in 0.3.4 to
  0.75 ms (medians of 0.73–0.84 ms across CUDA.jl 6.4.2 and 6.5.0). IdealCamera GPU
  throughput on 500 frames at 256×256 rises from 5,329 to 13,805 images/s. The GPU wins
  from one frame at all tested sizes (128, 256 and 512 px), with either one or 32 CPU
  threads on the AMD Ryzen 9 5950X. Release after a short call takes
  about 0.02 ms; returning 2 GiB to the driver takes about 42 ms.
  The lab pipeline (SMLMAnalysis with GaussMLE, which caps CUDA.jl at 5.11.3) does not
  see the 57 ms: under 5.11.3, `reclaim()` has no GC, and the same call takes 1.73 ms
  in 0.3.4 and 1.40 ms in 0.4.0.
- Removed unused internal `find_best_gpu`, `select_backend`, `wait_for_gpu_memory`,
  and `estimate_gpu_memory` helpers, the discarded per-box camera ROI construction, and
  the uncalled internal `extract_camera_roi` and `pixels_to_microns`.
  The per-frame memory estimator is unchanged.

## [0.3.4] - 2026-11-02

### Changed
- The CPU `getboxes` path is precompiled for direct parameters, IdealCamera and sCMOS
  filtering, and the `BoxerConfig` calling convention, without initializing a GPU. The
  first CPU call in a fresh process (256 x 256 x 2, after loading) took 3.0 s
  (IdealCamera) and 6.0 s (sCMOS) in 0.3.3 and 8-12 s before this workload; it now takes
  0.04 s and 0.4 s under KernelAbstractions 0.9, 1.0 s and 1.4-2.7 s under 0.10.
  Precompiling SMLMBoxer takes longer in exchange, about 13.5 s (0.9) and 16.3 s (0.10)
  instead of about 4.5 s, once per install.
- SMLMBoxer no longer depends on cuDNN. On the GPU, standard DoG convolution and
  local-maximum pooling now use KernelAbstractions kernels; the CPU path still uses NNlib.
  Environments loading SMLMBoxer can use current CUDA and Reactant together: cuDNN's
  CUDNN_jll no longer conflicts with Reactant's bundled cuDNN in one process. Reactant
  0.2.304 and CUDA 5.11.3 now resolve and run next to SMLMBoxer on Julia 1.12 and 1.13.
  The sCMOS path finds the same boxes as 0.3.3 (with #19's fix below) on CPU and GPU;
  on the RTX 4090 a `getboxes` call at the default `min_photons` takes within 5% of
  0.3.3's time per frame, and less under KernelAbstractions 0.10 (#20).
- KernelAbstractions compat is `0.9.38 - 0.10`: Reactant environments resolve 0.9, while
  current CUDA and AMDGPU environments resolve 0.10. GPU runs stay NVIDIA only; AMDGPU
  can sit in the same environment, but SMLMBoxer does not use it. Under 0.10 the CPU
  backend is OpenCL (POCL): the sCMOS filter on the CPU then rounds slightly differently
  (up to 4 boxes per 100,000 change at a low threshold) and runs about 25% slower than
  under 0.9.

### Fixed
- A repeatable GPU out-of-memory error retried the same batch until the timeout, forever
  by default under `backend=:gpu`. Resource retries now halve the batch cap immediately;
  only a single-frame failure resumes device polling. A frame too large for every visible
  GPU fails at once under `:gpu`, or warns and uses the CPU under `:auto`. Timeout errors
  report the failed batch's frame count and estimated memory need.
- With an IdealCamera or no camera, the DoG filter treated the outside of the image as
  zero, so a background produced false boxes along the image edges. Edge pixels are now
  filtered with the in-bounds part of each Gaussian, as the sCMOS path already did;
  pixels whose whole filter window lies inside the image are filtered exactly as before.
  Boxes change only within a few pixels of the edges: on simulated frames with a
  10 photon/px background, the share of boxes clamped at an edge falls from 72% to 15% at
  `min_photons = 50`.
- Without a camera, the ROIBatch's default IdealCamera was one pixel larger than the
  image on each axis. Box positions were and are unchanged.
- Overlap removal now applies to every frame of a stack; previously it applied only to
  frame 1 (#19). Overlap removal no longer allocates per pair of boxes (a 600-box frame
  takes 0.25 ms instead of 19 ms).
- `getboxes` retries GPU memory errors and an unavailable device until `auto_timeout`
  under `backend=:auto`, then uses the CPU; any other GPU failure warns and uses the CPU
  at once. Under `backend=:gpu`, resource errors retry until `gpu_timeout`, and other
  errors raise at once with their original backtrace. Cleanup errors after a sticky CUDA
  fault cannot mask the processing error or block CPU fallback. User errors such as a
  readnoise map with the wrong size raise from the CPU fallback without retrying, and an
  interrupt (Ctrl-C) stops the call under either backend (#20).
- Even local-maximum windows, such as boxsize 7 with overlap 3, no longer warn that
  cuDNN does not support asymmetric padding. The GPU window now matches the CPU's
  asymmetric padding exactly.
- Documentation: the README and docs now state which hardware runs on the GPU, how
  SMLMBoxer sits beside Reactant, and the CUDA.jl 6.5.0 memory behavior. The README speed
  table is re-measured on the RTX 4090 because its GPU figures were reachable only on
  long stacks and its IdealCamera CPU figures were far too low; it now gives the fixed
  per-call GPU cost.

### Known issues
- With CUDA.jl 6.5.0, which current KernelAbstractions 0.10 environments resolve,
  `CUDA.reclaim()` no longer returns freed memory-pool memory to the driver, so after a
  GPU `getboxes` call the process keeps its peak GPU memory reserved until it exits
  (reused by later calls in the same process, unavailable to other processes). CUDA.jl
  6.4 and earlier release it.
- The standard GPU convolution rounds differently from cuDNN, so a box at a near-tied
  maximum can move by one pixel.

## [0.3.3] - 2026-10-27

### Fixed
- With `CUDA_VISIBLE_DEVICES` set, SMLMBoxer could read the wrong GPU's memory, processes
  and utilization, waiting for or falling back from a free GPU, or starting on a full one:
  it passed CUDA's device number to NVML, which numbers every GPU on the machine. NVML
  queries now find each CUDA device by its UUID, including MIG devices. A regression test,
  run on machines with a GPU and NVML, checks that every visible CUDA device maps to the
  same UUID in NVML; the GPU wait tests, run with `CUDA_VISIBLE_DEVICES=1` on a two-GPU
  host, fail before this fix and pass after it (#26).

## [0.3.2] - 2026-10-25

### Changed
- Tests use the lab's grouped layout: Core runs on GitHub CI; GPU and Long run on lab
  machines and are recorded as the `lab/tests` commit status. CI uses the lab's shared
  workflow with QA and a downgrade job. The docs site is published from release tags.
- Compat floors are CUDA 5.9, KernelAbstractions 0.9.38 and NNlib 0.9.25: the oldest
  versions the downgrade job tests. CUDA 6 remains supported.

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
