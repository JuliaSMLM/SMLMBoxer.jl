# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and versions follow
Julia's rule for 0.Y.Z: raising Z breaks nothing, raising Y is an interface break. Releases before
0.3.1 are described in the [GitHub release notes](https://github.com/JuliaSMLM/SMLMBoxer.jl/releases).

## [Unreleased]

## [0.4.0]

### Breaking
- `min_photons` now uses the actual DoG filter response to a pixel-integrated Gaussian,
  replacing the empirical 0.65 factor. A noise-free spot centered on a pixel just reaches
  the threshold at that photon count; a pixel-corner spot needs about 1.1× more. The
  standard and sCMOS paths use their own kernels. Among 62,682 clean in-focus lab spots,
  half are detected near 487 fitted photons at `min_photons = 500`, versus 566 before.
  The same setting finds more boxes: on 2000 lab frames at `min_photons = 500`, 6.5%
  more with an IdealCamera and 4.1% more with an SCMOSCamera (10.3% and 6.7% at 1000);
  every box 0.3.4 found on the IdealCamera path is still found.
- Images must match their camera's (rows, columns), or raise `DimensionMismatch`.
  A 192 × 192 crop passed with its full 256 × 256 camera previously returned 24,347
  boxes with coordinates silently wrong by 64 pixels.
- sCMOS variance weighting now uses `(readnoise / gain)^2` in ADU², including maps.
  On real frames with gain 0.40–0.50 e⁻/ADU, 2.3% of boxes change: added boxes hold
  507 clean in-focus spots of at least 500 photons, versus 425 in the removed boxes.

### Fixed
- Increasing `overlap` now permits closer neighbors according to the max-pool window.
  Only maxima sharing that window compete; the dimmer goes, with ties keeping the
  earlier detection. On 2000 lab frames this adds 66 boxes at boxsize 9 / overlap 3,
  1,396 at boxsize 9 / overlap 4, and 201 at boxsize 11 / overlap 4, removing none;
  default-overlap results are unchanged.
- Even box sizes use the same clamped-corner rule as odd sizes; real-frame box sizes
  8 and 10 previously raised `DimensionMismatch`. Odd-size crops remain unchanged.
- After clamping, keep only the brightest filtered detection per (corner, frame).
  This removes the 3–13 duplicate boxes per 2000 real frames seen in 0.3.4.
- The chunked `recommend_batch_size` example offsets frame indices by the chunk start.
  The lab caller already tracks this offset; no new keyword is needed.

### Changed
- The release after each successful GPU call no longer runs a full garbage collection.
  Owned GPU arrays are explicitly freed within each batch, also on errors. CUDA.jl 6
  synchronizes, purges caches and trims the pool; CUDA.jl 5 retains its GC-free reclaim.
  A warmed-up 256×256×10 GPU call on the RTX 4090 falls from about 57 ms in 0.3.4 to
  0.75 ms (medians of 0.73–0.84 ms across CUDA.jl 6.4.2 and 6.5.0). IdealCamera GPU
  throughput on 500 frames at 256×256 rises from 5,329 to 13,805 images/s. The GPU wins
  from one frame at all tested sizes (128, 256 and 512 px), with either one or 32 CPU
  threads on the AMD Ryzen 9 5950X. Release after a short call takes 0.013 ms under
  CUDA.jl 6.5.0, which still retains pool memory (known issue below). Under 6.4.2,
  release returns memory to the driver: about 43 ms per 2 GiB, compared with 200 ms for
  `CUDA.reclaim()` on the same allocation workload.
- Removed unused internal `find_best_gpu`, `select_backend`, `wait_for_gpu_memory`,
  and `estimate_gpu_memory` helpers and the discarded per-box camera ROI construction.
  The per-frame memory estimator is unchanged.

### Known issues
- With CUDA.jl 6.5.0, freed pool memory is still not returned to the driver (see 0.3.4):
  after a 256-frame call a process kept 128-160 MiB more than before it. CUDA.jl 6.4.2
  returns all of it.

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
  Boxes change only within a few pixels of the edges: on 2000 frames of lab sCMOS data
  converted to photons, the share of boxes clamped at an edge falls from 22% to 11% at
  the default `min_photons` (the sCMOS path gives 11% on the same frames), and on
  simulated frames with a 10 photon/px background from 72% to 15% at `min_photons = 50`.
- Without a camera, the ROIBatch's default IdealCamera was one pixel larger than the
  image on each axis. Box positions were and are unchanged.
- Overlap removal now applies to every frame of a stack; previously it applied only to
  frame 1 (#19). Overlap removal no longer allocates per pair of boxes (a 600-box frame
  takes 0.25 ms instead of 19 ms), so the fix costs nothing measurable at the default
  `min_photons`; with 500-700 boxes per frame it adds 0.2-0.35 ms per frame.
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
- The standard GPU convolution rounds differently from cuDNN: on 2000 lab frames at
  `min_photons = 50`, 2 of 1.13 million boxes away from the edges moved by one pixel
  (near-tied maxima); none at the default `min_photons`.

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
