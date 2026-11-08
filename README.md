# SMLMBoxer.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaSMLM.github.io/SMLMBoxer.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSMLM.github.io/SMLMBoxer.jl/dev/)
[![Build Status](https://github.com/JuliaSMLM/SMLMBoxer.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaSMLM/SMLMBoxer.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaSMLM/SMLMBoxer.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSMLM/SMLMBoxer.jl)

Particle detection and ROI extraction for single-molecule localization microscopy. Difference-of-Gaussians filtering with automatic acceleration on NVIDIA GPUs and sCMOS variance weighting.

## Installation

```julia
using Pkg
Pkg.add("SMLMBoxer")
```

## Quick Start

```julia
using SMLMBoxer

# Input: image stack + camera from your acquisition pipeline
# camera = IdealCamera(1:256, 1:256, 0.1f0)  # from SMLMData.jl

# Detect: returns (ROIBatch, BoxesInfo)
roi_batch, info = getboxes(imagestack, camera, BoxerConfig(
    psf_sigma    = 0.13,    # PSF σ in μm — sets DoG filter scales automatically
    min_photons  = 500.0,   # photon threshold — converted to intensity threshold
    boxsize      = 11,      # ROI size in pixels
    backend      = :auto,   # :cpu, :gpu, or :auto (GPU w/ CPU fallback)
))

# Output: ROI patches with camera coordinates
roi_batch.data           # 11×11×n_rois array of image patches
roi_batch.x_corners      # x (column) corner positions
roi_batch.y_corners      # y (row) corner positions
roi_batch.frame_indices  # source frame for each ROI
```

For complete SMLM workflows (detection + fitting + rendering), see [SMLMAnalysis.jl](https://github.com/JuliaSMLM/SMLMAnalysis.jl).

The image's (rows, columns) must match the camera's pixel edges; a mismatch raises
`DimensionMismatch`. Use a camera cropped to the image's field of view.
Every `boxsize >= 1`, odd or even, is supported when it fits the image. Corners use
`clamp(center - (boxsize - 1) ÷ 2, 1, image_size - boxsize + 1)` on each axis.
If edge clamping gives multiple detections the same corner in a frame, the brightest
filtered detection supplies the single retained box.

## GPU Support

| Hardware | `backend = :auto` | `backend = :gpu` |
|---|---|---|
| NVIDIA GPU with a working CUDA.jl | GPU, CPU if it cannot run | GPU |
| AMD or Intel GPU, or no GPU | CPU | error: `GPU backend requested but CUDA is not functional` |

`Pkg.add("SMLMBoxer")` is the whole install for CPU and GPU. CUDA.jl comes with it.

AMD GPUs are not supported yet. The GPU path uses CUDA.jl and needs an NVIDIA GPU;
on a machine with an AMD GPU, `:auto` runs on the CPU. AMDGPU.jl can be installed and
loaded beside SMLMBoxer (resolving CUDA 6.5.0 and KernelAbstractions 0.10.4), but
SMLMBoxer does not use it. The kernels use KernelAbstractions, which an AMD backend
would build on; no AMD backend exists in this release.

SMLMBoxer does not use Reactant and needs no setup for it. Removing the cuDNN dependency
removed the library clash with Reactant's bundled copy, so both packages load in one
process. Tested with Reactant 0.2.304 and KernelAbstractions 0.9.45, with CUDA 6.2.2
(what a fresh install resolves) on Julia 1.13.0 and with CUDA 5.11.3 on Julia 1.12.6
and 1.13.0; `getboxes` with `backend = :gpu` returned the same boxes as the CPU.
Such an environment resolves KernelAbstractions 0.9 and the CUDA version Reactant
allows; SMLMBoxer accepts both. When Reactant first used the RTX 4090 it reserved
17.6 GiB of the card's 24 GB for its own allocator, so `getboxes` in that process has
less GPU memory available.

```julia
using Pkg
Pkg.add(["SMLMBoxer", "Reactant"])
using SMLMBoxer, Reactant
```

GPU scheduling uses NVML polling to avoid OOM during device discovery in multi-process environments. See the [API docs](https://JuliaSMLM.github.io/SMLMBoxer.jl/dev/api/) for details.

## Detection Modes

| Mode | Camera | Filtering | Use Case |
|------|--------|-----------|----------|
| Standard DoG | `IdealCamera` | NNlib on CPU, KernelAbstractions kernel on GPU | Uniform noise, simulated data |
| Variance-weighted DoG | `SCMOSCamera` | Inverse-variance weighting via KernelAbstractions | Spatially-varying readnoise |

Variance weighting is automatic when an `SCMOSCamera` is provided — low-noise pixels get high weight, high-noise pixels get reduced influence.
PSF-aware detection converts ADU to photons per pixel and uses `(readnoise / qe)^2`
in photons². Direct detection retains `(readnoise / gain)^2` in ADU².
SMLMData's gain is in e⁻/ADU; calibration can be scalar or mapped.

## Parameter Interfaces

Inputs are checked before any filtering, and each failure raises `ArgumentError` naming the
field or pixels at fault:

- Images are 2D or 3D `(rows, cols, frames)`, at least `boxsize` in each image dimension, and
  free of NaN and Inf (integer images skip that scan). A stack of zero frames returns an empty
  `ROIBatch`.
- An `SCMOSCamera` needs finite, positive `readnoise`, `gain` and `qe`, scalar or per pixel.
  For a noise-free readout, use an `IdealCamera` with the image in photons.
- `min_photons` needs `psf_sigma` (with `psf_sigma` alone, 500 is used), and `psf_sigma` needs a
  camera with square pixels of one size. Other cameras use `sigma_small`/`sigma_large` in pixels.
- Parameters: `0 < sigma_small < sigma_large`, positive `psf_sigma` and `min_photons`, finite
  `minval`, `boxsize >= 1`, `overlap >= 0`, `backend` in `:cpu`, `:gpu`, `:auto`, timeouts
  `>= 0`.

With an `SCMOSCamera`, the PSF-aware path detects in photons per pixel,
`(adu - offset) * gain / qe`, so `min_photons` means the same number of photons at every pixel
of a gain map. Boxes are still cut from the ADU image. The direct interface filters the image as
given, and `minval` stays in its units.

**PSF-aware (recommended)** — physical units, automatic filter configuration:

```julia
roi_batch, info = getboxes(imagestack, camera, BoxerConfig(
    psf_sigma    = 0.13,    # PSF σ in μm → sigma_small = 1.0×, sigma_large = 2.0× in pixels
    min_photons  = 500.0,   # total photons → DoG intensity threshold
))
```

**Direct control** — expert users set filter parameters directly:

```julia
roi_batch, info = getboxes(imagestack, camera, BoxerConfig(
    sigma_small  = 1.3,     # small Gaussian σ in pixels
    sigma_large  = 2.6,     # large Gaussian σ in pixels
    minval       = 10.0,    # DoG intensity threshold
))
```

A kwargs interface is also available for one-off calls: `getboxes(imagestack, camera; psf_sigma=0.13, min_photons=500.0)`.

`min_photons` sets the filtered peak of a noise-free, pixel-integrated Gaussian spot:
a spot with that many photons centered on a pixel just reaches the threshold. See
`BoxerConfig` for subpixel-position effects. The threshold uses the actual
filter kernels (uniform variance for sCMOS) with effective gain 1, so the camera paths
can give slightly different thresholds. Noise and spatially varying calibration affect
which real spots pass it.

`overlap` sets the local-maximum window `k = max(3, floor(Int, boxsize - overlap))`.
Detections are at least `cld(k, 2)` pixels apart along either axis; raising `overlap`
allows closer neighbors. Detections sharing a window have equal intensities;
the earlier detection is kept.

## Output Format

`getboxes()` returns `(ROIBatch, BoxesInfo)`.

**ROIBatch** fields:

| Field | Description |
|-------|-------------|
| `data` | ROI stack (boxsize × boxsize × n_rois) |
| `x_corners` | x (column) corner positions |
| `y_corners` | y (row) corner positions |
| `frame_indices` | source frame for each ROI |
| `camera` | camera object for coordinate tracking |
| `roi_size` | size of each ROI (square) |

**BoxesInfo** fields:

| Field | Description |
|-------|-------------|
| `elapsed_s` | Wall time (seconds) |
| `backend` | `:cpu` or `:gpu` (never `:auto`) |
| `device_id` | GPU index (0-based) or -1 for CPU |
| `n_rois` | Number of ROIs detected |
| `batch_size` | Frames per batch |
| `n_batches` | Number of batches processed |
| `memory_per_batch` | Estimated bytes per batch |

ROIBatch supports iteration and indexing:

```julia
for roi in roi_batch
    # roi.data, roi.corner, roi.frame_idx
end
first_roi = roi_batch[1]
```

## Performance

Benchmarked on AMD Ryzen 9 5950X (32 threads) / NVIDIA RTX 4090, SMLMBoxer 0.4.0,
CUDA.jl 6.5.0, KernelAbstractions 0.10.4 (images/s, whole `getboxes` call):

The CPU columns use 32 threads; start Julia with `--threads=auto` for the 32-thread
speed. Plain `julia` runs one thread.

| Image Size | Frames | Ideal CPU | Ideal GPU | sCMOS CPU | sCMOS GPU |
|------------|-------:|----------:|----------:|----------:|----------:|
| 128×128 | 1 | 270 | 2,011 | 317 | 2,583 |
| 128×128 | 10 | 1,155 | 19,858 | 325 | 16,571 |
| 128×128 | 500 | 2,754 | 49,095 | 284 | 36,262 |
| 256×256 | 1 | 256 | 1,558 | 49 | 2,357 |
| 256×256 | 10 | 963 | 10,417 | 81 | 10,233 |
| 256×256 | 500 | 876 | 13,805 | 76 | 10,643 |
| 512×512 | 1 | 188 | 1,954 | 23 | 961 |
| 512×512 | 10 | 260 | 4,420 | 21 | 2,492 |
| 512×512 | 500 | 262 | 4,317 | 19 | 2,973 |

With one CPU thread, 10 frames and the CUDA.jl 6.5.0 environment (images/s):

| Image Size | Ideal CPU (1 thread) | sCMOS CPU (1 thread) |
|------------|---------------------:|--------------------:|
| 128×128 | 543 | 34 |
| 256×256 | 150 | 8 |
| 512×512 | 40 | 2 |

After warmup, a 256×256×10 GPU call takes about 0.75 ms on the RTX 4090, down from
about 57 ms in 0.3.4 (medians of 0.73–0.84 ms across CUDA.jl 6.4.2 and 6.5.0).
The GPU wins from one frame at every size tested (128, 256 and 512 px), with either
camera path and either one or 32 CPU threads on this machine.

Under CUDA.jl 6.2 and later, 0.3.4 spent about 60 ms per GPU call in a full garbage
collection inside `CUDA.reclaim()`. 0.4.0 frees its GPU arrays as soon as each batch is done and releases
memory without a full garbage collection. On 500 frames at 256×256, IdealCamera GPU
throughput increases from 5,329 to 13,805 images/s.

Release after a short call takes about 0.02 ms on the RTX 4090; returning
2 GiB to the driver takes about 42 ms.

Standard DoG uses NNlib on CPU and a KernelAbstractions kernel on GPU;
variance-weighted sCMOS uses KernelAbstractions custom kernels on both. Run
`GROUP=Long julia --project -e 'using Pkg; Pkg.test("SMLMBoxer")'` locally to benchmark
your hardware.

## Algorithm Reference

sCMOS pixel-dependent noise model:

> Huang, F., Hartwich, T.M.P., Rivera-Molina, F.E. et al. "Video-rate nanoscopy using sCMOS camera-specific single-molecule localization algorithms." *Nat Methods* **10**, 653-658 (2013). [DOI: 10.1038/nmeth.2488](https://doi.org/10.1038/nmeth.2488)

## Related Packages

- **[SMLMAnalysis.jl](https://github.com/JuliaSMLM/SMLMAnalysis.jl)** - Complete SMLM workflow (detection + fitting + rendering)
- **[GaussMLE.jl](https://github.com/JuliaSMLM/GaussMLE.jl)** - MLE fitting of detected ROIs
- **[SMLMData.jl](https://github.com/JuliaSMLM/SMLMData.jl)** - Core data types for SMLM
- **[SMLMSim.jl](https://github.com/JuliaSMLM/SMLMSim.jl)** - SMLM data simulation
- **[MicroscopePSFs.jl](https://github.com/JuliaSMLM/MicroscopePSFs.jl)** - PSF models

## License

MIT License - see [LICENSE](LICENSE) file for details.
