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

With CUDA.jl 6.5.0, `CUDA.reclaim()` no longer returns freed pool memory to the driver.
After GPU `getboxes`, the process keeps its peak GPU memory reserved until it exits.
Later calls in that process reuse it; other processes cannot use it. CUDA.jl 6.4 and
earlier release it.

GPU scheduling uses NVML polling to avoid OOM during device discovery in multi-process environments. See the [API docs](https://JuliaSMLM.github.io/SMLMBoxer.jl/dev/api/) for details.

## Detection Modes

| Mode | Camera | Filtering | Use Case |
|------|--------|-----------|----------|
| Standard DoG | `IdealCamera` | NNlib on CPU, KernelAbstractions kernel on GPU | Uniform noise, simulated data |
| Variance-weighted DoG | `SCMOSCamera` | Inverse-variance weighting via KernelAbstractions | Spatially-varying readnoise |

Variance weighting is automatic when an `SCMOSCamera` is provided — low-noise pixels get high weight, high-noise pixels get reduced influence.
The variance is `(readnoise / gain)^2` in ADU², evaluated per pixel when either
calibration is a map; SMLMData's gain is in e⁻/ADU.

## Parameter Interfaces

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
a spot with that many photons centered on a pixel just reaches the threshold. A spot
centered on a pixel corner needs about 1.1× more photons. The threshold uses the actual
filter kernels (uniform variance for sCMOS) and effective gain, so the camera paths
can give slightly different thresholds. Noise and spatially varying calibration affect
which real spots pass it.

`overlap` sets the local-maximum window `k = max(3, floor(Int, boxsize - overlap))`.
Detections are at least `cld(k, 2)` pixels apart along either axis; raising `overlap`
allows closer neighbors. Within a shared window, the dimmer maximum is removed;
ties keep the earlier detection.

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

Benchmarked on AMD Ryzen 9 5950X (32 threads) / NVIDIA RTX 4090, SMLMBoxer 0.3.4, CUDA.jl 6.5.0,
KernelAbstractions 0.10.4 (images/s, whole `getboxes` call):

The CPU columns use 32 threads; start Julia with `--threads=auto` for the 32-thread
speed. Plain `julia` runs one thread.

| Image Size | Frames | Ideal CPU | Ideal GPU | sCMOS CPU | sCMOS GPU |
|------------|-------:|----------:|----------:|----------:|----------:|
| 128×128 | 10 | 1,671 | 166 | 328 | 162 |
| 128×128 | 500 | 2,837 | 6,660 | 335 | 6,011 |
| 256×256 | 10 | 725 | 157 | 87 | 164 |
| 256×256 | 500 | 945 | 5,329 | 80 | 4,340 |
| 512×512 | 10 | 248 | 164 | 25 | 155 |
| 512×512 | 500 | 299 | 3,113 | 25 | 2,020 |

With one CPU thread, 10 frames and the CUDA.jl 6.5.0 environment (images/s):

| Image Size | Ideal CPU (1 thread) | sCMOS CPU (1 thread) |
|------------|---------------------:|--------------------:|
| 128×128 | 630 | 36 |
| 256×256 | 162 | 9 |
| 512×512 | 46 | 2 |

Every GPU call has a fixed cost of about 60 ms, mostly releasing the GPU memory pool
after the call, whatever the stack size, so GPU throughput depends on stack length.
At 256×256 with an IdealCamera, the CPU and GPU break even at about 10 frames on one
CPU thread and about 50 frames on 32 threads; below that use `backend = :cpu` if speed
matters. On 500 frames, compared with 32 CPU threads, the GPU is about 2-10x faster
with an IdealCamera and about 20-80x faster with an SCMOSCamera. 0.3.3 has the same
fixed cost.

Standard DoG uses NNlib on CPU and a KernelAbstractions kernel on GPU; variance-weighted sCMOS uses KernelAbstractions custom kernels on both. Run `GROUP=Long julia --project -e 'using Pkg; Pkg.test("SMLMBoxer")'` locally to benchmark your hardware.

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
