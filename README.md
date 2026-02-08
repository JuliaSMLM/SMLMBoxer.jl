# SMLMBoxer.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaSMLM.github.io/SMLMBoxer.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSMLM.github.io/SMLMBoxer.jl/dev/)
[![Build Status](https://github.com/JuliaSMLM/SMLMBoxer.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaSMLM/SMLMBoxer.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaSMLM/SMLMBoxer.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSMLM/SMLMBoxer.jl)

Particle detection and ROI extraction for single-molecule localization microscopy. Difference-of-Gaussians filtering with automatic GPU acceleration and sCMOS variance weighting.

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

GPU scheduling uses NVML polling to avoid OOM during device discovery in multi-process environments. See the [API docs](https://JuliaSMLM.github.io/SMLMBoxer.jl/dev/api/) for details.

## Detection Modes

| Mode | Camera | Filtering | Use Case |
|------|--------|-----------|----------|
| Standard DoG | `IdealCamera` | NNlib convolution (cuDNN on GPU) | Uniform noise, simulated data |
| Variance-weighted DoG | `SCMOSCamera` | Inverse-variance weighting via KernelAbstractions | Spatially-varying readnoise |

Variance weighting is automatic when an `SCMOSCamera` is provided — low-noise pixels get high weight, high-noise pixels get reduced influence.

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
    # roi.data, roi.corner, roi.frame_idx, roi.camera
end
first_roi = roi_batch[1]
```

## Performance

Benchmarked on AMD Ryzen Threadripper PRO 5975WX / NVIDIA RTX A6000, 10 frames (images/s):

| Image Size | Ideal CPU | Ideal GPU | sCMOS CPU | sCMOS GPU |
|------------|----------:|----------:|----------:|----------:|
| 128×128 | 60 | 4,564 | 23 | 4,090 |
| 256×256 | 15 | 3,752 | 6 | 2,557 |
| 512×512 | 4 | 1,553 | 1 | 927 |

Standard DoG uses NNlib/cuDNN; variance-weighted sCMOS uses KernelAbstractions custom kernels. Run `Pkg.test("SMLMBoxer")` locally to benchmark your hardware.

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
