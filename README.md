# SMLMBoxer

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaSMLM.github.io/SMLMBoxer.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSMLM.github.io/SMLMBoxer.jl/dev/)
[![Build Status](https://github.com/JuliaSMLM/SMLMBoxer.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaSMLM/SMLMBoxer.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaSMLM/SMLMBoxer.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSMLM/SMLMBoxer.jl)

*SMLMBoxer.jl* is a Julia package that provides a fast and efficient method for detecting particles or blobs in a multidimensional image stack and cutting out sub-regions around local maxima. The package exports a single high-level interface function `getboxes()`.

## Installation

```julia
using Pkg
Pkg.add("SMLMBoxer")
```

For development version:

```julia
using Pkg
Pkg.add(url="https://github.com/JuliaSMLM/SMLMBoxer.jl")
```

## Usage
The main function provided by the package is `getboxes()`, which detects particles or blobs in a multidimensional image stack and returns an `ROIBatch` containing detected regions centered around local maxima. The function uses a Difference of Gaussians (DoG) filter optimized for blob detection and is capable of GPU acceleration.

### Two Calling Conventions

**Config-based (recommended for reusable settings):**
```julia
using SMLMBoxer, SMLMData

camera = IdealCamera(1:256, 1:256, 0.1f0)
config = BoxerConfig(psf_sigma=0.13, min_photons=500.0, boxsize=11)

(roi_batch, info) = getboxes(imagestack, camera, config)
```

**Kwargs-based (convenient for one-off calls):**
```julia
(roi_batch, info) = getboxes(imagestack, camera;
    psf_sigma = 0.13,
    min_photons = 500.0,
    boxsize = 11)
```

Both conventions are equivalent - kwargs are forwarded to a BoxerConfig internally.

### BoxerConfig

Configuration struct for ROI detection. Create with `@kwdef` defaults:

```julia
config = BoxerConfig(
    # PSF-aware interface (recommended)
    psf_sigma = 0.13,       # PSF sigma in microns (requires camera)
    min_photons = 500.0,    # Minimum photons for detection

    # Box parameters
    boxsize = 11,           # ROI size in pixels
    overlap = 2.0,          # Max overlap between detections

    # Backend
    backend = :auto         # :cpu, :gpu, or :auto
)
```

### Example (PSF-Aware Detection)
```julia
using SMLMBoxer, SMLMData

# Setup camera
camera = IdealCamera(1:256, 1:256, 0.1f0)  # 256×256 pixels, 100nm pixel size

# Detect with PSF-aware parameters (physical units)
(roi_batch, info) = getboxes(imagestack, camera;
    psf_sigma = 0.13,              # PSF sigma in microns (physical units)
    min_photons = 500.0,           # Minimum photon count to detect
    boxsize = 11)                  # ROI size in pixels

# info contains: backend, elapsed_s, device_id, n_rois, batch_size, n_batches, memory_per_batch
println("Processed in ", info.elapsed_s * 1000, " ms on ", info.backend)
```

### Primary Parameters (PSF-Aware Interface)
- `psf_sigma::Real`: PSF sigma in **microns** (physical units, e.g., 0.13 for 130nm PSF).
  Automatically converted to pixels using camera and sets optimal DoG filter scales
  (sigma_small = 1.0×psf_sigma, sigma_large = 2.0×psf_sigma). **Requires camera to be provided.**
- `min_photons::Real`: Minimum total photons for an emitter to be detected (default: 500.0).
  Automatically converted to appropriate intensity threshold accounting for PSF spreading and filter response.

### Advanced Parameters (Direct Control)
For expert users who want direct control over the DoG filter:
- `sigma_small::Real`: Small Gaussian sigma in pixels (default: 1.0).
- `sigma_large::Real`: Large Gaussian sigma in pixels (default: 2.0).
- `minval::Real`: DoG filter intensity threshold (default: 0.0).

**Note:** If `psf_sigma` is provided, it overrides `sigma_small`, `sigma_large`, and `minval`.

### Other Parameters
- `imagestack::AbstractArray{<:Real}`: The input image stack (2D or 3D).
- `camera`: Optional camera object (IdealCamera or SCMOSCamera from SMLMData). Enables proper coordinate tracking and variance-weighted filtering for sCMOS.
- `boxsize::Int`: Size of ROI boxes in pixels (default: 7).
- `overlap::Real`: Maximum overlap between detections in pixels (default: 2.0).
- `backend::Symbol`: Compute backend - `:cpu`, `:gpu`, or `:auto` (default: `:auto`).
  - `:cpu` - Always use CPU
  - `:gpu` - Require GPU, wait for memory if needed
  - `:auto` - Try GPU with timeout, fall back to CPU if unavailable
- `auto_timeout::Real`: Max seconds to wait for GPU in `:auto` mode (default: 30.0).
- `gpu_timeout::Real`: Max seconds to wait in `:gpu` mode (default: Inf).
- `on_wait::Function`: Optional callback for wait progress reporting.

### Returns
Tuple of `(ROIBatch, BoxesInfo)`:

**ROIBatch** with the following fields:
- `data`: ROI stack (boxsize × boxsize × n_rois) containing detected image patches.
- `x_corners`: Vector of x (column) corner positions in camera coordinates.
- `y_corners`: Vector of y (row) corner positions in camera coordinates.
- `frame_indices`: Vector of frame indices for each ROI.
- `camera`: Camera object for coordinate system tracking.
- `roi_size`: Size of each ROI.

**BoxesInfo** with processing metadata:
- `backend`: Compute backend used (`:gpu` or `:cpu`)
- `elapsed_s`: Wall time in seconds
- `device_id`: GPU device ID (0-based), or -1 for CPU
- `n_rois`: Number of ROIs detected
- `batch_size`: Frames per batch during processing
- `n_batches`: Number of batches processed
- `memory_per_batch`: Estimated memory per batch in bytes

### How It Works
The `getboxes()` function applies a Difference of Gaussians (DoG) filter to identify blob-like features. When using the PSF-aware interface, the filter scales are automatically matched to your PSF width for optimal detection sensitivity, and the photon threshold is converted to the appropriate intensity threshold accounting for PSF spreading and filter response.

### GPU Scheduling

SMLMBoxer uses a unified GPU retry loop that handles multi-process contention on shared GPU servers:

1. **NVML polling** scans all GPUs for one with sufficient free memory and low contention (no CUDA context created)
2. **GPU processing** is attempted: context creation, DoG filtering, local max detection
3. **On any failure** (no memory, context race, runtime OOM): GPU memory is released via `GC.gc()` + `CUDA.reclaim()`, then re-polls NVML with remaining timeout
4. **On timeout**: `:auto` falls back to CPU, `:gpu` errors

Memory is always reclaimed after GPU processing (both success and failure) so finished jobs don't block other processes waiting for GPU resources.

```julia
# Wait up to 30s for GPU, fall back to CPU
(roi_batch, info) = getboxes(imagestack, camera;
    psf_sigma=0.13, backend=:auto, auto_timeout=30.0)

# Monitor wait progress
(roi_batch, info) = getboxes(imagestack, camera;
    psf_sigma=0.13, backend=:auto,
    on_wait=(elapsed, avail, req) -> @info "Waiting..." elapsed avail req)
```

