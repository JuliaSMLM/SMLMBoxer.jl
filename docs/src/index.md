```@meta
CurrentModule = SMLMBoxer
```

# SMLMBoxer.jl

SMLMBoxer.jl is a high-performance Julia package for detecting particles and blobs in single molecule localization microscopy (SMLM) image stacks. It uses Difference of Gaussians (DoG) filtering optimized for blob detection with GPU acceleration support.

## Features

- **PSF-Aware Detection**: Automatic parameter tuning based on PSF characteristics and photon thresholds
- **GPU Acceleration**: NVIDIA GPUs through CUDA.jl and KernelAbstractions kernels (NNlib on CPU) for fast processing of large image stacks
- **Variance-Weighted Filtering**: Optimal detection in sCMOS data with spatially-varying noise
- **ROIBatch Integration**: Seamless integration with SMLMData.jl coordinate system
- **Flexible Interface**: Both user-friendly PSF-aware and expert-level direct control modes
- **KernelAbstractions Kernels**: GPU filtering and pooling, and sCMOS filtering on both CPU and GPU

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

## Quick Start

### Basic Detection with IdealCamera

```julia
using SMLMBoxer, SMLMData

# Create camera (256×256 pixels, 100nm pixel size)
camera = IdealCamera(1:256, 1:256, 0.1f0)

# Detect particles using PSF-aware interface
(roi_batch, info) = getboxes(imagestack, camera;
    psf_sigma = 0.13,      # PSF sigma in microns (130 nm)
    min_photons = 500.0,   # Minimum photon count threshold
    boxsize = 11)          # ROI size in pixels

# Access results
boxes = roi_batch.data              # (11 × 11 × n_rois) array
x_corners = roi_batch.x_corners     # x (column) positions
y_corners = roi_batch.y_corners     # y (row) positions
frames = roi_batch.frame_indices    # frame numbers

# Iterate over detected ROIs
for roi in roi_batch
    # Each roi is a SingleROI with .data, .corner, .frame_idx
    process(roi.data)
end
```

### sCMOS Detection with Variance Weighting

```julia
using SMLMBoxer, SMLMData

# Create sCMOS camera with per-pixel readnoise map
camera = SCMOSCamera(
    256, 256,           # dimensions
    0.1f0,              # pixel size (μm) - Float32 to match readnoise_map
    readnoise_map,      # per-pixel readnoise (e⁻)
    offset = 100.0f0,
    gain = 2.0f0
)

# Automatically uses variance-weighted filtering
(roi_batch, info) = getboxes(imagestack, camera;
    psf_sigma = 0.13,
    min_photons = 500.0,
    backend = :auto)    # GPU with CPU fallback
```

## How It Works

The `getboxes()` function implements a multi-stage blob detection pipeline:

1. **Difference of Gaussians (DoG) Filtering**: Convolves the image with the difference of two Gaussian kernels to enhance blob-like features while suppressing background
   - Small Gaussian (σ_small) matches the PSF to maximize SNR
   - Large Gaussian (σ_large) provides local background estimation
   - DoG = Small - Large highlights blob features
   - Pixels near the image edge are filtered with the in-bounds part of each Gaussian, on both filtering paths

2. **Variance Weighting (sCMOS)**: Each pixel is weighted by inverse variance in ADU²,
   with `variance = (readnoise / gain)^2` (gain in e⁻/ADU), including per-pixel maps:
   - Low-noise pixels receive high weight (strong influence)
   - High-noise pixels receive low weight (reduced false positives)
   - Implements optimal detection theory for spatially-varying noise

3. **Local Maxima Detection**: Identifies peaks in the filtered image above the threshold

4. **Overlap Removal**: Removes the dimmer maximum only within a shared max-pool
   window; ties keep the earlier detection. With
   `k = max(3, floor(Int, boxsize - overlap))`, detections are at least `cld(k, 2)`
   pixels apart along either axis.

5. **ROI Extraction**: Supports odd and even sizes, with corners clamped to the image.
   One box per corner and frame is kept, from the brightest filtered detection.

The image's (rows, columns) must match the camera's pixel edges, otherwise `getboxes`
raises `DimensionMismatch`. Crop the camera together with the image.

## PSF-Aware Interface

The recommended interface automatically configures filter parameters from physical units:

```julia
(roi_batch, info) = getboxes(imagestack, camera;
    psf_sigma = 0.13,      # PSF width in microns (physical units)
    min_photons = 500.0)   # Detection threshold in photons
```

This automatically calculates:
- σ_small = 1.0 × psf_sigma (in pixels) - matches PSF for optimal detection
- σ_large = 2.0 × psf_sigma (in pixels) - background suppression
- minval threshold from photons accounting for PSF spreading and DoG response

The threshold is the actual filter's peak response to a noise-free, pixel-integrated
Gaussian with `min_photons` photons centered on a pixel, times effective gain.
A spot centered on a pixel corner needs about 1.1× more photons. The sCMOS reference
uses uniform variance and its own Gaussian windows; its threshold can differ slightly
from the standard path. Noise and varying calibration affect real detections.

## Advanced Interface

Expert users can directly control filter parameters:

```julia
(roi_batch, info) = getboxes(imagestack;
    sigma_small = 1.5,  # Small Gaussian sigma in pixels
    sigma_large = 3.0,  # Large Gaussian sigma in pixels
    minval = 10.0)      # DoG intensity threshold
```

## Documentation Structure

- [Examples](@ref) - Detailed usage examples with SMLMSim integration
- [API Reference](@ref) - Complete API documentation for all exported functions
