```@meta
CurrentModule = SMLMBoxer
```

# SMLMBoxer.jl

SMLMBoxer.jl is a high-performance Julia package for detecting particles and blobs in single molecule localization microscopy (SMLM) image stacks. It uses Difference of Gaussians (DoG) filtering optimized for blob detection with GPU acceleration support.

## Features

- **PSF-Aware Detection**: Automatic parameter tuning based on PSF characteristics and photon thresholds
- **GPU Acceleration**: CUDA support via NNlib/cuDNN for fast processing of large image stacks
- **Variance-Weighted Filtering**: Optimal detection in sCMOS data with spatially-varying noise
- **ROIBatch Integration**: Seamless integration with SMLMData.jl coordinate system
- **Flexible Interface**: Both user-friendly PSF-aware and expert-level direct control modes
- **Device-Agnostic Kernels**: KernelAbstractions.jl for automatic CPU/GPU kernel execution

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

## Quick Start

### Basic Detection with IdealCamera

```julia
using SMLMBoxer, SMLMData

# Create camera (256×256 pixels, 100nm pixel size)
camera = IdealCamera(1:256, 1:256, 0.1f0)

# Detect particles using PSF-aware interface
roi_batch = getboxes(imagestack, camera;
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
roi_batch = getboxes(imagestack, camera;
    psf_sigma = 0.13,
    min_photons = 500.0,
    use_gpu = true)     # GPU acceleration for variance weighting
```

## How It Works

The `getboxes()` function implements a multi-stage blob detection pipeline:

1. **Difference of Gaussians (DoG) Filtering**: Convolves the image with the difference of two Gaussian kernels to enhance blob-like features while suppressing background
   - Small Gaussian (σ_small) matches the PSF to maximize SNR
   - Large Gaussian (σ_large) provides local background estimation
   - DoG = Small - Large highlights blob features

2. **Variance Weighting (sCMOS)**: When an SCMOSCamera is provided, each pixel's contribution is weighted by inverse variance (1/readnoise²):
   - Low-noise pixels receive high weight (strong influence)
   - High-noise pixels receive low weight (reduced false positives)
   - Implements optimal detection theory for spatially-varying noise

3. **Local Maxima Detection**: Identifies peaks in the filtered image above the threshold

4. **Overlap Removal**: Filters detections to ensure minimum separation between spots

5. **ROI Extraction**: Cuts out square regions around each detection with coordinate tracking

## PSF-Aware Interface

The recommended interface automatically configures filter parameters from physical units:

```julia
roi_batch = getboxes(imagestack, camera;
    psf_sigma = 0.13,      # PSF width in microns (physical units)
    min_photons = 500.0)   # Detection threshold in photons
```

This automatically calculates:
- σ_small = 1.0 × psf_sigma (in pixels) - matches PSF for optimal detection
- σ_large = 2.0 × psf_sigma (in pixels) - background suppression
- minval threshold from photons accounting for PSF spreading and DoG response

## Advanced Interface

Expert users can directly control filter parameters:

```julia
roi_batch = getboxes(imagestack;
    sigma_small = 1.5,  # Small Gaussian sigma in pixels
    sigma_large = 3.0,  # Large Gaussian sigma in pixels
    minval = 10.0)      # DoG intensity threshold
```

## Documentation Structure

- [Examples](@ref) - Detailed usage examples with SMLMSim integration
- [API Reference](@ref) - Complete API documentation for all exported functions
