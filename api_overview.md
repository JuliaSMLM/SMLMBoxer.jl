# SMLMBoxer.jl API Reference

Particle/blob detection in SMLM image stacks using difference-of-Gaussians filtering with GPU acceleration and sCMOS variance-weighted filtering support.

## Exports

**Total exports:** 6
- `getboxes` - Main detection function
- `BoxerConfig` - Configuration struct for detection parameters
- `BoxesInfo` - Metadata struct returned alongside ROIBatch
- `recommend_batch_size` - Memory-aware batch sizing utility
- `ROIBatch` - Re-exported from SMLMData.jl
- `SingleROI` - Re-exported from SMLMData.jl

Note: `SMLMBoxer.api()` is available but not exported to avoid conflicts with other JuliaSMLM packages.

## Key Concepts

### Difference of Gaussians (DoG) Filtering
Detects blob-like features by subtracting two Gaussian-blurred versions of the image:
- **sigma_small**: Matches PSF size for optimal blob detection
- **sigma_large**: Suppresses background (typically 2× sigma_small)
- **minval**: Intensity threshold after filtering

### PSF-Aware Detection (Recommended)
Specify physical PSF parameters (in microns), automatically converted to optimal filter settings:
- **psf_sigma**: PSF sigma in microns (e.g., 0.13 for 130nm PSF)
- **min_photons**: Total photon threshold (automatically converted to DoG intensity threshold)

### Variance-Weighted Filtering (sCMOS)
When `SCMOSCamera` is provided, implements SMITE-style inverse variance weighting:
- Each pixel weighted by `gaussian_kernel / variance` where variance = readnoise²
- Low-noise pixels: high weight (strong detection influence)
- High-noise pixels: low weight (reduced false positives)
- GPU-accelerated via KernelAbstractions.jl (device-agnostic kernels)

### GPU Acceleration and Scheduling
- Standard DoG: NNlib with cuDNN backend (10-100x speedup)
- Variance-weighted: KernelAbstractions custom kernels (same code for CPU/GPU)
- Multi-GPU support: NVML-based polling selects GPU with most free memory across all devices

**Unified GPU retry loop** handles multi-process contention:
1. Poll all GPUs via NVML (no CUDA context creation) for sufficient free memory and low utilization
2. Acquire GPU context and run processing
3. On any failure (no memory, TOCTOU context race, runtime OOM): release memory via `GC.gc() + CUDA.reclaim()`, re-poll NVML with remaining timeout
4. On timeout: `:auto` falls back to CPU, `:gpu` errors
5. On success: reclaim GPU memory pool so finished jobs don't block other processes

**Backend modes:**
- `:cpu` - Always CPU, no GPU involvement
- `:gpu` - Require GPU, retry until `gpu_timeout` (default: Inf), error if unavailable
- `:auto` - Try GPU, retry until `auto_timeout` (default: 300s), fall back to CPU

**Wait progress callback:**
```julia
config = BoxerConfig(
    psf_sigma=0.13,
    backend=:auto,
    on_wait=(elapsed, available, required) -> @info "Waiting for GPU" elapsed available required
)
```

## Configuration

### `BoxerConfig`

Configuration struct for ROI detection parameters. Supports `@kwdef` construction with defaults.

```julia
@kwdef struct BoxerConfig
    # PSF-aware interface (recommended)
    psf_sigma::Union{Float64,Nothing} = nothing  # PSF sigma in microns
    min_photons::Float64 = 500.0                 # Minimum photons for detection

    # Advanced interface (direct control)
    sigma_small::Float64 = 1.0    # Small Gaussian sigma in pixels
    sigma_large::Float64 = 2.0    # Large Gaussian sigma in pixels
    minval::Float64 = 0.0         # DoG intensity threshold

    # Box parameters
    boxsize::Int = 7              # ROI size in pixels
    overlap::Float64 = 2.0        # Max overlap between detections

    # Backend parameters
    backend::Symbol = :auto       # :cpu, :gpu, or :auto
    auto_timeout::Float64 = 300.0  # Max wait for GPU in :auto mode
    gpu_timeout::Float64 = Inf    # Max wait in :gpu mode
    on_wait::Union{Function,Nothing} = nothing  # Optional wait progress callback
end
```

**Usage:**
```julia
# PSF-aware (recommended)
config = BoxerConfig(psf_sigma=0.13, min_photons=500.0, boxsize=11)

# Advanced (direct control)
config = BoxerConfig(sigma_small=1.5, sigma_large=3.0, minval=10.0)

# GPU-specific
config = BoxerConfig(psf_sigma=0.13, backend=:gpu, gpu_timeout=60.0)
```

## Core Function

### `getboxes` - Two Calling Conventions

**Config-based (recommended for reusable settings):**
```julia
getboxes(imagestack, camera, config::BoxerConfig) -> (ROIBatch, BoxesInfo)
```

**Kwargs-based (convenient for one-off calls):**
```julia
getboxes(imagestack, camera=nothing; kwargs...) -> (ROIBatch, BoxesInfo)
```

Both conventions are equivalent - kwargs are forwarded to a BoxerConfig internally.

Main detection function. Applies DoG filtering, finds local maxima, extracts ROI patches.

**Arguments:**
- `imagestack::AbstractArray{<:Real}` - Input image stack (2D or 3D)
- `camera::Union{AbstractCamera,Nothing}` - Camera object (IdealCamera or SCMOSCamera)
- `config::BoxerConfig` - Configuration struct (config-based convention)

**Kwargs (kwargs-based convention):**

*PSF-Aware Interface (Recommended):*
- `psf_sigma::Real` - PSF sigma in microns (requires camera for pixel size conversion)
- `min_photons::Real` - Minimum total photons for detection (default: 500.0)

*Advanced Interface (Direct Control):*
- `sigma_small::Real` - Small Gaussian sigma in pixels (default: 1.0)
- `sigma_large::Real` - Large Gaussian sigma in pixels (default: 2.0)
- `minval::Real` - DoG intensity threshold (default: 0.0)

*Other Parameters:*
- `boxsize::Int` - ROI size in pixels (default: 7)
- `overlap::Real` - Maximum overlap between detections in pixels (default: 2.0)
- `backend::Symbol` - Compute backend: `:cpu`, `:gpu`, or `:auto` (default: `:auto`)
  - `:cpu` - Always use CPU
  - `:gpu` - Require GPU, wait for memory if needed (waits forever by default)
  - `:auto` - Try GPU with timeout, fall back to CPU if memory unavailable
- `auto_timeout::Real` - Max seconds to wait for GPU in `:auto` mode (default: 300.0)
- `gpu_timeout::Real` - Max seconds to wait in `:gpu` mode (default: Inf)
- `on_wait::Function` - Optional callback `(elapsed, available, required) -> nothing` for wait progress

**Returns:** Tuple of `(ROIBatch, BoxesInfo)`

`ROIBatch` with fields:
- `data` - ROI stack (boxsize × boxsize × n_rois)
- `x_corners` - Vector of x (column) corner positions
- `y_corners` - Vector of y (row) corner positions
- `frame_indices` - Vector of frame indices for each ROI
- `camera` - Camera object (provided or default IdealCamera)
- `roi_size` - Size of each ROI (square)

`BoxesInfo` with fields:
- `backend` - Compute backend used (`:gpu` or `:cpu`)
- `elapsed_s` - Wall time in seconds
- `device_id` - GPU device ID (0-based), or -1 for CPU
- `n_rois` - Number of ROIs detected
- `batch_size` - Frames per batch during processing
- `n_batches` - Number of batches processed
- `memory_per_batch` - Estimated memory per batch in bytes

### `recommend_batch_size(height, width; backend=:auto, memory_fraction=0.8) -> Int`

Returns the recommended maximum number of frames to load at once given memory constraints.

Use this when processing very large datasets to determine optimal chunk size before calling `getboxes()`.

**Arguments:**
- `height::Int` - Image height in pixels
- `width::Int` - Image width in pixels
- `backend::Symbol` - Compute backend: `:cpu`, `:gpu`, or `:auto` (default: `:auto`)
- `memory_fraction::Real` - Fraction of free memory to use (default: 0.8)

**Returns:** Maximum recommended number of frames to load at once

**Memory Model:**
The processing pipeline requires approximately 6× the raw image size:
- Input imagestack
- Filtered stack (DoG output)
- Local maxima detection intermediates
- Coordinate arrays
- Box extraction workspace
- Broadcast temporaries

**Example:**
```julia
using SMLMBoxer

# Check how many 512×512 frames to load at once
max_frames = recommend_batch_size(512, 512)
println("Load up to $max_frames frames at a time")

# Load and process in chunks
for chunk_start in 1:max_frames:total_frames
    chunk_end = min(chunk_start + max_frames - 1, total_frames)
    imagestack = load_frames(chunk_start:chunk_end)
    (roi_batch, info) = getboxes(imagestack, camera; psf_sigma=0.13)
    # ... process results
end
```

## Re-Exported Types

### `ROIBatch` (from SMLMData.jl)
Container for multiple ROIs from particle detection. Supports iteration and indexing.

```julia
# Iteration
for roi in roi_batch
    # roi is a SingleROI with .data, .corner, .frame_idx, .camera
    process(roi.data)
end

# Indexing
roi = roi_batch[1]  # Returns SingleROI
```

### `SingleROI` (from SMLMData.jl)
Individual ROI with image data, position, frame index, and camera calibration.

## Common Workflows

### PSF-Aware Detection (Recommended)
```julia
using SMLMBoxer, SMLMData

# Create camera with physical pixel size (100nm pixels)
camera = IdealCamera(1:256, 1:256, 0.1f0)

# Config-based (recommended for reusable settings)
config = BoxerConfig(psf_sigma=0.13, min_photons=500.0, boxsize=11)
(roi_batch, info) = getboxes(imagestack, camera, config)

# OR kwargs-based (convenient for one-off calls)
(roi_batch, info) = getboxes(imagestack, camera;
    psf_sigma = 0.13,        # 130nm PSF in microns
    min_photons = 500.0,     # Minimum 500 photons
    boxsize = 11)

# Access results
n_detections = length(roi_batch.x_corners)
boxes = roi_batch.data              # 11×11×n ROI patches
positions_x = roi_batch.x_corners   # Column positions
positions_y = roi_batch.y_corners   # Row positions
frames = roi_batch.frame_indices

# Check processing info
println("Backend: ", info.backend)
println("Elapsed: ", info.elapsed_s * 1000, " ms")
```

### sCMOS Variance-Weighted Detection
```julia
using SMLMData

# Create sCMOS camera with readnoise map (Float32 for type consistency)
readnoise_map = Float32.(load_readnoise_calibration("camera_calib.mat"))
camera = SCMOSCamera(256, 256, 0.1f0, readnoise_map)

# Variance-weighted detection (automatically enabled)
(roi_batch, info) = getboxes(imagestack, camera;
    psf_sigma = 0.13,
    min_photons = 300.0,  # Lower threshold possible with noise weighting
    backend = :auto)      # GPU with CPU fallback
```

### Advanced: Direct Parameter Control
```julia
# Expert mode: bypass PSF-aware interface

# Config-based
config = BoxerConfig(sigma_small=1.5, sigma_large=3.0, minval=10.0, boxsize=9, overlap=1.5)
(roi_batch, info) = getboxes(imagestack, nothing, config)

# OR kwargs-based
(roi_batch, info) = getboxes(imagestack;
    sigma_small = 1.5,   # Custom filter sigma (pixels)
    sigma_large = 3.0,
    minval = 10.0,       # Direct intensity threshold
    boxsize = 9,
    overlap = 1.5)
```

### Processing Individual ROIs
```julia
(roi_batch, info) = getboxes(imagestack, camera; psf_sigma=0.13)

# Iterate over ROIs
for roi in roi_batch
    # SingleROI fields:
    # - roi.data: Image patch
    # - roi.corner: (x, y) corner position
    # - roi.frame_idx: Frame index
    # - roi.camera: Camera ROI calibration

    fit_gaussian(roi.data)
end

# Direct indexing
first_roi = roi_batch[1]
```

## Internal Functions (Not Exported)

These functions are documented but not part of the public API. Use at your own risk as they may change.

**Filtering:**
- `dog_filter(imagestack, args)` - Apply DoG filter (routes to standard or variance-weighted)
- `dog_filter_variance_weighted(imagestack, σ_small, σ_large, args)` - Variance-weighted DoG
- `convolve(imagestack, kernel; use_gpu)` - Standard convolution via NNlib
- `convolve_variance_weighted(imagestack, variance_map, σ, use_gpu)` - Variance-weighted via KA
- `gaussian_2d(sigma, kernelsize)` - Create 2D Gaussian kernel
- `dog_kernel(sigma_small, sigma_large)` - Create DoG kernel

**Local Maxima Detection:**
- `findlocalmax(imagestack, kernelsize; minval, use_gpu)` - Find local maximum coordinates
- `genlocalmaximage(imagestack, kernelsize; minval, use_gpu)` - Generate local max image

**Coordinate Processing:**
- `maxima2coords(imagestack)` - Convert non-zero pixels to coordinates
- `removeoverlap(coords, args)` - Remove overlapping detections

**Box Extraction:**
- `getboxstack(imagestack, coords, args)` - Extract ROI patches at coordinates
- `fillbox!(box, imagestack, row, col, im, boxsize)` - Fill single ROI patch

**Helper Functions:**
- `get_pixel_size(camera)` - Extract pixel size from camera
- `photons_to_dog_threshold(min_photons, psf_sigma)` - Convert photon threshold to DoG threshold
- `pixels_to_microns(pixel_coords, camera)` - Coordinate conversion
- `extract_camera_roi(camera, row_range, col_range)` - Extract camera calibration for ROI
- `get_variance_map(camera, imagesize)` - Compute variance map from camera
- `reshape_for_flux(arr)` - Reshape array for NNlib convolution

## Algorithm Details

### Detection Pipeline
1. **Filtering**: Apply DoG filter (standard or variance-weighted based on camera type)
2. **Local Maxima**: Find peaks above threshold using max pooling
3. **Overlap Removal**: Eliminate overlapping detections (keep higher intensity)
4. **Box Extraction**: Cut out ROI patches around each detection
5. **ROIBatch Construction**: Package results with camera calibration

### PSF-Aware Parameter Conversion
When `psf_sigma` (microns) is provided:
```julia
# Convert to pixels
psf_sigma_pixels = psf_sigma / pixel_size

# Automatic filter sizing
sigma_small = 1.0 × psf_sigma_pixels  # Match PSF
sigma_large = 2.0 × psf_sigma_pixels  # Background suppression

# Photon threshold → DoG intensity threshold
σ_eff = √(psf_sigma² + sigma_small²)  # Effective sigma after filtering
peak_filtered = min_photons / (2π × σ_eff²)
minval = 0.65 × peak_filtered  # DoG reduction factor
```

### Variance-Weighted Convolution
For sCMOS cameras with readnoise map:
```julia
# At each pixel (i,j):
weightsum = sum(gaussian_weight / variance[ii,jj] * input[ii,jj])
varsum = sum(gaussian_weight / variance[ii,jj])
output[i,j] = weightsum / varsum
```

Implements optimal inverse variance weighting for spatially-varying noise.

## Performance Notes

- **GPU Memory Management**: Automatically batches frames if image stack exceeds GPU memory
- **Type Stability**: All inputs converted to Float32 at entry point
- **Multi-GPU NVML Polling**: Scans all GPUs via NVML without creating CUDA contexts. Checks free memory, process contention, and compute utilization. First GPU with sufficient memory and low contention wins.
- **Contention-Safe Retry**: Unified retry loop handles all GPU failure modes (insufficient memory, TOCTOU context race, runtime OOM). Releases memory and re-polls with remaining timeout budget.
- **Memory Pool Reclaim**: Calls `GC.gc() + CUDA.reclaim()` after both successful and failed GPU processing to return memory to the system, preventing finished jobs from blocking other processes.
- **Jittered Backoff**: NVML polling uses jittered sleep intervals to avoid thundering herd when multiple processes compete for GPUs.
- **Backend Abstraction**: KernelAbstractions enables same code for CPU/GPU variance weighting
- **Typical Speedup**: 10-100x with GPU depending on image size and number of frames
