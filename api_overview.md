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
- Each pixel weighted by `gaussian_kernel / variance`, with
  `variance = (readnoise / gain)^2` in ADU² (gain in e⁻/ADU)
- Low-noise pixels: high weight (strong detection influence)
- High-noise pixels: low weight (reduced false positives)
- GPU-accelerated via KernelAbstractions.jl (device-agnostic kernels)

### GPU Acceleration and Scheduling
- Standard DoG: NNlib on CPU, KernelAbstractions kernel on GPU. On 500 frames at
  256×256, IdealCamera GPU throughput is 13,805 images/s versus 876 with 32 CPU threads,
  measured on AMD Ryzen 9 5950X / NVIDIA RTX 4090 with SMLMBoxer 0.4.0, CUDA.jl 6.5.0
  and KernelAbstractions 0.10.4
- Variance-weighted: KernelAbstractions custom kernels (same code for CPU/GPU)
- Multi-GPU support: polling selects the eligible GPU with the most free memory

GPU support is NVIDIA only through CUDA.jl. AMD or Intel GPUs, or no GPU, use
the CPU under `:auto` and raise `GPU backend requested but CUDA is not functional`
under `:gpu`. Reactant loads beside SMLMBoxer after removal of the conflicting cuDNN
dependency; SMLMBoxer does not use Reactant or need setup for it.

**Unified GPU retry loop** handles multi-process contention:
1. Check that one frame, including the 1.5x safety margin, fits at least one visible GPU's
   total memory. If none can hold it, `:gpu` errors and `:auto` warns and uses the CPU
   at once
2. Poll all GPUs via NVML without creating CUDA contexts for sufficient free memory
   and low utilization. When NVML cannot report a GPU's memory, ask CUDA for it
3. Acquire GPU context and run processing
4. On GPU memory errors or an unavailable device (including context-creation races):
   try to release memory via `GC.gc() + CUDA.reclaim()`. Halve the failed batch's frame
   cap and retry immediately; only after a single-frame failure re-poll with the
   remaining timeout
5. On any other GPU failure: try to release memory; under `:auto`, warn and use the CPU
   at once; under `:gpu`, raise the original error at once
6. On timeout: `:auto` falls back to CPU, `:gpu` errors with the failed batch size and
   memory need
7. On success: release explicitly freed GPU memory without full garbage collection
   (see the CUDA.jl 6.5.x caveat below)

**Backend modes:**
- `:cpu` - Always CPU, no GPU involvement
- `:gpu` - Require GPU, retry GPU memory errors and an unavailable device until
  `gpu_timeout` (default: Inf); error on timeout; raise any other error at once
- `:auto` - Try GPU, retry GPU memory errors and an unavailable device until
  `auto_timeout` (default: 300s), then use the CPU; any other GPU failure uses the CPU
  at once

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
    overlap::Float64 = 2.0        # Local-maximum window is max(3, ⌊boxsize - overlap⌋)

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
Image (rows, columns) must match the camera pixel edges, or `DimensionMismatch` is raised.
Crop the camera together with the image. Odd and even box sizes share the corner rule
`clamp(center - (boxsize - 1) ÷ 2, 1, image_size - boxsize + 1)` on each axis.
After clamping, keep the brightest filtered detection per corner and frame.

**Arguments:**
- `imagestack::AbstractArray{<:Real}` - Input image stack (2D or 3D)
- `camera::Union{AbstractCamera,Nothing}` - Camera object (IdealCamera or SCMOSCamera)
- `config::BoxerConfig` - Configuration struct (config-based convention)

**Kwargs (kwargs-based convention):**

*PSF-Aware Interface (Recommended):*
- `psf_sigma::Real` - PSF sigma in microns (requires camera for pixel size conversion)
- `min_photons::Real` - Photon threshold (default: 500.0). A noise-free,
  pixel-integrated Gaussian centered on a pixel just reaches the threshold at this
  count; at a pixel corner it needs about 1.1× more photons.

*Advanced Interface (Direct Control):*
- `sigma_small::Real` - Small Gaussian sigma in pixels (default: 1.0)
- `sigma_large::Real` - Large Gaussian sigma in pixels (default: 2.0)
- `minval::Real` - DoG intensity threshold (default: 0.0)

*Other Parameters:*
- `boxsize::Int` - ROI size in pixels, odd or even, at least 1 (default: 7)
- `overlap::Real` - pixels (default 2.0). Local maxima are found in a square window of k =
  max(3, ⌊boxsize − overlap⌋) pixels, so two detections in one frame are at least ⌈k/2⌉
  pixels apart along x or y. A larger `overlap` lets detections sit closer together. This is
  not a bound on box overlap: with the defaults, neighboring boxes can share 4 rows or
  columns.
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
    roi_batch.frame_indices .+= chunk_start - 1
    # ... process results
end
```

## Re-Exported Types

### `ROIBatch` (from SMLMData.jl)
Container for multiple ROIs from particle detection. Supports iteration and indexing.

```julia
# Iteration
for roi in roi_batch
    # roi is a SingleROI with .data, .corner, .frame_idx
    process(roi.data)
end

# Indexing
roi = roi_batch[1]  # Returns SingleROI
```

### `SingleROI` (from SMLMData.jl)
Individual ROI: image patch, corner (x, y) = (column, row) in camera pixels (1-based), and
frame index. Calibration stays on the batch (`roi_batch.camera`, full image).

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
- `convolve(imagestack, kernel; use_gpu)` - NNlib on CPU, KernelAbstractions on GPU
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
3. **Overlap Removal**: Only maxima sharing a max-pool window compete; the dimmer
   goes, with ties keeping the earlier detection
4. **Box Extraction**: Clamp corners and retain one box per corner and frame,
   keeping the brightest filtered detection
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
# P[i,j] is the integral over pixel (i,j) of a unit-photon Gaussian centered on a pixel.
# Ks, Kl are the actual normalized small and large Gaussian kernels of this path.
minval = min_photons * effective_gain * (sum(Ks .* P_small) - sum(Kl .* P_large))
# effective_gain: 1 for IdealCamera (photons); QE / gain for SCMOSCamera (ADU)
```

The standard kernels come from `dog_kernel_pair`, with common odd size at least
`ceil(4 * sigma_large)`. The sCMOS reference uses uniform variance and its actual
windows, each with half-width `ceil(3 * sigma)`. Pixel integrals use numerical
quadrature accurate to 1e-4 relative; no empirical DoG factor is used.
At PSF sigma 1.35 pixels, the response is about 0.547 (standard) or 0.581 (sCMOS)
times `N / (2π * (sqrt(2) * 1.35)^2)`. A noise-free spot of `min_photons` photons
centered on a pixel just reaches the threshold; at a pixel corner it needs about
1.1× more. Noise and varying calibration affect real detections.

### Variance-Weighted Convolution
For sCMOS cameras with readnoise map:
```julia
# Calibration is converted per pixel when readnoise or gain is a map:
variance = (readnoise ./ gain).^2  # ADU²
# At each pixel (i,j):
weightsum = sum(gaussian_weight / variance[ii,jj] * input[ii,jj])
varsum = sum(gaussian_weight / variance[ii,jj])
output[i,j] = weightsum / varsum
```

Implements optimal inverse variance weighting for spatially-varying noise.

## Performance Notes

- **GPU Memory Management**: Automatically batches frames if image stack exceeds GPU memory
- **Type Stability**: All inputs converted to Float32 at entry point
- **Multi-GPU NVML Polling**: Checks free memory, process contention, and compute
  utilization via NVML without creating CUDA contexts. When NVML cannot report a GPU's
  memory, SMLMBoxer asks CUDA for it. The eligible GPU with the most free memory wins.
- **Contention-Safe Retry**: GPU memory errors and an unavailable device are retried until
  the timeout. Any other GPU failure uses the CPU at once under `:auto`, or raises the
  original error at once under `:gpu`. Each failed attempt tries to release memory;
  cleanup errors cannot mask the original error or prevent CPU fallback.
- **Memory Pool Release**: Explicitly frees owned GPU arrays within each batch, including
  on errors. Successful calls synchronize, purge caches and trim the pool without full
  garbage collection on CUDA.jl 6; CUDA.jl 5 uses its GC-free `CUDA.reclaim()`. Failed
  attempts still use `GC.gc(false) + CUDA.reclaim()`. Release after a short call takes
  0.013 ms on the RTX 4090 under CUDA.jl 6.5.0, which retains freed pool memory until
  process exit (see the changelog's known issue). Under CUDA.jl 6.4.2, release returns
  memory to the driver: about 43 ms per 2 GiB released, compared with 200 ms for
  `CUDA.reclaim()` on the same allocation workload.
- **Jittered Backoff**: NVML polling uses jittered sleep intervals to avoid thundering herd when multiple processes compete for GPUs.
- **Backend Abstraction**: KernelAbstractions enables same code for CPU/GPU variance weighting
- **Typical Speedup**: A warmed-up 256×256×10 GPU call takes about 0.75 ms in 0.4.0,
  down from about 57 ms in 0.3.4 on the RTX 4090 (medians of 0.73–0.84 ms across
  CUDA.jl 6.4.2 and 6.5.0). The GPU wins from one frame at every size tested (128,
  256 and 512 px), with either camera path and either one or 32 CPU threads on the
  AMD Ryzen 9 5950X. On 500 frames at 256×256, GPU throughput is 13,805 images/s with
  an IdealCamera and 10,643 with an SCMOSCamera, versus 876 and 76 with 32 CPU threads,
  measured with CUDA.jl 6.5.0 and KernelAbstractions 0.10.4.
