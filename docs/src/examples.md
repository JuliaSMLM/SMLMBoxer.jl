# Examples

This page demonstrates common usage patterns for SMLMBoxer.jl with complete working examples.

## Basic Detection Pipeline

This example shows the complete workflow from synthetic data generation to detection and validation.

### Setup and Synthetic Data

```julia
using SMLMBoxer, SMLMData, SMLMSim
using MicroscopePSFs

# Create camera
pixel_size = 0.1f0  # μm/pixel (Float32)
camera = IdealCamera(1:256, 1:256, pixel_size)

# Create synthetic emitters
emitters = Emitter2DFit{Float64}[]
for i in 1:10, j in 1:10
    x = (i + 0.5) * 2.0  # 2 μm spacing
    y = (j + 0.5) * 2.0
    photons = 1000.0
    bg = 10.0
    push!(emitters, Emitter2DFit{Float64}(
        x, y, photons, bg,
        0.0, 0.0, 0.0, 0.0,  # uncertainties
        1, 1, 0, length(emitters)+1  # frame, dataset, track, id
    ))
end

# Generate synthetic images
smld = BasicSMLD(emitters, camera, 1, 1)
psf = GaussianPSF(0.13)  # 130 nm sigma
imagestack = gen_images(smld, psf, poisson_noise=true, bg=10.0)
```

### Detection with PSF-Aware Interface

```julia
# Detect using recommended PSF-aware interface
roi_batch = getboxes(Float32.(imagestack), camera;
    psf_sigma = 0.13,      # PSF sigma in microns
    min_photons = 500.0,   # Detection threshold
    boxsize = 11,          # ROI size
    backend = :cpu)        # Use :auto for GPU acceleration

# Check results
println("Ground truth: $(length(emitters)) emitters")
println("Detected: $(length(roi_batch)) ROIs")
println("Detection rate: $(length(roi_batch)/length(emitters)*100)%")
```

### Accessing Detection Results

```julia
# ROIBatch provides multiple access patterns

# 1. Direct field access
boxes = roi_batch.data              # (boxsize × boxsize × n_rois)
x_positions = roi_batch.x_corners   # x (column) corners
y_positions = roi_batch.y_corners   # y (row) corners
frames = roi_batch.frame_indices    # frame numbers

# 2. Iteration over SingleROI objects
for (i, roi) in enumerate(roi_batch)
    println("ROI $i:")
    println("  Corner: ($(roi.corner[1]), $(roi.corner[2]))")
    println("  Frame: $(roi.frame_idx)")
    println("  Data size: $(size(roi.data))")
    println("  Total intensity: $(sum(roi.data))")
end

# 3. Array-like indexing
first_roi = roi_batch[1]  # Returns SingleROI
println("First detection at ($(first_roi.corner[1]), $(first_roi.corner[2]))")
```

## sCMOS Camera with Variance Weighting

This example demonstrates variance-weighted detection for sCMOS cameras with spatially-varying noise.

### Creating sCMOS Camera

```julia
using SMLMBoxer, SMLMData, SMLMSim

# Create readnoise map with spatial gradient
n_pixels = 256
readnoise_map = zeros(Float32, n_pixels, n_pixels)
for i in 1:n_pixels, j in 1:n_pixels
    # Low noise (2 e⁻) on left, high noise (20 e⁻) on right
    gradient = (j - 1) / (n_pixels - 1)
    readnoise_map[i, j] = 2.0 + 18.0 * gradient
end

# Create sCMOS camera (use Float32 literals to match readnoise_map type)
camera = SCMOSCamera(
    n_pixels, n_pixels,     # dimensions
    0.1f0,                  # pixel size (μm)
    readnoise_map,          # per-pixel readnoise (e⁻)
    offset = 100.0f0,       # ADU
    gain = 2.0f0,           # e⁻/ADU
    qe = 0.9f0              # quantum efficiency
)
```

### Detection with Automatic Variance Weighting

```julia
# Generate sCMOS image (see basic example for emitter creation)
imagestack = gen_images(smld, psf, camera_noise=true, bg=10.0)

# Detection automatically uses variance weighting for SCMOSCamera
roi_batch = getboxes(Float32.(imagestack), camera;
    psf_sigma = 0.13,
    min_photons = 500.0,
    boxsize = 11,
    backend = :auto)  # GPU with CPU fallback

# Analyze detection uniformity across noise regions
function count_by_noise_region(batch, n_pixels, split_col)
    low_noise = 0
    high_noise = 0
    for i in 1:length(batch)
        if batch.x_corners[i] < split_col
            low_noise += 1
        else
            high_noise += 1
        end
    end
    return low_noise, high_noise
end

low, high = count_by_noise_region(roi_batch, n_pixels, n_pixels÷2)
println("Low-noise region detections: $low")
println("High-noise region detections: $high")
```

## Advanced Interface: Direct Parameter Control

For expert users who want precise control over filter parameters.

### Custom DoG Parameters

```julia
# Direct control over DoG filter sigmas and threshold
roi_batch = getboxes(imagestack;
    sigma_small = 1.5,  # Small Gaussian sigma (pixels)
    sigma_large = 3.0,  # Large Gaussian sigma (pixels)
    minval = 15.0,      # DoG intensity threshold
    boxsize = 9,
    overlap = 2.0)

# This bypasses PSF-aware automatic configuration
# Useful for non-standard PSFs or custom detection criteria
```

### Parameter Tuning Workflow

```julia
# Test different sigma values to find optimal detection
sigma_values = [1.0, 1.3, 1.5, 2.0]
results = []

for sigma in sigma_values
    roi_batch = getboxes(imagestack, camera;
        sigma_small = sigma,
        sigma_large = 2.0 * sigma,
        minval = 10.0,
        backend = :cpu)

    push!(results, (sigma=sigma, n_detected=length(roi_batch)))
end

# Find optimal sigma
best = argmax(r -> r.n_detected, results)
println("Optimal sigma: $(results[best].sigma)")
println("Detections: $(results[best].n_detected)")
```

## GPU Acceleration

Leverage GPU acceleration for processing large datasets.

### Basic GPU Usage

```julia
using CUDA

# Check GPU availability
if CUDA.functional()
    println("GPU detected: $(CUDA.name(CUDA.device()))")
    println("Available memory: $(CUDA.available_memory() / 1024^3) GB")
else
    println("No GPU available, using CPU")
end

# GPU acceleration enabled by default
roi_batch_gpu = getboxes(imagestack, camera;
    psf_sigma = 0.13,
    min_photons = 500.0,
    backend = :auto)  # Try GPU, fall back to CPU

# Benchmark GPU vs CPU
using BenchmarkTools

@time roi_batch_cpu = getboxes(imagestack, camera;
    psf_sigma = 0.13, min_photons = 500.0, backend = :cpu)

@time roi_batch_gpu = getboxes(imagestack, camera;
    psf_sigma = 0.13, min_photons = 500.0, backend = :auto)
```

### Large Dataset Processing

```julia
# SMLMBoxer automatically batches large datasets that exceed GPU memory
large_stack = zeros(Float32, 512, 512, 1000)  # 1000 frames

# Automatic batching handles memory constraints
roi_batch = getboxes(large_stack, camera;
    psf_sigma = 0.13,
    min_photons = 500.0,
    backend = :auto)  # Batches frames to fit GPU memory

println("Processed $(size(large_stack, 3)) frames")
println("Total detections: $(length(roi_batch))")
```

## Integration with Downstream Analysis

### Preparing ROIs for Fitting

```julia
# ROIBatch integrates directly with SMLMData fitting pipelines

# Example: Filter ROIs by intensity before fitting
min_intensity = 100.0
filtered_rois = filter(roi_batch) do roi
    sum(roi.data) > min_intensity
end

println("Kept $(length(filtered_rois))/$(length(roi_batch)) ROIs")

# ROIs maintain coordinate information for fitting
for roi in filtered_rois
    # Each roi contains:
    # - roi.data: Image patch
    # - roi.corner: Position in camera coordinates
    # - roi.frame_idx: Frame number
    # Ready for localization fitting (e.g., GaussMLE.jl)
end
```

### Extracting ROI Centers

```julia
# Convert corner positions to approximate centers
roi_size = roi_batch.roi_size
center_offset = roi_size ÷ 2

x_centers = roi_batch.x_corners .+ center_offset
y_centers = roi_batch.y_corners .+ center_offset

# Convert to physical coordinates (microns)
pixel_size = camera.pixel_edges_x[2] - camera.pixel_edges_x[1]
x_microns = (x_centers .- 1) .* pixel_size
y_microns = (y_centers .- 1) .* pixel_size

println("First detection center: ($(x_microns[1]), $(y_microns[1])) μm")
```

## Common Patterns

### Multi-Frame Time Series

```julia
# Detect particles in each frame of time series
n_frames = size(imagestack, 3)

# getboxes processes entire stack and tracks frame indices
roi_batch = getboxes(imagestack, camera;
    psf_sigma = 0.13,
    min_photons = 500.0)

# Group detections by frame
using StatsBase
detections_per_frame = countmap(roi_batch.frame_indices)

println("Frame-by-frame detection counts:")
for frame in sort(collect(keys(detections_per_frame)))
    println("  Frame $frame: $(detections_per_frame[frame]) detections")
end
```

### Quality Filtering

```julia
# Filter detections by quality metrics
function filter_by_quality(roi_batch; min_snr=3.0)
    keep_indices = Int[]

    for i in 1:length(roi_batch)
        roi = roi_batch[i]

        # Calculate simple SNR estimate
        sorted_pixels = sort(vec(roi.data))
        signal = maximum(sorted_pixels)
        bg_estimate = median(sorted_pixels[1:length(sorted_pixels)÷4])
        noise_estimate = std(sorted_pixels[1:length(sorted_pixels)÷4])

        snr = (signal - bg_estimate) / noise_estimate

        if snr >= min_snr
            push!(keep_indices, i)
        end
    end

    return roi_batch[keep_indices]
end

# Apply filtering
high_quality_rois = filter_by_quality(roi_batch, min_snr=5.0)
println("Kept $(length(high_quality_rois))/$(length(roi_batch)) high-quality ROIs")
```

### Batch Processing Multiple Files

```julia
# Process multiple SMLM acquisitions
file_paths = ["dataset1.tif", "dataset2.tif", "dataset3.tif"]

all_detections = []
for (i, path) in enumerate(file_paths)
    # Load image stack (implementation depends on file format)
    imagestack = load_image_stack(path)

    # Detect
    roi_batch = getboxes(imagestack, camera;
        psf_sigma = 0.13,
        min_photons = 500.0)

    println("File $i: $(length(roi_batch)) detections")
    push!(all_detections, roi_batch)
end

# Combine results for aggregate analysis
total_detections = sum(length, all_detections)
println("Total detections across all files: $total_detections")
```
