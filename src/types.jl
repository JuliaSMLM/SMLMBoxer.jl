"""
    BoxesInfo

Metadata returned alongside ROIBatch from getboxes().

# Fields
- `backend::Symbol`: Compute backend used (:gpu or :cpu)
- `elapsed_s::Float64`: Wall time in seconds
- `device_id::Int`: GPU device ID (0-based), or -1 for CPU
- `n_rois::Int`: Number of ROIs detected
- `batch_size::Int`: Frames per batch during processing
- `n_batches::Int`: Number of batches processed
- `memory_per_batch::Int`: Estimated memory per batch in bytes
"""
struct BoxesInfo
    backend::Symbol
    elapsed_s::Float64
    device_id::Int
    n_rois::Int
    batch_size::Int
    n_batches::Int
    memory_per_batch::Int
end

function Base.show(io::IO, info::BoxesInfo)
    elapsed_ms = info.elapsed_s * 1000
    mem_kb = info.memory_per_batch / 1024
    mem_str = mem_kb >= 1024 ? "$(round(mem_kb/1024, digits=1)) MB" : "$(round(mem_kb, digits=1)) KB"
    print(io, "BoxesInfo($(info.n_rois) ROIs, $(round(elapsed_ms, digits=1)) ms, $(info.backend), $(info.n_batches) batches × $(info.batch_size), $(mem_str)/batch)")
end

"""
    get_pixel_size(camera::AbstractCamera)

Extract pixel size from camera pixel edges (in microns).
Assumes approximately square pixels - returns x-direction pixel size.

For non-square pixels, pixel_size_x and pixel_size_y may differ slightly.
This function returns pixel_size_x for simplicity.

# Arguments
- `camera`: Camera object (IdealCamera or SCMOSCamera)

# Returns
- Pixel size in microns (x-direction)
"""
function get_pixel_size(camera::AbstractCamera)
    # Pixel size is the difference between consecutive edges
    return camera.pixel_edges_x[2] - camera.pixel_edges_x[1]
end

"""
    get_effective_gain(camera::AbstractCamera)

Get effective gain for converting photons to image ADU units (ADU/photon).

For IdealCamera: returns 1.0 (assumes image is in photon units)
For SCMOSCamera: returns QE / gain (photons → ADU conversion factor)
  - SMLMData defines gain as e⁻/ADU (electrons per ADU)
  - Physical conversion: photon → QE electrons → electrons/gain ADU
  - So: ADU/photon = QE / gain

Used to convert photon-based thresholds to image-unit thresholds.
"""
function get_effective_gain(camera::IdealCamera)
    return 1.0f0  # IdealCamera assumes photon units
end

function get_effective_gain(camera::SCMOSCamera)
    # For threshold calculation, use mean values if arrays
    # SMLMData defines gain as e⁻/ADU (electrons per ADU)
    # Conversion: ADU = (photons × QE) / gain
    # So effective_gain (ADU/photon) = QE / gain
    gain = camera.gain isa Real ? camera.gain : mean(camera.gain)
    qe = camera.qe isa Real ? camera.qe : mean(camera.qe)
    return Float32(qe / gain)  # Combined: photons → ADU (QE / gain, not QE × gain)
end

"""
    photons_to_dog_threshold(min_photons, psf_sigma; effective_gain=1.0)

Convert total photon count threshold to DoG filter intensity threshold in image units (ADU).

# Arguments
- `min_photons`: Minimum signal photons above background for detection
- `psf_sigma`: PSF sigma in pixels
- `effective_gain`: Camera gain factor (QE × gain) to convert photons → ADU (default: 1.0)

# Returns
- `minval`: DoG filter intensity threshold in image units (ADU)

# Physics
For a 2D Gaussian PSF with total photons N and sigma σ_psf, the peak intensity is:
    I_peak = N / (2π σ_psf²)  [photons/pixel]

After convolution with the small Gaussian filter (sigma_small = 1.0 × psf_sigma),
the effective sigma becomes:
    σ_eff = √(σ_psf² + sigma_small²)

The peak after filtering is:
    I_filtered = N / (2π σ_eff²)  [photons/pixel]

The DoG response (small - large Gaussian) has a lower peak than the small Gaussian alone.
For sigma_large = 2×sigma_small, the DoG peak is approximately 0.65× the small Gaussian peak.

For raw camera data in ADU, the threshold is scaled by effective_gain = QE × gain.
"""
function photons_to_dog_threshold(min_photons::Real, psf_sigma::Real; effective_gain::Real=1.0)
    # DoG filter uses sigma_small = 1.0 × psf_sigma
    sigma_small = 1.0 * psf_sigma

    # Effective sigma after small Gaussian blur (convolution of two Gaussians)
    σ_eff = sqrt(psf_sigma^2 + sigma_small^2)

    # Peak intensity after small Gaussian filtering (in photons)
    peak_filtered_photons = min_photons / (2π * σ_eff^2)

    # Convert to image units (ADU) using camera gain
    peak_filtered = peak_filtered_photons * effective_gain

    # DoG reduces peak (empirical factor for sigma_large = 2×sigma_small)
    dog_factor = 0.65

    return Float32(dog_factor * peak_filtered)
end

"""
    GetBoxesArgs

Internal structure for getboxes parameters. Users should call getboxes() with
keyword arguments rather than constructing this directly.

# Primary Interface (Recommended)
- `psf_sigma::Real`: PSF sigma in microns (physical units, e.g., 0.13 for 130nm PSF)
  Requires camera to be provided for pixel size conversion.
- `min_photons::Real`: Minimum total photons for detection (default: 500.0)

When psf_sigma is provided:
- Converted to pixels using camera pixel size
- sigma_small = 1.0 × psf_sigma_pixels (automatically calculated)
- sigma_large = 2.0 × psf_sigma_pixels (automatically calculated)
- minval = photons_to_dog_threshold(min_photons, psf_sigma_pixels) (automatically calculated)

# Advanced Interface (Direct Control)
- `sigma_small::Real`: Small Gaussian sigma in pixels (default: 1.0)
- `sigma_large::Real`: Large Gaussian sigma in pixels (default: 2.0)
- `minval::Real`: DoG intensity threshold (default: 0.0)

# Other Parameters
- `imagestack`: Input image stack
- `camera`: Camera object (IdealCamera or SCMOSCamera)
- `boxsize::Int`: ROI box size in pixels (default: 7)
- `overlap::Real`: Maximum overlap between detections in pixels (default: 2.0)
- `use_gpu::Bool`: Use GPU acceleration (default: true) - DEPRECATED, use backend instead
- `backend::Symbol`: Compute backend :cpu, :gpu, or :auto (default: :auto)
- `auto_timeout::Real`: Max wait seconds for :auto mode before CPU fallback (default: 30.0)
- `gpu_timeout::Real`: Max wait seconds for :gpu mode (default: Inf)
- `on_wait`: Optional callback(elapsed, available, required) for wait progress
"""
mutable struct GetBoxesArgs
    imagestack::AbstractArray
    camera::Union{AbstractCamera,Nothing}
    boxsize::Int
    overlap::Float32
    sigma_small::Float32
    sigma_large::Float32
    minval::Float32
    use_gpu::Bool
    backend::Symbol
    auto_timeout::Float64
    gpu_timeout::Float64
    on_wait::Union{Function,Nothing}

    # Inner constructor handles conversion logic
    function GetBoxesArgs(;
        imagestack = rand(Float32, 256, 256, 50) .> 0.999,
        camera::Union{AbstractCamera,Nothing} = nothing,
        boxsize::Int = 7,
        overlap::Real = 2.0,
        psf_sigma::Union{Real,Nothing} = nothing,
        min_photons::Real = 500.0,
        sigma_small::Union{Real,Nothing} = nothing,
        sigma_large::Union{Real,Nothing} = nothing,
        minval::Union{Real,Nothing} = nothing,
        use_gpu::Union{Bool,Nothing} = nothing,
        backend::Symbol = :auto,
        auto_timeout::Real = 30.0,
        gpu_timeout::Real = Inf,
        on_wait::Union{Function,Nothing} = nothing
    )
        # Determine which interface is being used
        if psf_sigma !== nothing
            # NEW INTERFACE: PSF-aware detection (recommended)
            # psf_sigma is in physical units (microns) - convert to pixels
            if camera !== nothing
                pixel_size_μm = get_pixel_size(camera)
                psf_sigma_pixels = psf_sigma / pixel_size_μm
            else
                error("psf_sigma in physical units (microns) requires camera to be provided. " *
                      "Either provide a camera or use the advanced interface with sigma_small/sigma_large in pixels.")
            end

            σ_small = Float32(1.0 * psf_sigma_pixels)
            σ_large = Float32(2.0 * psf_sigma_pixels)
            effective_gain = get_effective_gain(camera)
            min_val = photons_to_dog_threshold(min_photons, psf_sigma_pixels; effective_gain=effective_gain)
        else
            # OLD INTERFACE: Direct control (backward compatible)
            σ_small = Float32(sigma_small !== nothing ? sigma_small : 1.0)
            σ_large = Float32(sigma_large !== nothing ? sigma_large : 2.0)
            min_val = Float32(minval !== nothing ? minval : 0.0)
        end

        # Handle backwards compatibility: use_gpu overrides backend if explicitly set
        actual_backend = backend
        if use_gpu !== nothing
            actual_backend = use_gpu ? :auto : :cpu
        end

        # Validate backend
        actual_backend in (:cpu, :gpu, :auto) || error("backend must be :cpu, :gpu, or :auto")

        # use_gpu is determined later in _getboxes_impl based on backend and memory availability
        # For now, set it based on backend intent (will be refined during processing)
        initial_use_gpu = actual_backend != :cpu

        new(imagestack, camera, boxsize, Float32(overlap), σ_small, σ_large, min_val,
            initial_use_gpu, actual_backend, Float64(auto_timeout), Float64(gpu_timeout), on_wait)
    end
end

"""
    pixels_to_microns(pixel_coords, camera::AbstractCamera)

Convert pixel coordinates (row, col) to micron coordinates (x, y) using camera geometry.

# Arguments
- `pixel_coords`: N×2 matrix of (row, col) coordinates
- `camera`: Camera object with pixel_edges_x and pixel_edges_y

# Returns
- N×2 matrix of (x, y) coordinates in microns
"""
function pixels_to_microns(pixel_coords::AbstractMatrix, camera::AbstractCamera)
    ncoords = size(pixel_coords, 1)
    coords_microns = similar(pixel_coords)

    for i in 1:ncoords
        row, col = pixel_coords[i, 1], pixel_coords[i, 2]
        # Convert to 1-based pixel centers
        # pixel_edges are the edges, so center of pixel i is at (edges[i] + edges[i+1])/2
        x = (camera.pixel_edges_x[Int(col)] + camera.pixel_edges_x[Int(col)+1]) / 2
        y = (camera.pixel_edges_y[Int(row)] + camera.pixel_edges_y[Int(row)+1]) / 2
        coords_microns[i, 1] = x
        coords_microns[i, 2] = y
    end

    return coords_microns
end

"""
    extract_camera_roi(camera::AbstractCamera, row_range, col_range)

Extract a camera ROI with calibration data for the specified pixel region.

# Arguments
- `camera`: Source camera object
- `row_range`: Range of rows to extract
- `col_range`: Range of columns to extract

# Returns
- Camera object of the same type with ROI calibration data
"""
function extract_camera_roi(camera::IdealCamera{T}, row_range, col_range) where T
    return IdealCamera(
        camera.pixel_edges_x[col_range],  # pixel_edges_x (positional)
        camera.pixel_edges_y[row_range]   # pixel_edges_y (positional)
    )
end

function extract_camera_roi(camera::SCMOSCamera{T}, row_range, col_range) where T
    # Handle both scalar and per-pixel calibration parameters
    # SMLMData 0.6+: SCMOSCamera calibration arrays use (ny, nx) = (rows, cols) convention
    # This matches Julia's standard image indexing: array[row, col]
    offset = camera.offset isa AbstractArray ? camera.offset[row_range[1:end-1], col_range[1:end-1]] : camera.offset
    gain = camera.gain isa AbstractArray ? camera.gain[row_range[1:end-1], col_range[1:end-1]] : camera.gain
    readnoise = camera.readnoise isa AbstractArray ? camera.readnoise[row_range[1:end-1], col_range[1:end-1]] : camera.readnoise
    qe = camera.qe isa AbstractArray ? camera.qe[row_range[1:end-1], col_range[1:end-1]] : camera.qe

    return SCMOSCamera(
        camera.pixel_edges_x[col_range],  # pixel_edges_x (positional)
        camera.pixel_edges_y[row_range],  # pixel_edges_y (positional)
        readnoise = readnoise,
        offset = offset,
        gain = gain,
        qe = qe
    )
end

"""
    get_variance_map(camera::AbstractCamera, imagesize)

Compute variance map from camera calibration.

# Arguments
- `camera`: Camera object with noise calibration
- `imagesize`: Tuple of (nrows, ncols) for the image

# Returns
- Variance map (variance = readnoise²) matching image dimensions
"""
function get_variance_map(camera::IdealCamera{T}, imagesize::Tuple{Int,Int}) where T
    # IdealCamera has no readnoise, return uniform variance of 1.0
    # Always Float32 to match imagestack type (getboxes converts all images to Float32)
    return ones(Float32, imagesize)
end

function get_variance_map(camera::SCMOSCamera{T}, imagesize::Tuple{Int,Int}) where T
    nrows, ncols = imagesize

    if camera.readnoise isa AbstractArray
        # Per-pixel readnoise map: variance = readnoise²
        variance_map = camera.readnoise .^ 2
        # SMLMData 0.6+: readnoise uses (ny, nx) = (rows, cols), matching image convention
        # Defensive transpose for legacy data with inverted convention
        if size(variance_map) == (ncols, nrows) && ncols != nrows
            variance_map = transpose(variance_map)
        end
        @assert size(variance_map) == imagesize "Readnoise map size $(size(variance_map)) doesn't match image size $imagesize"
        # Convert to Float32 to match imagestack type (getboxes converts all images to Float32)
        return Float32.(collect(variance_map))
    else
        # Scalar readnoise: uniform variance (always Float32 to match imagestack)
        return fill(Float32(camera.readnoise^2), imagesize)
    end
end

"""
    recommend_batch_size(height, width; backend=:auto, memory_fraction=0.8) -> Int

Return recommended maximum number of frames to load at once given memory constraints.

This helps users decide how much data to load before calling `getboxes()`. For very large
datasets, loading data in chunks of this size ensures efficient processing without
running out of memory.

# Arguments
- `height::Int`: Image height in pixels
- `width::Int`: Image width in pixels
- `backend::Symbol`: Compute backend :cpu, :gpu, or :auto (default: :auto)
- `memory_fraction::Real`: Fraction of free memory to use (default: 0.8)
- `use_gpu::Bool`: DEPRECATED - use backend instead

# Returns
- Maximum recommended number of frames to load at once

# Memory Model
The processing pipeline requires approximately 6× the raw image size:
- Input imagestack
- Filtered stack (DoG output)
- Local maxima detection intermediates
- Coordinate arrays
- Box extraction workspace
- Broadcast temporaries

# Example
```julia
using SMLMBoxer

# Check how many 512×512 frames to load at once
max_frames = recommend_batch_size(512, 512)
println("Load up to \$max_frames frames at a time")

# Load and process in chunks
for chunk_start in 1:max_frames:total_frames
    chunk_end = min(chunk_start + max_frames - 1, total_frames)
    imagestack = load_frames(chunk_start:chunk_end)
    roi_batch = getboxes(imagestack, camera; psf_sigma=0.13)
    # ... process results
end
```
"""
function recommend_batch_size(height::Int, width::Int;
        backend::Symbol=:auto,
        use_gpu::Union{Bool,Nothing}=nothing,
        memory_fraction::Real=0.8)
    # Memory multiplier: accounts for all processing stages
    # Matches n_copies in _getboxes_impl for consistency
    n_copies = 6
    bytes_per_frame = height * width * sizeof(Float32) * n_copies

    # Handle backwards compatibility
    actual_backend = backend
    if use_gpu !== nothing
        actual_backend = use_gpu ? :auto : :cpu
    end

    # Determine if GPU should be used
    use_gpu_actual = actual_backend != :cpu && has_cuda() && CUDA.functional()

    if use_gpu_actual
        # GPU: find device with most free memory
        max_free_mem = 0
        for i in 0:length(CUDA.devices())-1
            CUDA.device!(i)
            free_mem = CUDA.free_memory()
            max_free_mem = max(max_free_mem, free_mem)
        end
        available = max_free_mem * memory_fraction
    else
        # CPU: use system free memory
        available = Sys.free_memory() * memory_fraction
    end

    return max(1, floor(Int, available / bytes_per_frame))
end