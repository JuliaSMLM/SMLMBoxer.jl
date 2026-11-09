"""
    BoxerConfig

Configuration for ROI detection via getboxes(). Use the keyword constructor; it
converts values to field types before validation (e.g. boxsize=9.0 and Int32 fields).

Images must be finite, 2D or 3D, with both dimensions >= boxsize. A zero-frame stack
returns an empty ROIBatch and BoxesInfo. Non-finite pixels raise ArgumentError with
the count and first (row, col, frame), including under :auto. Integer inputs skip the scan.
For SCMOSCamera, every readnoise, gain and qe value must be finite and > 0; errors
name the field and affected pixel count. For a noise-free readout, use IdealCamera
with the image in photons.

With an `SCMOSCamera`, the PSF-aware path (`psf_sigma`/`min_photons`) detects in
photons per pixel: `(adu - offset) * gain / qe`, with variance `(readnoise / qe)^2`
in photons². Offset, gain and QE may each be scalar or maps; offset must be finite
when this conversion is used. The threshold uses effective gain 1 and the
variance-weighted reference response to `min_photons` photons (uniform variance).
Conversion runs on an owned batch copy on its device. ROI pixels are still cut
from the caller's ADU image and retain the supplied camera and coordinate convention.
The direct interface (`sigma_small`, `sigma_large`, `minval`) filters the image as
given with `(readnoise / gain)^2` in ADU²; `minval` stays in those image units.

Use either PSF-aware interface (psf_sigma + min_photons) or advanced interface
(sigma_small + sigma_large + minval). PSF-aware is recommended.

# Fields

## PSF-Aware Interface (Recommended)
- `psf_sigma::Union{Float64,Nothing}`: PSF sigma in microns (e.g., 0.13 for 130nm PSF).
  Must be finite and > 0. Requires a camera with uniform square pixels
  (relative tolerance 1e-4); otherwise use sigma_small/sigma_large in pixels. When set, overrides sigma_small/sigma_large/minval.
- `min_photons::Union{Float64,Nothing}`: Positive finite photon threshold. Defaults to
  `nothing`; requires `psf_sigma`. With `psf_sigma`, omission stores 500.0. A noise-free,
  pixel-integrated Gaussian centered on a pixel just reaches the threshold at this
  photon count. A spot centered on a pixel corner needs more photons, about 1.1× at a
  PSF sigma of 1.35 pixels (0.135 µm on 0.1 µm pixels), 1.2× at 1 pixel and
  1.5× at 0.7 pixel.

## Advanced Interface (Direct Control)
Sigmas must be finite with `0 < sigma_small < sigma_large`; `minval` must be finite.

- `sigma_small::Float64`: Small Gaussian sigma in pixels (default: 1.0)
- `sigma_large::Float64`: Large Gaussian sigma in pixels (default: 2.0)
- `minval::Float64`: DoG intensity threshold (default: 0.0)

## Box Parameters
`boxsize >= 1` must fit both image dimensions; `overlap` must be finite and >= 0.

- `boxsize::Int`: ROI box size in pixels, odd or even (default: 7)
- `overlap::Float64`: pixels (default 2.0). Local maxima are found in a square window of k =
  max(3, ⌊boxsize − overlap⌋) pixels, so two detections in one frame are at least ⌈k/2⌉
  pixels apart along x or y. A larger `overlap` lets detections sit closer together. This is
  not a bound on box overlap: with the defaults, neighboring boxes can share 4 rows or
  columns.

## Backend Parameters
Timeouts must be >= 0 (Inf allowed); invalid parameters raise `ArgumentError`.

- `backend::Symbol`: Compute backend :cpu, :gpu, or :auto (default: :auto)
- `auto_timeout::Float64`: Max wait for GPU in :auto mode before CPU fallback (default: 300.0)
- `gpu_timeout::Float64`: Max wait for GPU in :gpu mode (default: Inf)
- `on_wait::Union{Function,Nothing}`: Optional callback `(elapsed, available, required) -> nothing` for GPU wait progress (default: nothing)

GPU memory errors and an unavailable device are retried until `auto_timeout` under `:auto`,
then processing falls back to the CPU; any other GPU failure uses the CPU at once.
Under `:gpu`, resource errors are retried until `gpu_timeout`; any other error is raised
at once; an interrupt (Ctrl-C) always stops the call.
Resource retries halve the batch cap immediately; after a single-frame failure they
resume device polling. A frame larger than any GPU can free errors under `:gpu` or
uses the CPU immediately under `:auto`.

# Examples
```julia
# PSF-aware (recommended)
config = BoxerConfig(psf_sigma=0.13, min_photons=500.0, boxsize=11)

# Advanced (direct control)
config = BoxerConfig(sigma_small=1.5, sigma_large=3.0, minval=10.0)

# GPU-specific
config = BoxerConfig(psf_sigma=0.13, backend=:gpu, gpu_timeout=60.0)
```
"""
struct BoxerConfig <: AbstractSMLMConfig
    # PSF-aware interface
    psf_sigma::Union{Float64,Nothing}
    min_photons::Union{Float64,Nothing}

    # Advanced interface (direct control)
    sigma_small::Float64
    sigma_large::Float64
    minval::Float64

    # Box parameters
    boxsize::Int
    overlap::Float64

    # Backend parameters
    backend::Symbol
    auto_timeout::Float64
    gpu_timeout::Float64
    on_wait::Union{Function,Nothing}
end

"""Validate keyword parameters shared by the configuration and internal argument paths."""
function validate_boxer_parameters(; psf_sigma, min_photons, sigma_small, sigma_large,
        minval, boxsize, overlap, backend, auto_timeout, gpu_timeout, on_wait)
    for (field, value) in ((:sigma_small, sigma_small), (:sigma_large, sigma_large),
                           (:psf_sigma, psf_sigma), (:min_photons, min_photons))
        value === nothing && continue
        isfinite(value) && value > 0 || throw(ArgumentError(
            "$field=$value must be finite and > 0"))
    end
    sigma_small < sigma_large || throw(ArgumentError(
        "sigma_small=$sigma_small must be smaller than sigma_large=$sigma_large"))
    min_photons !== nothing && psf_sigma === nothing && throw(ArgumentError(
        "min_photons=$min_photons requires psf_sigma; use minval for a direct pixel threshold"))
    isfinite(minval) || throw(ArgumentError("minval=$minval must be finite"))
    boxsize >= 1 || throw(ArgumentError("boxsize=$boxsize must be >= 1"))
    isfinite(overlap) && overlap >= 0 || throw(ArgumentError(
        "overlap=$overlap must be finite and >= 0"))
    backend in (:cpu, :gpu, :auto) || throw(ArgumentError(
        "backend=$backend must be :cpu, :gpu, or :auto"))
    for (field, value) in ((:auto_timeout, auto_timeout), (:gpu_timeout, gpu_timeout))
        value >= 0 || throw(ArgumentError("$field=$value must be >= 0 (Inf is allowed)"))
    end
    on_wait === nothing || on_wait isa Function || throw(ArgumentError(
        "on_wait=$on_wait must be a function or nothing"))
    return psf_sigma === nothing ? nothing : something(min_photons, 500.0)
end

function BoxerConfig(; psf_sigma=nothing,
        min_photons=nothing, sigma_small=1.0,
        sigma_large=2.0, minval=0.0, boxsize=7, overlap=2.0,
        backend=:auto, auto_timeout=300.0, gpu_timeout=Inf,
        on_wait=nothing)
    # Preserve @kwdef's field conversions, then validate the values actually stored.
    raw = (psf_sigma, min_photons, sigma_small, sigma_large, minval,
        boxsize, overlap, backend, auto_timeout, gpu_timeout, on_wait)
    converted = map(fieldtypes(BoxerConfig), fieldnames(BoxerConfig), raw) do T, field, value
        try
            convert(T, value)
        catch e
            e isa InterruptException && rethrow()
            throw(ArgumentError("$field=$value must be convertible to $T"))
        end
    end
    values = NamedTuple{fieldnames(BoxerConfig)}(converted)
    photons = validate_boxer_parameters(; values...)
    return BoxerConfig(values.psf_sigma, photons, values.sigma_small, values.sigma_large,
        values.minval, values.boxsize, values.overlap, values.backend,
        values.auto_timeout, values.gpu_timeout, values.on_wait)
end

function Base.show(io::IO, config::BoxerConfig)
    if config.psf_sigma !== nothing
        print(io, "BoxerConfig(psf_sigma=$(config.psf_sigma), min_photons=$(config.min_photons), boxsize=$(config.boxsize), backend=$(config.backend))")
    else
        print(io, "BoxerConfig(σ_small=$(config.sigma_small), σ_large=$(config.sigma_large), minval=$(config.minval), boxsize=$(config.boxsize), backend=$(config.backend))")
    end
end

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
struct BoxesInfo <: AbstractSMLMInfo
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

Extract pixel size in microns for PSF-aware detection. Every pixel must be square
and uniformly sized (relative tolerance 1e-4); otherwise use direct pixel sigmas.
The size is each axis's span divided by its pixel count, in Float64.

# Arguments
- `camera`: Camera object (IdealCamera or SCMOSCamera)

# Returns
- Pixel size in microns (x-direction), Float64
"""
function get_pixel_size(camera::AbstractCamera)
    pitch_x = uniform_pitch(camera.pixel_edges_x)
    pitch_y = uniform_pitch(camera.pixel_edges_y)
    pitch_x !== nothing && pitch_y !== nothing && isapprox(pitch_x, pitch_y; rtol=1e-4) ||
        throw(ArgumentError("psf_sigma requires uniform square pixels (relative tolerance 1e-4); " *
            "this camera can use sigma_small/sigma_large in pixels"))
    return pitch_x
end

"""
    uniform_pitch(edges) -> Union{Float64, Nothing}

Pixel pitch of one axis, its span over its pixel count in Float64, so an origin far from zero
keeps its accuracy; `nothing` if any pixel's width differs from it by more than 1e-4 relative
beyond the rounding of the stored edges (a width of a uniform camera stored in Float32 can be off
by one unit in the last place of its largest edge: 2.4e-4 of a pixel for 2048 pixels of 0.1 µm).
"""
function uniform_pitch(edges)
    n = length(edges) - 1
    n >= 1 || return nothing
    first_edge, last_edge = Float64(first(edges)), Float64(last(edges))
    pitch = (last_edge - first_edge) / n
    isfinite(pitch) && pitch > 0 || return nothing
    rounding = 2 * Float64(eps(float(max(abs(first(edges)), abs(last(edges))))))
    previous = first_edge
    for i in 2:(n + 1)
        edge = Float64(edges[i])
        abs((edge - previous) - pitch) <= 1e-4 * pitch + rounding || return nothing
        previous = edge
    end
    return pitch
end

"""
    get_effective_gain(camera::AbstractCamera)

Get effective gain for converting photons to image ADU units (ADU/photon).

For IdealCamera: returns 1.0 (assumes image is in photon units)
For SCMOSCamera: returns QE / gain (photons → ADU conversion factor)
  - SMLMData defines gain as e⁻/ADU (electrons per ADU)
  - Physical conversion: photon → QE electrons → electrons/gain ADU
  - So: ADU/photon = QE / gain

For maps this is a mean conversion factor. PSF-aware getboxes instead converts
each image pixel to photons and always uses effective gain 1 for its threshold.
"""
function get_effective_gain(camera::IdealCamera)
    return 1.0f0  # IdealCamera assumes photon units
end

function get_effective_gain(camera::SCMOSCamera)
    # Legacy mean conversion utility; PSF-aware thresholds do not use it.
    # SMLMData defines gain as e⁻/ADU (electrons per ADU)
    # Conversion: ADU = (photons × QE) / gain
    # So effective_gain (ADU/photon) = QE / gain
    gain = camera.gain isa Real ? camera.gain : mean(camera.gain)
    qe = camera.qe isa Real ? camera.qe : mean(camera.qe)
    return Float32(qe / gain)  # Combined: photons → ADU (QE / gain)
end

"""
    photons_to_dog_threshold(min_photons, psf_sigma; effective_gain=1.0,
                             variance_weighted=false)

Convert total photon count threshold to a DoG filter response. Public PSF-aware
detection works in photons (effective_gain=1), including sCMOS.

# Arguments
- `min_photons`: Minimum signal photons above background for detection
- `psf_sigma`: PSF sigma in pixels
- `effective_gain`: Optional scale from photons to filter units (default 1.0).
  Public PSF-aware detection uses 1.0 after per-pixel photon conversion for sCMOS.
- `variance_weighted`: Use the sCMOS Gaussian windows under uniform variance.

# Returns
- `minval`: DoG threshold in the scaled detection units (photons for PSF-aware calls)

# Calibration
The threshold is the peak response of this path's actual normalized Gaussian kernels
to a noise-free, pixel-integrated Gaussian of `min_photons` photons centered on a pixel,
times effective gain. See `BoxerConfig` for subpixel-position effects.
The standard path uses `dog_kernel_pair`; the sCMOS reference uses its own windows
under uniform variance. Spatially varying calibration and noise affect real detections.
"""
function photons_to_dog_threshold(min_photons::Real, psf_sigma::Real;
        effective_gain::Real=1.0, variance_weighted::Bool=false)
    sigma_small, sigma_large = Float32(psf_sigma), Float32(2 * psf_sigma)
    if variance_weighted
        kernel_small = gaussian_2d(sigma_small,
            2 * variance_gaussian_window(sigma_small) + 1)
        kernel_large = gaussian_2d(sigma_large,
            2 * variance_gaussian_window(sigma_large) + 1)
    else
        kernel_small, kernel_large = dog_kernel_pair(sigma_small, sigma_large)
    end

    # Separable pixel integrals, evaluated with composite Simpson quadrature.
    # At least 32 intervals per pixel and 16 per sigma resolve narrow spots as well.
    sigma = Float64(psf_sigma)
    intervals = max(32, 2 * ceil(Int, 8 / sigma))
    radius = max(size(kernel_small, 1), size(kernel_large, 1)) ÷ 2
    masses = [gaussian_pixel_mass(i, sigma, intervals) for i in -radius:radius]
    response = gaussian_spot_response(kernel_small, masses) -
        gaussian_spot_response(kernel_large, masses)
    return Float32(min_photons * effective_gain * response)
end

"""Integrate a unit Gaussian over one pixel using an even number of Simpson intervals."""
function gaussian_pixel_mass(offset::Int, sigma::Float64, intervals::Int)
    left = offset - 0.5
    density(x) = exp(-x^2 / (2 * sigma^2)) / (sqrt(2π) * sigma)
    total = density(left) + density(left + 1)
    for i in 1:intervals-1
        total += (isodd(i) ? 4 : 2) * density(left + i / intervals)
    end
    return total / (3 * intervals)
end

"""Return a normalized Gaussian kernel's response to separable pixel masses."""
function gaussian_spot_response(kernel, masses)
    radius = size(kernel, 1) ÷ 2
    center = (length(masses) + 1) ÷ 2
    response = 0.0
    for j in axes(kernel, 2), i in axes(kernel, 1)
        response += kernel[i, j] * masses[center + i - radius - 1] *
            masses[center + j - radius - 1]
    end
    return response
end

"""
    GetBoxesArgs

Internal structure for getboxes parameters. Users should call getboxes() with
keyword arguments rather than constructing this directly.

# Primary Interface (Recommended)
- `psf_sigma::Real`: PSF sigma in microns (physical units, e.g., 0.13 for 130nm PSF)
  Requires camera to be provided for pixel size conversion.
- `min_photons::Union{Real,Nothing}`: Positive finite photon threshold. Defaults to
  `nothing`; requires `psf_sigma`. With `psf_sigma`, omission uses 500.0. A noise-free,
  pixel-integrated Gaussian centered on a pixel just reaches the threshold at this
  photon count. See `BoxerConfig` for subpixel-position effects.

When psf_sigma is provided:
- Converted to pixels using camera pixel size
- sigma_small = 1.0 × psf_sigma_pixels (automatically calculated)
- sigma_large = 2.0 × psf_sigma_pixels (automatically calculated)
- minval from the actual filter kernels with effective gain 1 (uniform variance for sCMOS)

# Advanced Interface (Direct Control)
- `sigma_small::Real`: Small Gaussian sigma in pixels (default: 1.0)
- `sigma_large::Real`: Large Gaussian sigma in pixels (default: 2.0)
- `minval::Real`: DoG intensity threshold (default: 0.0)

# Other Parameters
- `imagestack`: Input image stack
- `camera`: Camera object (IdealCamera or SCMOSCamera)
- `boxsize::Int`: ROI box size in pixels, odd or even (default: 7)
- `overlap::Real`: pixels (default 2.0). Local maxima are found in a square window of k =
  max(3, ⌊boxsize − overlap⌋) pixels, so two detections in one frame are at least ⌈k/2⌉
  pixels apart along x or y. A larger `overlap` lets detections sit closer together. This is
  not a bound on box overlap: with the defaults, neighboring boxes can share 4 rows or
  columns.
- `backend::Symbol`: Compute backend :cpu, :gpu, or :auto (default: :auto)
- `auto_timeout::Real`: Max wait seconds for :auto mode before CPU fallback (default: 300.0)
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
    check_pixels::Bool
    detect_in_photons::Bool

    # Inner constructor handles conversion logic
    function GetBoxesArgs(;
        imagestack = rand(Float32, 256, 256, 50) .> 0.999,
        camera::Union{AbstractCamera,Nothing} = nothing,
        boxsize::Int = 7,
        overlap::Real = 2.0,
        psf_sigma::Union{Real,Nothing} = nothing,
        min_photons::Union{Real,Nothing} = nothing,
        sigma_small::Union{Real,Nothing} = nothing,
        sigma_large::Union{Real,Nothing} = nothing,
        minval::Union{Real,Nothing} = nothing,
        backend::Symbol = :auto,
        auto_timeout::Real = 300.0,
        gpu_timeout::Real = Inf,
        on_wait = nothing,
        check_pixels::Bool = !(eltype(imagestack) <: Integer)
    )
        min_photons = validate_boxer_parameters(; psf_sigma, min_photons,
            sigma_small=something(sigma_small, 1.0), sigma_large=something(sigma_large, 2.0),
            minval=something(minval, 0.0), boxsize, overlap, backend,
            auto_timeout, gpu_timeout, on_wait)
        detect_in_photons = camera isa SCMOSCamera && psf_sigma !== nothing
        camera isa SCMOSCamera && validate_scmos_calibration(camera;
            require_offset=detect_in_photons)
        # Determine which interface is being used
        if psf_sigma !== nothing
            # PSF-aware detection (recommended)
            # psf_sigma is in physical units (microns) - convert to pixels
            if camera !== nothing
                pixel_size_μm = get_pixel_size(camera)
                psf_sigma_pixels = psf_sigma / pixel_size_μm
            else
                throw(ArgumentError("psf_sigma=$psf_sigma in physical units (microns) requires camera to be provided. " *
                      "Either provide a camera or use the advanced interface with sigma_small/sigma_large in pixels."))
            end

            σ_small = Float32(1.0 * psf_sigma_pixels)
            σ_large = Float32(2.0 * psf_sigma_pixels)
        else
            # Direct control interface
            σ_small = Float32(sigma_small !== nothing ? sigma_small : 1.0)
            σ_large = Float32(sigma_large !== nothing ? sigma_large : 2.0)
            min_val = Float32(minval !== nothing ? minval : 0.0)
        end

        # The filter stores these values in Float32, so conversion must preserve validity.
        for (field, value) in ((:sigma_small, σ_small), (:sigma_large, σ_large))
            isfinite(value) && value > 0 || throw(ArgumentError(
                "$field=$value must remain finite and > 0 in Float32"))
        end
        σ_small < σ_large || throw(ArgumentError(
            "sigma_small=$σ_small must remain smaller than sigma_large=$σ_large in Float32"))
        if psf_sigma !== nothing
            min_val = photons_to_dog_threshold(min_photons, psf_sigma_pixels;
                effective_gain=1.0f0, variance_weighted=camera isa SCMOSCamera)
        end
        isfinite(min_val) || throw(ArgumentError("minval=$min_val must remain finite in Float32"))

        # use_gpu is determined later in _getboxes_impl based on backend and memory availability
        initial_use_gpu = backend != :cpu

        new(imagestack, camera, boxsize, Float32(overlap), σ_small, σ_large, min_val,
            initial_use_gpu, backend, Float64(auto_timeout), Float64(gpu_timeout), on_wait,
            check_pixels, detect_in_photons)
    end
end

"""Reject sCMOS calibration values that make inverse-variance filtering undefined."""
function validate_scmos_calibration(camera::SCMOSCamera; require_offset::Bool=false)
    n_pixels = (length(camera.pixel_edges_y) - 1) * (length(camera.pixel_edges_x) - 1)
    for field in (:readnoise, :gain, :qe)
        value = getproperty(camera, field)
        invalid(x) = !isfinite(x) || x <= 0
        affected = value isa Real ? (invalid(value) ? n_pixels : 0) : count(invalid, value)
        affected == 0 || throw(ArgumentError(
            "$field has $affected affected pixels; values must be finite and positive (> 0). " *
            "A noise-free readout should use an IdealCamera with the image in photons."))
    end
    if require_offset
        offset = camera.offset
        affected = offset isa Real ? (!isfinite(offset) ? n_pixels : 0) : count(!isfinite, offset)
        affected == 0 || throw(ArgumentError(
            "offset has $affected affected pixels; values must be finite for photon conversion"))
    end
    return nothing
end

"""
    get_variance_map(camera::AbstractCamera, imagesize; photons=false)

Compute variance map from camera calibration.

# Arguments
- `camera`: Camera object with noise calibration
- `imagesize`: Tuple of (nrows, ncols) for the image

# Returns
- With `photons=true`, variance in photons²: `(readnoise / qe)^2`.
- Otherwise the direct-path variance in ADU²: `(readnoise / gain)^2`.
  Both support scalar and mapped calibration.
  Image and camera dimensions must already match (checked by `getboxes`).
"""
function get_variance_map(camera::IdealCamera{T}, imagesize::Tuple{Int,Int}; photons::Bool=false) where T
    # IdealCamera has no readnoise, return uniform variance of 1.0
    # Always Float32 to match imagestack type (getboxes converts all images to Float32)
    return ones(Float32, imagesize)
end

function get_variance_map(camera::SCMOSCamera{T}, imagesize::Tuple{Int,Int}; photons::Bool=false) where T
    validate_scmos_calibration(camera)
    # PSF-aware detection uses photons²; direct parameters retain ADU².
    divisor = photons ? camera.qe : camera.gain
    # Square in Float64: low-precision calibration (Float16) can overflow before the Float32 result,
    # e.g. readnoise 20 with qe 0.05 gives 160,000 photons², above Float16's largest value.
    if camera.readnoise isa AbstractArray || divisor isa AbstractArray
        return Float32.((Float64.(camera.readnoise) ./ Float64.(divisor)).^2)
    else
        return fill(Float32((Float64(camera.readnoise) / Float64(divisor))^2), imagesize)
    end
end

"""
    recommend_batch_size(height, width; backend=:auto, memory_fraction=0.8) -> Int

Return recommended maximum number of frames to load at once given memory constraints.

This helps users decide how much data to load before calling `getboxes()`. For very large
datasets, loading data in chunks of this size ensures efficient processing without
running out of memory.

GPU memory is queried via NVML when supported. Otherwise CUDA is queried on each
device, restoring the previous device after each query. Other query errors propagate.

# Arguments
- `height::Int`: Image height in pixels
- `width::Int`: Image width in pixels
- `backend::Symbol`: Compute backend :cpu, :gpu, or :auto (default: :auto)
- `memory_fraction::Real`: Fraction of free memory to use (default: 0.8)

# Returns
- Maximum recommended number of frames to load at once

# Memory Model
This helper uses the standard DoG estimate of 6× the raw image size.
`getboxes` budgets 8× for direct sCMOS and 9× for PSF-aware sCMOS, including an
owned photon copy, and batches the call using that estimate:
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
    (roi_batch, info) = getboxes(imagestack, camera; psf_sigma=0.13)
    roi_batch.frame_indices .+= chunk_start - 1
    # ... process results
end
```
"""
function recommend_batch_size(height::Int, width::Int;
        backend::Symbol=:auto,
        memory_fraction::Real=0.8)
    # Memory multiplier: accounts for all processing stages
    # Matches n_copies in _getboxes_impl for consistency
    n_copies = 6
    bytes_per_frame = height * width * sizeof(Float32) * n_copies

    # Determine if GPU should be used
    use_gpu_actual = backend != :cpu && has_cuda() && CUDA.functional()

    if use_gpu_actual
        # GPU: find device with most free memory, preferring NVML
        max_free_mem = 0
        for i in 0:length(CUDA.devices())-1
            max_free_mem = max(max_free_mem, gpu_free_memory(i))
        end
        available = max_free_mem * memory_fraction
    else
        # CPU: use system free memory
        available = Sys.free_memory() * memory_fraction
    end

    return max(1, floor(Int, available / bytes_per_frame))
end
