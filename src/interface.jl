"""
    getboxes(imagestack, camera=nothing; kwargs...) -> (ROIBatch, BoxesInfo)

Detect particles/blobs in a multidimensional image stack and return
ROI batch with location tracking and processing metadata.

# Arguments
- `imagestack::AbstractArray{<:Real}`: A finite 2D image or 3D (rows, cols, frames) stack.
  Both image dimensions must be >= boxsize. Unsupported dimensions and undersized
  images raise `ArgumentError` before filtering. Zero frames returns an empty ROIBatch
  and BoxesInfo (CPU, zero batches). NaN/Inf raises `ArgumentError` with the count and
  first (row, col, frame), including under `:auto`; integer inputs skip the scan.
- `camera::Union{AbstractCamera,Nothing}`: Optional camera object (IdealCamera or SCMOSCamera) from SMLMData.
  If not provided, an IdealCamera of the image's size with 1 um pixels is created
  (coordinates in um equal pixels).
  The image's (rows, columns) must match the camera's pixel edges, otherwise raises
  `DimensionMismatch`. Crop the camera together with the image.

## Primary Interface (Recommended - PSF-Aware)
- `psf_sigma::Real`: PSF sigma in microns (physical units, e.g., 0.13 for 130nm PSF).
  Automatically converted to pixels using camera pixel size and sets optimal DoG filter parameters.
  Must be finite and > 0. **Requires a camera with uniform square pixels**
  (relative tolerance 1e-4); otherwise use sigma_small/sigma_large in pixels.
- `min_photons::Union{Real,Nothing}`: Positive finite minimum total photons. Defaults to
  `nothing`; requires `psf_sigma`. With `psf_sigma`, omission uses 500.0.
  For sCMOS, each pixel is converted to photons before detection; the threshold
  uses effective gain 1. ROI data is still cut from the supplied ADU image.
  A noise-free, pixel-integrated Gaussian centered on a pixel just reaches the
  threshold at this photon count. See `BoxerConfig` for subpixel-position effects.

## Advanced Interface (Direct Control)
For expert users who want direct control over filter parameters:
Sigmas must be finite with `0 < sigma_small < sigma_large`; `minval` must be finite.

- `sigma_small::Real`: Small Gaussian sigma in pixels (default: 1.0).
- `sigma_large::Real`: Large Gaussian sigma in pixels (default: 2.0).
- `minval::Real`: DoG filter intensity threshold (default: 0.0).

Note: If `psf_sigma` is provided, it overrides sigma_small/sigma_large/minval.

## Other Parameters
`boxsize >= 1`; `overlap` must be finite and >= 0. Timeouts must be >= 0 (Inf allowed).
Invalid parameter values raise `ArgumentError` naming the field and value.

- `boxsize::Int`: ROI size in pixels, odd or even (default: 7).
- `overlap::Real`: pixels (default 2.0). Local maxima are found in a square window of k =
  max(3, ⌊boxsize − overlap⌋) pixels, so two detections in one frame are at least ⌈k/2⌉
  pixels apart along x or y. A larger `overlap` lets detections sit closer together. This is
  not a bound on box overlap: with the defaults, neighboring boxes can share 4 rows or
  columns.
- `backend::Symbol`: Compute backend - `:cpu`, `:gpu`, or `:auto` (default: `:auto`).
  - `:cpu` - Always use CPU
  - `:gpu` - Require GPU, wait for memory if needed (waits forever by default)
  - `:auto` - Try GPU with timeout, fall back to CPU on timeout or other GPU failures
- `auto_timeout::Real`: Max seconds to wait for GPU memory in `:auto` mode (default: 300.0).
- `gpu_timeout::Real`: Max seconds to wait for GPU memory in `:gpu` mode (default: Inf).
- `on_wait::Function`: Optional callback `(elapsed, available, required) -> nothing` for wait progress.

GPU memory queries use NVML when supported. Otherwise CUDA is queried on each device,
restoring the previous device and skipping NVML contention checks for that device.
Other device query errors warn and fall back to CPU in `:auto` mode, or propagate in `:gpu`.

GPU memory errors and an unavailable device are retried until `auto_timeout` under `:auto`,
then processing falls back to the CPU; any other GPU failure uses the CPU at once.
Under `:gpu`, resource errors are retried until `gpu_timeout`; any other error is raised
at once; an interrupt (Ctrl-C) always stops the call.
Resource retries halve the batch cap immediately; after a single-frame failure they
resume device polling. A frame larger than any GPU can free errors under `:gpu` or
uses the CPU immediately under `:auto`.

# Returns
Tuple of `(ROIBatch, BoxesInfo)`:

`ROIBatch` with the following fields:
- `data`: ROI stack (boxsize × boxsize × n_rois) containing image patches
- `x_corners`: Vector of x (column) corner positions in camera coordinates
- `y_corners`: Vector of y (row) corner positions in camera coordinates
- `frame_indices`: Vector of frame indices for each ROI
- `camera`: Provided camera, or an IdealCamera of the image's size with 1 um pixels
  (coordinates in um equal pixels)
- `roi_size`: Size of each ROI (square)

`BoxesInfo` with the following fields:
- `backend`: Compute backend used (:gpu or :cpu)
- `elapsed_s`: Wall time in seconds
- `device_id`: GPU device ID (0-based), or -1 for CPU
- `n_rois`: Number of ROIs detected
- `batch_size`: Frames per batch during processing
- `n_batches`: Number of batches processed
- `memory_per_batch`: Estimated memory per batch in bytes

# Details on filtering

The image stack is convolved with a difference of Gaussians (DoG) filter
to identify blobs and local maxima. The DoG is computed from two Gaussian
kernels with standard deviations `sigma_small` and `sigma_large`.

When using the PSF-aware interface with `psf_sigma` (in microns):
- psf_sigma is converted to pixels using camera pixel size
- sigma_small = 1.0 × psf_sigma_pixels (matches PSF for optimal blob detection)
- sigma_large = 2.0 × psf_sigma_pixels (background suppression)
- minval uses the actual filter kernels' peak response to a pixel-integrated Gaussian
  of min_photons photons, with effective gain 1. The sCMOS reference
  uses uniform variance and its own Gaussian windows, so thresholds can differ by path.

## Variance-Weighted Filtering (sCMOS)

When an SCMOSCamera is provided, the package uses **variance-weighted filtering** based on the
SMITE algorithm. Every readnoise, gain and qe value must be finite and > 0, including
maps; otherwise `ArgumentError` names the field and affected pixel count. A noise-free
readout should use an IdealCamera with the image in photons.
With an `SCMOSCamera`, the PSF-aware path (`psf_sigma`/`min_photons`) detects in
photons per pixel: `(adu - offset) * gain / qe`, with variance `(readnoise / qe)^2`
in photons². Offset, gain and QE may each be scalar or maps; offset must be finite
when this conversion is used. The threshold uses effective gain 1 and the
variance-weighted reference response to `min_photons` photons (uniform variance).
Conversion runs on an owned batch copy on its device. ROI pixels are still cut
from the caller's ADU image and retain the supplied camera and coordinate convention.
The direct interface (`sigma_small`, `sigma_large`, `minval`) filters the image as
given with `(readnoise / gain)^2` in ADU²; `minval` stays in those image units.

Each pixel's contribution to the convolution is weighted by:

    weight = gaussian_kernel / variance

For PSF-aware detection, variance = (readnoise / qe)² in photons².
For direct parameters, variance = (readnoise / gain)² in ADU²
(gain in e⁻/ADU), including maps. This implements inverse variance weighting:
- Low-noise pixels receive high weight (strong influence on detection)
- High-noise pixels receive low weight (reduced influence, avoiding false positives)

This significantly improves detection sensitivity in sCMOS data with spatially-varying noise.

**GPU Acceleration:** Variance-weighted filtering uses KernelAbstractions.jl for device-agnostic
computation. The same kernel code runs on both CPU and GPU, automatically selected based on `backend`.
This provides GPU acceleration for sCMOS cameras (10-100x speedup on large images).

## Standard Filtering (IdealCamera or no camera)

Standard DoG convolution is used when no camera is provided or with IdealCamera.
The convolution runs as a KernelAbstractions kernel on the GPU, or via NNlib on the CPU,
depending on `backend`. At image edges, each Gaussian is normalized by its in-bounds
weights, as in the sCMOS path. Interior pixels retain the original convolution values.

After filtering, local maxima above `minval` are identified. Maxima sharing a
max-pool window have equal intensities; keep the earlier detection.
Boxes use `clamp(center - (boxsize - 1) ÷ 2, 1, image_size - boxsize + 1)` on each axis.
After clamping, only the brightest filtered detection per corner and frame is kept.

# Examples
```julia
# Recommended: PSF-aware detection with physical units
camera = IdealCamera(1:256, 1:256, 0.1f0)  # 256×256 pixels, 100nm pixel size

(roi_batch, info) = getboxes(imagestack, camera;
    psf_sigma = 0.13,              # PSF sigma in microns (physical units)
    min_photons = 500.0,           # Detect emitters with ≥500 photons
    boxsize = 11)

# Access results
boxes = roi_batch.data             # (11 × 11 × n_rois)
x_corners = roi_batch.x_corners    # x (col) positions
y_corners = roi_batch.y_corners    # y (row) positions
frames = roi_batch.frame_indices

# Check processing info
println("Backend: ", info.backend)
println("Elapsed: ", info.elapsed_s * 1000, " ms")

# Advanced: Direct control over filter parameters
(roi_batch, info) = getboxes(imagestack;
    sigma_small = 1.5,  # Custom small Gaussian sigma
    sigma_large = 3.0,  # Custom large Gaussian sigma
    minval = 10.0)      # Custom intensity threshold

# Iterate over ROIs
for roi in roi_batch
    # roi is a SingleROI with .data, .corner, .frame_idx
    process(roi.data)
end
```
"""
function getboxes(imagestack::AbstractArray{<:Real}, camera::Union{AbstractCamera,Nothing}, config::BoxerConfig)
  ndims(imagestack) in (2, 3) || throw(ArgumentError(
      "imagestack has $(ndims(imagestack)) dimensions; expected a 2D image or 3D (rows, cols, frames) stack"))
  for dim in 1:2
    size(imagestack, dim) >= config.boxsize || throw(ArgumentError(
        "image dimension $dim has size $(size(imagestack, dim)), smaller than boxsize=$(config.boxsize)"))
  end
  if camera !== nothing
    image_size = (size(imagestack, 1), size(imagestack, 2))
    camera_size = (length(camera.pixel_edges_y) - 1, length(camera.pixel_edges_x) - 1)
    image_size == camera_size || throw(DimensionMismatch(
        "Image size $image_size does not match camera size $camera_size (rows, columns)"))
  end

  # Box extraction needs a host copy anyway; keep caller-owned GPU storage untouched
  # and avoid allocating GPU conversion/reshape aliases outside batch cleanup.
  check_pixels = !(eltype(imagestack) <: Integer)
  if imagestack isa CuArray
    check_pixels && validate_finite_pixels(imagestack)
    # Float64 values can still overflow during the subsequent Float32 conversion.
    check_pixels = check_pixels && !(eltype(imagestack) <: Union{Float16,Float32})
    imagestack = Array(imagestack)
  end
  # Convert to Float32 for type stability throughout pipeline
  imagestack_f32 = imagestack isa AbstractArray{Float32} ? imagestack : Float32.(imagestack)
  # A finite wider value can overflow in that conversion; name that cause, not "non-finite".
  check_pixels && !(eltype(imagestack) <: Union{Float16,Float32}) &&
    !pixels_are_finite(imagestack_f32) && validate_float32_range(imagestack)

  # Create args from config
  args = GetBoxesArgs(;
      imagestack=imagestack_f32,
      camera=camera,
      boxsize=config.boxsize,
      overlap=config.overlap,
      psf_sigma=config.psf_sigma,
      min_photons=config.min_photons,
      sigma_small=config.psf_sigma === nothing ? config.sigma_small : nothing,
      sigma_large=config.psf_sigma === nothing ? config.sigma_large : nothing,
      minval=config.psf_sigma === nothing ? config.minval : nothing,
      backend=config.backend,
      auto_timeout=config.auto_timeout,
      gpu_timeout=config.gpu_timeout,
      on_wait=config.on_wait,
      check_pixels
  )
  return _getboxes_impl(args)
end

# Kwargs calling convention (forwards to Config form)
function getboxes(imagestack::AbstractArray{<:Real}, camera::Union{AbstractCamera,Nothing}=nothing;
                  psf_sigma::Union{Real,Nothing}=nothing,
                  min_photons::Union{Real,Nothing}=nothing,
                  sigma_small::Union{Real,Nothing}=nothing,
                  sigma_large::Union{Real,Nothing}=nothing,
                  minval::Union{Real,Nothing}=nothing,
                  boxsize::Int=7,
                  overlap::Real=2.0,
                  backend::Symbol=:auto,
                  auto_timeout::Real=300.0,
                  gpu_timeout::Real=Inf,
                  on_wait=nothing)
  # Build config from kwargs
  config = BoxerConfig(
      psf_sigma=psf_sigma === nothing ? nothing : Float64(psf_sigma),
      min_photons=min_photons === nothing ? nothing : Float64(min_photons),
      sigma_small=Float64(sigma_small === nothing ? 1.0 : sigma_small),
      sigma_large=Float64(sigma_large === nothing ? 2.0 : sigma_large),
      minval=Float64(minval === nothing ? 0.0 : minval),
      boxsize=boxsize,
      overlap=Float64(overlap),
      backend=backend,
      auto_timeout=Float64(auto_timeout),
      gpu_timeout=Float64(gpu_timeout),
      on_wait=on_wait
  )

  return getboxes(imagestack, camera, config)
end

"""Scan host pixels without a mask; CuArray dispatch uses an explicitly freed device flag."""
pixels_are_finite(image) = mapreduce(isfinite, &, image; init=true)

"""Report all non-finite pixels and the first (row, column, frame), in storage order."""
function validate_finite_pixels(image)
  eltype(image) <: Integer && return nothing
  pixels_are_finite(image) && return nothing
  # Diagnostics are only needed on error. A host copy avoids GC-managed count and
  # findfirst reduction buffers when the caller supplied a CuArray directly.
  diagnostics = image isa CuArray ? Array(image) : image
  affected = count(!isfinite, diagnostics)
  index = Tuple(findfirst(!isfinite, diagnostics))
  row, col = index[1:2]
  frame = length(index) == 2 ? 1 : index[end]
  throw(ArgumentError("imagestack has $affected non-finite pixels; first (row, col, frame) = $((row, col, frame))"))
end

"""Report finite pixels that are outside Float32's range, which detection converts to."""
function validate_float32_range(image)
  outside(x) = isfinite(x) && !isfinite(Float32(x))
  affected = count(outside, image)
  affected == 0 && return nothing
  index = Tuple(findfirst(outside, image))
  row, col = index[1:2]
  frame = length(index) == 2 ? 1 : index[end]
  throw(ArgumentError("imagestack has $affected finite pixels outside Float32's range " *
    "(magnitude above $(floatmax(Float32))), which detection converts to; first (row, col, frame) = " *
    "$((row, col, frame))"))
end

"""
    frames_per_batch(nframes, memory_per_frame, max_free_mem, max_batch_frames) -> Int

Choose a batch size from available memory and the frame cap retained across GPU retries.
Always attempt at least one frame.
"""
function frames_per_batch(nframes, memory_per_frame, max_free_mem, max_batch_frames)
  memory_cap = max(1, floor(Int, max_free_mem / memory_per_frame))
  return min(nframes, memory_cap, max_batch_frames)
end

"""
    _process_with_batching(imagestack, args, kernelsize, max_free_mem; use_gpu,
                           max_batch_frames=size(imagestack, 4), batch_cleanup=nothing)

Process imagestack with memory-aware batching. Handles both single-batch (fits in memory)
and multi-batch (too large) cases.

Returns `(coords, batch_size, n_batches, memory_per_batch)`. If an input pixel is
non-finite, `coords` is the marker `nothing`: all owned temporaries are freed normally,
and the caller reports the input error after releasing device memory.

# Arguments
- `imagestack`: 4D image stack (ny, nx, 1, nframes)
- `args`: GetBoxesArgs with filter parameters
- `kernelsize`: Kernel size for local max detection
- `max_free_mem`: Available memory in bytes
- `use_gpu`: Whether to use GPU for processing
- `max_batch_frames`: Optional cap on frames per batch, retained across GPU retries
- `batch_cleanup`: Optional function called after each batch (e.g., for GC)
"""
function _process_with_batching(imagestack, args, kernelsize, max_free_mem;
    use_gpu::Bool, max_batch_frames::Int=size(imagestack, 4),
    batch_cleanup::Union{Function,Nothing}=nothing)
  n_images = size(imagestack, 4)
  n_images == 0 && return Matrix{Float32}[], 0, 0, 0
  memory_per_frame = estimate_gpu_memory_per_frame(
    size(imagestack, 1), size(imagestack, 2), args.camera;
    detect_in_photons=args.detect_in_photons)
  batch_size = frames_per_batch(n_images, memory_per_frame, max_free_mem, max_batch_frames)
  n_batches = cld(n_images, batch_size)
  memory_per_batch = batch_size * memory_per_frame

  coords = Vector{Matrix{Float32}}(undef, 0)
  for i in 1:n_batches
    start_idx = (i - 1) * batch_size + 1
    end_idx = min(i * batch_size, n_images)
    batch = filtered_batch = nothing
    local coords_batch
    completed = false
    try
      # Upload once, then check the same storage the GPU filter will read. Return
      # invalid input normally so its ArgumentError never crosses the GPU catch.
      if use_gpu
        source = n_batches == 1 ? imagestack : @view(imagestack[:, :, :, start_idx:end_idx])
        batch = source isa CuArray ? copy(source) : CuArray(source)
      else
        batch = n_batches == 1 ? imagestack : imagestack[:, :, :, start_idx:end_idx]
      end
      if args.check_pixels && !pixels_are_finite(batch)
        completed = true
        return nothing, batch_size, n_batches, memory_per_batch
      end
      filtered_batch = dog_filter(batch, args; use_gpu,
          owned_input=use_gpu || n_batches != 1)
      coords_batch = findlocalmax(filtered_batch, kernelsize; minval=args.minval, use_gpu)
      completed = true
    finally
      free_gpu_arrays(filtered_batch, use_gpu || n_batches != 1 ? batch : nothing;
        failed=!completed)
    end

    if n_batches == 1
      coords = coords_batch
    else
      # Offset frame indices to actual frame numbers.
      frame_offset = start_idx - 1
      for coord_matrix in coords_batch
        coord_matrix[:, 3] .+= frame_offset
      end
      append!(coords, coords_batch)
    end
    if n_batches > 1 && batch_cleanup !== nothing
      filtered_batch = nothing
      batch = nothing
      batch_cleanup()
    end
  end
  return coords, batch_size, n_batches, memory_per_batch
end

"""
    _getboxes_impl(args::GetBoxesArgs)

Internal implementation of getboxes that does the actual work.
"""
function _getboxes_impl(args::GetBoxesArgs)
  start_ns = time_ns()

  imagestack = reshape_for_flux(args.imagestack)

  minkernelsize = 3
  kernelsize = floor(Int, max(minkernelsize, args.boxsize - args.overlap))

  # Estimate memory needed for at least 1 frame (minimum batch)
  nrows, ncols = size(imagestack, 1), size(imagestack, 2)
  min_memory_needed = estimate_gpu_memory_per_frame(nrows, ncols, args.camera;
      detect_in_photons=args.detect_in_photons)

  # Track results
  actual_backend = :cpu
  device_id = -1
  batch_size = 0
  n_batches = 0
  memory_per_batch = 0
  gpu_succeeded = false

  empty_stack = size(imagestack, 4) == 0
  coords = Matrix{Float32}[]
  if !empty_stack && args.backend != :cpu && has_cuda()
    timeout = args.backend == :auto ? args.auto_timeout : args.gpu_timeout
    deadline = time() + timeout
    gpu_fallback = false
    batch_cap = size(imagestack, 4)
    failed_batch_size = 1
    poll_device = true
    dev_id = -1

    totals = try
      map(CUDA.totalmem, CUDA.devices())
    catch e
      (args.backend == :gpu || !is_gpu_query_error(e)) && rethrow()
      @warn "GPU device query failed, using CPU" exception=(e, catch_backtrace())
      gpu_fallback = true
      Int[]
    end
    frame_need = ceil(Int, 1.5 * min_memory_needed)
    if !gpu_fallback && !frame_fits_some_gpu(frame_need, totals, GPU_CONTEXT_RESERVE)
      largest = isempty(totals) ? 0 : maximum(totals)
      largest_free = largest >= GPU_CONTEXT_RESERVE ? largest - GPU_CONTEXT_RESERVE : 0
      message = "No visible GPU can free enough memory for one frame: need " *
        "$(Base.format_bytes(frame_need)) including the safety margin; " *
        "largest device can free $(Base.format_bytes(largest_free)) " *
        "(after a $(Base.format_bytes(GPU_CONTEXT_RESERVE)) context/driver reserve)"
      args.backend == :gpu && error(message)
      @warn "$message; using the CPU"
      gpu_fallback = true
    end

    while !gpu_succeeded && !gpu_fallback && time() < deadline
      if poll_device
        remaining = max(0.0, deadline - time())
        available, dev_id = try
          poll_gpu_nvml(min_memory_needed; timeout=remaining, on_wait=args.on_wait)
        catch e
          (args.backend == :gpu || !is_gpu_query_error(e)) && rethrow()
          @warn "GPU device query failed, using CPU" exception=(e, catch_backtrace())
          gpu_fallback = true
          break
        end
        !available && break
      end

      attempt_batch_size = 1
      try
        CUDA.device!(dev_id)
        max_free_mem = CUDA.free_memory()
        attempt_batch_size = frames_per_batch(size(imagestack, 4), min_memory_needed,
          max_free_mem, batch_cap)
        coords, batch_size, n_batches, memory_per_batch = _process_with_batching(
          imagestack, args, kernelsize, max_free_mem;
          use_gpu=true, max_batch_frames=batch_cap)
        release_gpu_memory()
        device_id = dev_id
        actual_backend = :gpu
        gpu_succeeded = true
      catch e
        reclaim_failed_gpu_attempt()
        e isa InterruptException && rethrow()
        if !is_gpu_resource_error(e)
          args.backend == :gpu && rethrow()
          @warn "GPU processing failed; using the CPU" exception=(e, catch_backtrace())
          gpu_fallback = true
          break
        end
        failed_batch_size = attempt_batch_size
        @warn "GPU failed, releasing memory and retrying" exception=e
        if attempt_batch_size > 1
          batch_cap = max(1, attempt_batch_size ÷ 2)
          poll_device = false
        else
          poll_device = true
        end
      end
    end

    if !gpu_succeeded
      if args.backend == :gpu
        failed_memory = failed_batch_size * min_memory_needed
        error("GPU processing failed after $(timeout)s. " *
          "Failed batch: $failed_batch_size frame(s), " *
          "requiring $(Base.format_bytes(failed_memory))")
      elseif !gpu_fallback
        @warn "GPU unavailable after $(timeout)s, using CPU"
      end
    end
  elseif !empty_stack && args.backend == :gpu
      args.check_pixels && validate_finite_pixels(args.imagestack)
      error("GPU backend requested but CUDA is not functional")
  end

  if !empty_stack && !gpu_succeeded
      # CPU path
      max_free_mem = Sys.free_memory()
      coords, batch_size, n_batches, memory_per_batch = _process_with_batching(
          imagestack, args, kernelsize, max_free_mem;
          use_gpu=false,
          batch_cleanup=() -> GC.gc(false))
  end

  # A pixel-check marker takes the normal cleanup path above. Reporting it here
  # cannot be confused with a GPU fault or trigger the :auto CPU fallback.
  coords === nothing && validate_finite_pixels(args.imagestack)

  maxcoords = empty_stack ? zeros(Float32, 0, 4) : removeoverlap(coords, args)

  boxstack, boxcoords = getboxstack(imagestack, maxcoords, args)

  # Create ROIBatch
  # Convert boxcoords (row, col, frame) to separate x_corners, y_corners vectors
  n_rois = size(boxstack, 3)
  x_corners = Vector{Int32}(undef, n_rois)
  y_corners = Vector{Int32}(undef, n_rois)
  frame_indices = Vector{Int32}(undef, n_rois)

  for i in 1:n_rois
    x_corners[i] = Int32(boxcoords[i, 2])  # x = col
    y_corners[i] = Int32(boxcoords[i, 1])  # y = row
    frame_indices[i] = Int32(boxcoords[i, 3])
  end

  # Use provided camera or create an image-sized IdealCamera with 1 um pixels
  camera = if args.camera !== nothing
    args.camera
  else
    # The three-argument constructor takes pixel centers
    img_rows, img_cols = size(imagestack, 1), size(imagestack, 2)
    IdealCamera(
      1:img_cols,  # pixel centers x
      1:img_rows,  # pixel centers y
      1.0f0        # pixel size (um)
    )
  end

  roi_batch = ROIBatch(boxstack, x_corners, y_corners, frame_indices, camera)
  elapsed_s = (time_ns() - start_ns) / 1e9
  info = BoxesInfo(actual_backend, elapsed_s, device_id, n_rois, batch_size, n_batches, memory_per_batch)

  return (roi_batch, info)
end
