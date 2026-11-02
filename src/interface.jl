"""
    getboxes(imagestack, camera=nothing; kwargs...) -> (ROIBatch, BoxesInfo)

Detect particles/blobs in a multidimensional image stack and return
ROI batch with location tracking and processing metadata.

# Arguments
- `imagestack::AbstractArray{<:Real}`: The input image stack. Should be 2D or 3D.
- `camera::Union{AbstractCamera,Nothing}`: Optional camera object (IdealCamera or SCMOSCamera) from SMLMData.
  If not provided, an IdealCamera of the image's size with 1 um pixels is created
  (coordinates in um equal pixels).
  The image's (rows, columns) must match the camera's pixel edges, otherwise raises
  `DimensionMismatch`. Crop the camera together with the image.

## Primary Interface (Recommended - PSF-Aware)
- `psf_sigma::Real`: PSF sigma in microns (physical units, e.g., 0.13 for 130nm PSF).
  Automatically converted to pixels using camera pixel size and sets optimal DoG filter parameters.
  **Requires camera to be provided.**
- `min_photons::Real`: Minimum total photons for detection (default: 500.0).
  A noise-free, pixel-integrated Gaussian centered on a pixel just reaches the
  threshold at this photon count; centered on a pixel corner it needs about 1.1× more.

## Advanced Interface (Direct Control)
For expert users who want direct control over filter parameters:
- `sigma_small::Real`: Small Gaussian sigma in pixels (default: 1.0).
- `sigma_large::Real`: Large Gaussian sigma in pixels (default: 2.0).
- `minval::Real`: DoG filter intensity threshold (default: 0.0).

Note: If `psf_sigma` is provided, it overrides sigma_small/sigma_large/minval.

## Other Parameters
- `boxsize::Int`: ROI size in pixels, odd or even, at least 1 (default: 7).
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
  of min_photons photons, times effective gain (QE / gain for sCMOS). The sCMOS reference
  uses uniform variance and its own Gaussian windows, so thresholds can differ by path.

## Variance-Weighted Filtering (sCMOS)

When an SCMOSCamera is provided, the package uses **variance-weighted filtering** based on the
SMITE algorithm. Each pixel's contribution to the convolution is weighted by:

    weight = gaussian_kernel / variance

where variance = (readnoise / gain)² in ADU², per pixel when either calibration is a map
(gain in e⁻/ADU). This implements inverse variance weighting:
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

After filtering, local maxima above `minval` are identified. Maxima within the same
max-pool window compete by filtered brightness; ties keep the earlier detection.
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
  if camera !== nothing
    image_size = (size(imagestack, 1), size(imagestack, 2))
    camera_size = (length(camera.pixel_edges_y) - 1, length(camera.pixel_edges_x) - 1)
    image_size == camera_size || throw(DimensionMismatch(
        "Image size $image_size does not match camera size $camera_size (rows, columns)"))
  end

  # Convert to Float32 for type stability throughout pipeline
  imagestack_f32 = imagestack isa AbstractArray{Float32} ? imagestack : Float32.(imagestack)

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
      on_wait=config.on_wait
  )
  return _getboxes_impl(args)
end

# Kwargs calling convention (forwards to Config form)
function getboxes(imagestack::AbstractArray{<:Real}, camera::Union{AbstractCamera,Nothing}=nothing;
                  psf_sigma::Union{Real,Nothing}=nothing,
                  min_photons::Real=500.0,
                  sigma_small::Union{Real,Nothing}=nothing,
                  sigma_large::Union{Real,Nothing}=nothing,
                  minval::Union{Real,Nothing}=nothing,
                  boxsize::Int=7,
                  overlap::Real=2.0,
                  backend::Symbol=:auto,
                  auto_timeout::Real=300.0,
                  gpu_timeout::Real=Inf,
                  on_wait::Union{Function,Nothing}=nothing)
  # Build config from kwargs
  config = BoxerConfig(
      psf_sigma=psf_sigma === nothing ? nothing : Float64(psf_sigma),
      min_photons=Float64(min_photons),
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

Returns `(coords, batch_size, n_batches, memory_per_batch)`.

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
  memory_per_frame = estimate_gpu_memory_per_frame(
    size(imagestack, 1), size(imagestack, 2), args.camera)
  batch_size = frames_per_batch(n_images, memory_per_frame, max_free_mem, max_batch_frames)
  n_batches = cld(n_images, batch_size)
  memory_per_batch = batch_size * memory_per_frame

  if n_batches == 1
    filtered_stack = dog_filter(imagestack, args; use_gpu)
    coords = findlocalmax(filtered_stack, kernelsize; minval=args.minval, use_gpu)
  else
    coords = Vector{Matrix{Float32}}(undef, 0)
    for i in 1:n_batches
      start_idx = (i - 1) * batch_size + 1
      end_idx = min(i * batch_size, n_images)
      batch = imagestack[:, :, :, start_idx:end_idx]
      filtered_batch = dog_filter(batch, args; use_gpu)
      coords_batch = findlocalmax(filtered_batch, kernelsize; minval=args.minval, use_gpu)

      # Offset frame indices to actual frame numbers.
      frame_offset = start_idx - 1
      for coord_matrix in coords_batch
        coord_matrix[:, 3] .+= frame_offset
      end
      append!(coords, coords_batch)

      if batch_cleanup !== nothing
        filtered_batch = nothing
        batch = nothing
        batch_cleanup()
      end
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
  kernelsize = Int(floor(args.boxsize - args.overlap))
  kernelsize = max(minkernelsize, kernelsize)

  # Estimate memory needed for at least 1 frame (minimum batch)
  nrows, ncols = size(imagestack, 1), size(imagestack, 2)
  min_memory_needed = estimate_gpu_memory_per_frame(nrows, ncols, args.camera)

  # Track results
  actual_backend = :cpu
  device_id = -1
  batch_size = 0
  n_batches = 0
  memory_per_batch = 0
  gpu_succeeded = false

  if args.backend != :cpu && has_cuda()
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
        CUDA.synchronize()
        GC.gc(false)
        CUDA.reclaim()
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
  elseif args.backend == :gpu
      error("GPU backend requested but CUDA is not functional")
  end

  if !gpu_succeeded
      # CPU path
      max_free_mem = Sys.free_memory()
      coords, batch_size, n_batches, memory_per_batch = _process_with_batching(
          imagestack, args, kernelsize, max_free_mem;
          use_gpu=false,
          batch_cleanup=() -> GC.gc(false))
  end

  maxcoords = removeoverlap(coords, args)

  # Ensure imagestack is on CPU for box extraction (uses scalar indexing)
  imagestack_cpu = imagestack isa CuArray ? Array(imagestack) : imagestack
  boxstack, boxcoords = getboxstack(imagestack_cpu, maxcoords, args)

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
