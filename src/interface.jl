"""
    getboxes(imagestack, camera=nothing; kwargs...)

Detect particles/blobs in a multidimensional image stack and return
coordinates and boxed regions centered around local maxima.

# Arguments
- `imagestack::AbstractArray{<:Real}`: The input image stack. Should be 2D or 3D.
- `camera::Union{AbstractCamera,Nothing}`: Optional camera object (IdealCamera or SCMOSCamera) from SMLMData.
  If provided, enables micron coordinate conversion and camera ROI extraction.
- `boxsize::Int`: Size of the box to cut out around each local maximum (pixels).
- `overlap::Real`: Amount of overlap allowed between boxes (pixels).
- `sigma_small::Real`: Sigma for small Gaussian blur kernel (pixels).
- `sigma_large::Real`: Sigma for large Gaussian blur kernel (pixels).
- `minval::Real`: Minimum value to consider as a local maximum.
- `use_gpu::Bool`: Perform convolution and local max finding on GPU.

# Returns
A NamedTuple with the following fields:
- `boxes`: Array with dimensions (boxsize, boxsize, nboxes) containing image patches
- `coords_pixels`: N×3 matrix of detection centers (row, col, frame) in pixels
- `coords_microns`: N×2 matrix of detection centers (x, y) in microns (only if camera provided)
- `boxcoords`: N×3 matrix of box upper-left corners (row, col, frame) in pixels
- `camera_rois`: Vector of camera ROIs for each box (only if camera provided)
- `metadata`: Additional information (number of detections, etc.)

# Details on filtering

The image stack is convolved with a difference of Gaussians (DoG) filter
to identify blobs and local maxima. The DoG is computed from two Gaussian
kernels with standard deviations `sigma_small` and `sigma_large`.

The convolution is performed either on CPU or GPU, depending on `use_gpu`.
After filtering, local maxima above `minval` are identified. Boxes are cut
out around each maximum, excluding overlaps.

# Examples
```julia
# Without camera (backward compatible)
result = getboxes(imagestack; boxsize=7, overlap=2.0, sigma_small=1.0, sigma_large=2.0)
boxes = result.boxes
coords = result.coords_pixels

# With camera (enables micron coordinates and camera ROIs)
camera = IdealCamera(pixel_edges_x=0:0.1:25.6, pixel_edges_y=0:0.1:25.6)
result = getboxes(imagestack, camera; boxsize=7, overlap=2.0)
boxes = result.boxes
coords_microns = result.coords_microns
camera_rois = result.camera_rois
```
"""
function getboxes(imagestack::AbstractArray{<:Real}, camera::Union{AbstractCamera,Nothing}=nothing; kwargs...)
  # Create args with camera
  args = GetBoxesArgs(; imagestack=imagestack, camera=camera, kwargs...)
  return _getboxes_impl(args)
end

# Legacy keyword-only interface for backward compatibility
function getboxes(; kwargs...)
  args = GetBoxesArgs(; kwargs...)
  return _getboxes_impl(args)
end

"""
    _getboxes_impl(args::GetBoxesArgs)

Internal implementation of getboxes that does the actual work.
"""
function _getboxes_impl(args::GetBoxesArgs)

  imagestack = reshape_for_flux(args.imagestack)

  minkernelsize = 3
  kernelsize = Int(floor(args.boxsize - args.overlap))
  kernelsize = max(minkernelsize, kernelsize)

  # Determine whether to perform calculations on the GPU
  args.use_gpu = args.use_gpu && has_cuda() && CUDA.functional()

  if args.use_gpu
      # Find the device with the most free memory
      max_free_mem = 0
      best_device = 0
      for i in 0:length(CUDA.devices())-1
          CUDA.device!(i)
          free_mem = CUDA.available_memory()
          if free_mem > max_free_mem
              max_free_mem = free_mem
              best_device = i
          end
      end
      # Switch to the best device
      CUDA.device!(best_device)

      # Check the size of the image stack
      n_copies = 3 # 3x for the filtered stack, coords, and boxstack
      memory_required = sizeof(imagestack) * n_copies

      if memory_required <= max_free_mem
          # If the image stack fits in memory, perform the operation on the whole stack
          filtered_stack = dog_filter(imagestack |> gpu, args)
          coords = findlocalmax(filtered_stack, kernelsize; minval=args.minval, use_gpu=args.use_gpu)
      else
          # If the image stack is too big, split it into smaller batches and process each batch separately
          memory_required_per_frame = size(imagestack, 1)*size(imagestack, 2) * sizeof(eltype(imagestack)) * n_copies
          # println("Memory required per frame: ", memory_required_per_frame / 1024^3, " GB\n")
          batch_size = Int(floor(max_free_mem / memory_required_per_frame))

          n_images = size(imagestack, 4)
          n_batches = Int(ceil(n_images / batch_size))
          
          coords = Vector{Matrix{Float32}}(undef, 0)

          for i in 1:n_batches
              start_idx = (i-1)*batch_size + 1
              end_idx = min(i*batch_size, n_images)
              batch = imagestack[:, :,:, start_idx:end_idx] |> gpu
              filtered_batch = dog_filter(batch, args)
              coords_batch = findlocalmax(filtered_batch, kernelsize; minval=args.minval, use_gpu=args.use_gpu)    
              append!(coords, coords_batch)
          end
      end

      CUDA.synchronize()
  else
      filtered_stack = dog_filter(imagestack, args)
      coords = findlocalmax(filtered_stack, kernelsize; minval=args.minval, use_gpu=args.use_gpu)
  end
  
  maxcoords = removeoverlap(coords, args)
  boxstack, boxcoords, camera_rois = getboxstack(imagestack, maxcoords, args)

  # Convert coordinates to microns if camera is provided
  coords_microns = if args.camera !== nothing
    pixels_to_microns(maxcoords[:, 1:2], args.camera)
  else
    nothing
  end

  # Build metadata
  ndetections = size(maxcoords, 1)
  metadata = (
    ndetections = ndetections,
    boxsize = args.boxsize,
    has_camera = args.camera !== nothing
  )

  # Return as NamedTuple
  return (
    boxes = boxstack,
    coords_pixels = maxcoords,
    coords_microns = coords_microns,
    boxcoords = boxcoords,
    camera_rois = camera_rois,
    metadata = metadata
  )
end




