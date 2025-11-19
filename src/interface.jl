"""
    getboxes(imagestack, camera=nothing; kwargs...) -> ROIBatch

Detect particles/blobs in a multidimensional image stack and return
ROI batch with location tracking.

# Arguments
- `imagestack::AbstractArray{<:Real}`: The input image stack. Should be 2D or 3D.
- `camera::Union{AbstractCamera,Nothing}`: Optional camera object (IdealCamera or SCMOSCamera) from SMLMData.
  If not provided, a default IdealCamera is created.
- `boxsize::Int`: Size of the box to cut out around each local maximum (pixels).
- `overlap::Real`: Amount of overlap allowed between boxes (pixels).
- `sigma_small::Real`: Sigma for small Gaussian blur kernel (pixels).
- `sigma_large::Real`: Sigma for large Gaussian blur kernel (pixels).
- `minval::Real`: Minimum value to consider as a local maximum.
- `use_gpu::Bool`: Perform convolution and local max finding on GPU.

# Returns
`ROIBatch` with the following fields:
- `data`: ROI stack (boxsize × boxsize × n_rois) containing image patches
- `corners`: (2 × n_rois) matrix of (x,y) = (col,row) corner positions in camera coordinates
- `frame_indices`: Vector of frame indices for each ROI
- `camera`: Camera object (provided or default IdealCamera)
- `roi_size`: Size of each ROI (square)

# Details on filtering

The image stack is convolved with a difference of Gaussians (DoG) filter
to identify blobs and local maxima. The DoG is computed from two Gaussian
kernels with standard deviations `sigma_small` and `sigma_large`.

## Variance-Weighted Filtering (sCMOS)

When an SCMOSCamera is provided, the package uses **variance-weighted filtering** based on the
SMITE algorithm. Each pixel's contribution to the convolution is weighted by:

    weight = gaussian_kernel / variance

where variance = readnoise². This implements optimal inverse variance weighting:
- Low-noise pixels receive high weight (strong influence on detection)
- High-noise pixels receive low weight (reduced influence, avoiding false positives)

This significantly improves detection sensitivity in sCMOS data with spatially-varying noise.

**GPU Acceleration:** Variance-weighted filtering uses KernelAbstractions.jl for device-agnostic
computation. The same kernel code runs on both CPU and GPU, automatically selected based on `use_gpu`.
This provides GPU acceleration for sCMOS cameras (10-100x speedup on large images).

## Standard Filtering (IdealCamera or no camera)

Standard DoG convolution is used when no camera is provided or with IdealCamera.
The convolution is performed via NNlib (using cuDNN on GPU) or CPU, depending on `use_gpu`.

After filtering, local maxima above `minval` are identified. Boxes are cut
out around each maximum, excluding overlaps.

# Examples
```julia
# Basic usage
roi_batch = getboxes(imagestack; boxsize=7, overlap=2.0, sigma_small=1.0, sigma_large=2.0)
boxes = roi_batch.data  # (7 × 7 × n_rois)
corners = roi_batch.corners  # (2 × n_rois) [x;y] = [col;row]
frames = roi_batch.frame_indices

# With camera for proper coordinate system
camera = IdealCamera(1:256, 1:256, 0.1f0)  # npixels_x, npixels_y, pixel_size
roi_batch = getboxes(imagestack, camera; boxsize=7, overlap=2.0)

# Iterate over ROIs
for roi in roi_batch
    # roi is a SingleROI with .data, .corner, .frame_idx
    process(roi.data)
end
```
"""
function getboxes(imagestack::AbstractArray{<:Real}, camera::Union{AbstractCamera,Nothing}=nothing; kwargs...)
  # Convert to Float32 for type stability throughout pipeline
  imagestack_f32 = imagestack isa AbstractArray{Float32} ? imagestack : Float32.(imagestack)

  # Create args with camera
  args = GetBoxesArgs(; imagestack=imagestack_f32, camera=camera, kwargs...)
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
          filtered_stack = dog_filter(imagestack, args)
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
              batch = imagestack[:, :,:, start_idx:end_idx]
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

  # Ensure imagestack is on CPU for box extraction (uses scalar indexing)
  imagestack_cpu = imagestack isa CuArray ? Array(imagestack) : imagestack
  boxstack, boxcoords, camera_rois = getboxstack(imagestack_cpu, maxcoords, args)

  # Create ROIBatch
  # Convert boxcoords (row, col, frame) to corners (x, y) = (col, row) format
  n_rois = size(boxstack, 3)
  corners = Matrix{Int32}(undef, 2, n_rois)
  frame_indices = Vector{Int32}(undef, n_rois)

  for i in 1:n_rois
    corners[1, i] = Int32(boxcoords[i, 2])  # x = col
    corners[2, i] = Int32(boxcoords[i, 1])  # y = row
    frame_indices[i] = Int32(boxcoords[i, 3])
  end

  # Use provided camera or create default IdealCamera if none provided
  camera = if args.camera !== nothing
    args.camera
  else
    # Create minimal IdealCamera with pixel edges matching image size
    img_rows, img_cols = size(imagestack, 1), size(imagestack, 2)
    IdealCamera(
      1:(img_cols+1),  # pixel_edges_x
      1:(img_rows+1),  # pixel_edges_y
      1.0f0            # pixel size (arbitrary for default)
    )
  end

  return ROIBatch(boxstack, corners, frame_indices, camera)
end




