"""
    getboxes(imagestack, camera=nothing; kwargs...) -> ROIBatch

Detect particles/blobs in a multidimensional image stack and return
ROI batch with location tracking.

# Arguments
- `imagestack::AbstractArray{<:Real}`: The input image stack. Should be 2D or 3D.
- `camera::Union{AbstractCamera,Nothing}`: Optional camera object (IdealCamera or SCMOSCamera) from SMLMData.
  If not provided, a default IdealCamera is created.

## Primary Interface (Recommended - PSF-Aware)
- `psf_sigma::Real`: PSF sigma in microns (physical units, e.g., 0.13 for 130nm PSF).
  Automatically converted to pixels using camera pixel size and sets optimal DoG filter parameters.
  **Requires camera to be provided.**
- `min_photons::Real`: Minimum total photons for detection (default: 500.0).
  Automatically converted to appropriate intensity threshold.

## Advanced Interface (Direct Control)
For expert users who want direct control over filter parameters:
- `sigma_small::Real`: Small Gaussian sigma in pixels (default: 1.0).
- `sigma_large::Real`: Large Gaussian sigma in pixels (default: 2.0).
- `minval::Real`: DoG filter intensity threshold (default: 0.0).

Note: If `psf_sigma` is provided, it overrides sigma_small/sigma_large/minval.

## Other Parameters
- `boxsize::Int`: Size of the box to cut out around each local maximum in pixels (default: 7).
- `overlap::Real`: Maximum overlap allowed between boxes in pixels (default: 2.0).
- `use_gpu::Bool`: Perform convolution and local max finding on GPU (default: true).

# Returns
`ROIBatch` with the following fields:
- `data`: ROI stack (boxsize × boxsize × n_rois) containing image patches
- `x_corners`: Vector of x (column) corner positions in camera coordinates
- `y_corners`: Vector of y (row) corner positions in camera coordinates
- `frame_indices`: Vector of frame indices for each ROI
- `camera`: Camera object (provided or default IdealCamera)
- `roi_size`: Size of each ROI (square)

# Details on filtering

The image stack is convolved with a difference of Gaussians (DoG) filter
to identify blobs and local maxima. The DoG is computed from two Gaussian
kernels with standard deviations `sigma_small` and `sigma_large`.

When using the PSF-aware interface with `psf_sigma` (in microns):
- psf_sigma is converted to pixels using camera pixel size
- sigma_small = 1.0 × psf_sigma_pixels (matches PSF for optimal blob detection)
- sigma_large = 2.0 × psf_sigma_pixels (background suppression)
- minval is automatically calculated from min_photons accounting for PSF spreading and DoG response

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
# Recommended: PSF-aware detection with physical units
camera = IdealCamera(1:256, 1:256, 0.1f0)  # 256×256 pixels, 100nm pixel size

roi_batch = getboxes(imagestack, camera;
    psf_sigma = 0.13,              # PSF sigma in microns (physical units)
    min_photons = 500.0,           # Detect emitters with ≥500 photons
    boxsize = 11)

# Access results
boxes = roi_batch.data             # (11 × 11 × n_rois)
x_corners = roi_batch.x_corners    # x (col) positions
y_corners = roi_batch.y_corners    # y (row) positions
frames = roi_batch.frame_indices

# Advanced: Direct control over filter parameters
roi_batch = getboxes(imagestack;
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

  return ROIBatch(boxstack, x_corners, y_corners, frame_indices, camera)
end




