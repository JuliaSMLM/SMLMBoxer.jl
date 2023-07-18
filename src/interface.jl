"""
    getboxes(; kwargs...)

Detect particles/blobs in a multidimensional image stack and return
coordinates and boxed regions centered around local maxima. 

# Arguments
- `imagestack::AbstractArray{<:Real}`: The input image stack. Should be 2D or 3D.
- `boxsize::Int`: Size of the box to cut out around each local maximum.  
- `overlap::Real`: Minimum distance between box centers. Boxes closer than this will
  be removed.
- `sigma_small::Real`: Sigma for small Gaussian blur kernel. 
- `sigma_large::Real`: Sigma for large Gaussian blur kernel.
- `minval::Real`: Minimum value to consider as a local maximum.  
- `use_gpu::Bool`: Perform convolution on GPU. Requires CuArrays package.

# Returns
- `boxstack::AbstractArray{<:Real}`: Array with dimensions (boxsize, boxsize, nboxes).
  Each image in the stack contains a small boxed region from imagestack. 
- `coords::Matrix{Float32}`: Coordinates of boxes N x (row, col, frame).
  
# Details on filtering

The image stack is convolved with a difference of Gaussians (DoG) filter
to identify blobs and local maxima. The DoG is computed from two Gaussian
kernels with standard deviations `sigma_small` and `sigma_large`.  

The convolution is performed either on CPU or GPU, depending on `use_gpu`.
After filtering, local maxima above `minval` are identified. Boxes are cut
out around each maximum, excluding overlaps.

# Examples  
```julia
boxes, coords = getboxes(imagestack, boxsize=7, overlap=2.0,  
                         sigma_small=1.0, sigma_large=2.0)
```
"""
function getboxes(; kwargs...)

  args = GetBoxesArgs(; kwargs...)

  imagestack = reshape_for_flux(args.imagestack)

  # Determine whether to perform calculations on the GPU
  args.use_gpu = args.use_gpu && has_cuda() && CUDA.functional()

  minkernelsize = 3
  kernelsize = Int(floor(args.boxsize - args.overlap))
  kernelsize = max(minkernelsize, kernelsize)

  if args.use_gpu
    filtered_stack = dog_filter(imagestack |> gpu, args)
    coords = findlocalmax(filtered_stack, kernelsize; minval=args.minval, use_gpu=args.use_gpu)
    CUDA.synchronize()
  else
    filtered_stack = dog_filter(imagestack, args)
    coords = findlocalmax(filtered_stack, kernelsize; minval=args.minval, use_gpu=args.use_gpu)
  end

  coords = removeoverlap(coords, args)
  boxstack = getboxstack(imagestack, coords, args)

  # Return the box stack and the coords  
  return boxstack, coords
end



