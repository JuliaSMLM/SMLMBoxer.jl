var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = SMLMBoxer","category":"page"},{"location":"#SMLMBoxer","page":"Home","title":"SMLMBoxer","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for SMLMBoxer.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [SMLMBoxer]","category":"page"},{"location":"#SMLMBoxer.convolve-Tuple{AbstractArray{<:Real}, Matrix{Float32}}","page":"Home","title":"SMLMBoxer.convolve","text":"convolve(imagestack, kernel; use_gpu=false)\n\nConvolve imagestack with given kernel.\n\nArguments\n\nimagestack: Input array of image data\nkernel: Kernel to convolve with \n\nKeyword Arguments\n\nuse_gpu: Whether to use GPU \n\nReturns\n\nfiltered_stack: Convolved image stack\n\n\n\n\n\n","category":"method"},{"location":"#SMLMBoxer.dog_filter-Tuple{AbstractArray{<:Real}, SMLMBoxer.GetBoxesArgs}","page":"Home","title":"SMLMBoxer.dog_filter","text":"dog_filter(imagestack, args)\n\nApply DoG filter to imagestack based on args.\n\nArguments\n\nimagestack: Input array of image data \nargs: Arguments with sigma values  \n\nReturns\n\nfiltered_stack: Filtered image stack \n\n\n\n\n\n","category":"method"},{"location":"#SMLMBoxer.dog_kernel-Tuple{Float32, Float32}","page":"Home","title":"SMLMBoxer.dog_kernel","text":"dog_kernel(s1, s2)\n\nCompute difference of Gaussian kernels.   \n\nArguments\n\nsigma_small: Sigma for small Gaussian \nsigma_large: Sigma for large Gaussian\n\nReturns\n\ndog: Difference of Gaussians kernel\n\n\n\n\n\n","category":"method"},{"location":"#SMLMBoxer.fillbox!-Tuple{AbstractMatrix{<:Real}, AbstractArray{<:Real, 4}, Vararg{Int64, 4}}","page":"Home","title":"SMLMBoxer.fillbox!","text":"fillbox!(box, imagestack, row, col, im, boxsize)\n\nFill a box with a crop from the imagestack.\n\nArguments\n\nbox: Array to fill with box crop\nimagestack: Input image stack \nrow, col, im: Coords for crop\nboxsize: Size of box\n\nReturns\n\nNothing, fills box in-place\n\n\n\n\n\n","category":"method"},{"location":"#SMLMBoxer.findlocalmax-Tuple{AbstractArray{<:Real}, Int64}","page":"Home","title":"SMLMBoxer.findlocalmax","text":"findlocalmax(imagestack, kernelsize; minval=0.0, use_gpu=false)\n\nFind the coordinates of local maxima in an image.\n\nArguments\n\nimagestack: An array of real numbers representing the image data.\nkernelsize: The size of the kernel used to identify local maxima.\n\nKeyword Arguments\n\nminval: The minimum value a local maximum must have to be considered valid (default: 0.0).\nuse_gpu: Whether or not to use GPU acceleration (default: false).\n\nReturns\n\ncoords: The coordinates of the local maxima in the image.\n\n\n\n\n\n","category":"method"},{"location":"#SMLMBoxer.gaussian_2d-Tuple{Float32, Int64}","page":"Home","title":"SMLMBoxer.gaussian_2d","text":"gaussian_2d(sigma, ksize)\n\nCreate a 2D Gaussian kernel.\n\nArguments\n\nsigma: Standard deviation\nkernelsize: Kernel size\n\nReturns\n\nkernel: Normalized 2D Gaussian kernel  \n\n\n\n\n\n","category":"method"},{"location":"#SMLMBoxer.genlocalmaximage-Tuple{AbstractArray{<:Real}, Int64}","page":"Home","title":"SMLMBoxer.genlocalmaximage","text":"genlocalmaximage(imagestack, kernelsize; minval=0.0, use_gpu=false)\n\nGenerate an image highlighting the local maxima.\n\nArguments\n\nimagestack: An array of real numbers representing the image data.\nkernelsize: The size of the kernel used to identify local maxima.\n\nKeyword Arguments\n\nminval: The minimum value a local maximum must have to be considered valid (default: 0.0).\nuse_gpu: Whether or not to use GPU acceleration (default: false).\n\nReturns\n\nlocalmaximage: An image with local maxima highlighted.\n\n\n\n\n\n","category":"method"},{"location":"#SMLMBoxer.getboxes-Tuple{}","page":"Home","title":"SMLMBoxer.getboxes","text":"getboxes(; kwargs...)\n\nDetect particles/blobs in a multidimensional image stack and return coordinates and boxed regions centered around local maxima. \n\nArguments\n\nimagestack::AbstractArray{<:Real}: The input image stack. Should be 2D or 3D.\nboxsize::Int: Size of the box to cut out around each local maximum.  \noverlap::Real: Minimum distance between box centers. Boxes closer than this will be removed.\nsigma_small::Real: Sigma for small Gaussian blur kernel. \nsigma_large::Real: Sigma for large Gaussian blur kernel.\nminval::Real: Minimum value to consider as a local maximum.  \nuse_gpu::Bool: Perform convolution on GPU. Requires CuArrays package.\n\nReturns\n\nboxstack::AbstractArray{<:Real}: Array with dimensions (boxsize, boxsize, nboxes). Each image in the stack contains a small boxed region from imagestack. \ncoords::Matrix{Float32}: Coordinates of boxes N x (row, col, frame).\n\nDetails on filtering\n\nThe image stack is convolved with a difference of Gaussians (DoG) filter to identify blobs and local maxima. The DoG is computed from two Gaussian kernels with standard deviations sigma_small and sigma_large.  \n\nThe convolution is performed either on CPU or GPU, depending on use_gpu. After filtering, local maxima above minval are identified. Boxes are cut out around each maximum, excluding overlaps.\n\nExamples\n\nboxes, coords = getboxes(imagestack, boxsize=7, overlap=2.0,  \n                         sigma_small=1.0, sigma_large=2.0)\n\n\n\n\n\n","category":"method"},{"location":"#SMLMBoxer.getboxstack-Tuple{Any, Any, SMLMBoxer.GetBoxesArgs}","page":"Home","title":"SMLMBoxer.getboxstack","text":"getboxstack(imagestack, coords, args::GetBoxesArgs) \n\nCut out box regions from imagestack centered on coords.\n\nArguments\n\nimagestack: Input image stack\ncoords: Coords of box centers  \nargs: Parameters\n\nReturns\n\nboxstack: Array with box crops from imagestack\n\n\n\n\n\n","category":"method"},{"location":"#SMLMBoxer.maxima2coords-Tuple{AbstractArray{Float32}}","page":"Home","title":"SMLMBoxer.maxima2coords","text":"maxima2coords(imagestack)\n\nGet coordinates of all non-zero pixels in input stack \n\nArguments\n\nimagestack: Input image stack\n\nReturns\n\ncoords: List of coords for each frame\n\n\n\n\n\n","category":"method"},{"location":"#SMLMBoxer.removeoverlap-Tuple{Vector{Matrix{Float32}}, SMLMBoxer.GetBoxesArgs}","page":"Home","title":"SMLMBoxer.removeoverlap","text":"removeoverlap(coords, args)\n\nRemove overlapping coords based on distance.\n\nArguments\n\ncoords: List of coords\nargs: Parameters  \n\nReturns\n\ncoords: Coords with overlaps removed \n\n\n\n\n\n","category":"method"},{"location":"#SMLMBoxer.reshape_for_flux-Tuple{AbstractArray{<:Real}}","page":"Home","title":"SMLMBoxer.reshape_for_flux","text":"reshapeforflux(arr::AbstractArray)\n\nReshape array to have singleton dims for Flux.jl convolution.  \n\nArguments\n\narr: Input array, must be 2D or 3D\n\nReturns\n\nReshaped array with added singleton dimensions\n\n\n\n\n\n","category":"method"}]
}
