# SMLMBoxer

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaSMLM.github.io/SMLMBoxer.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSMLM.github.io/SMLMBoxer.jl/dev/)
[![Build Status](https://github.com/JuliaSMLM/SMLMBoxer.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaSMLM/SMLMBoxer.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaSMLM/SMLMBoxer.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSMLM/SMLMBoxer.jl)

*SMLMBoxer.jl* is a Julia package that provides a fast and efficient method for detecting particles or blobs in a multidimensional image stack and cutting out sub-regions around local maxima. The package exports a single high-level interface function `getboxes()`.

## Usage
The main function provided by the package is `getboxes()`, which detects particles or blobs in a multidimensional image stack and returns coordinates and boxed regions centered around local maxima. This function is highly customizable with several optional arguments and is capable of performing calculations on the GPU if available.

### Example
```julia
boxes, boxcoords, maxcoords = getboxes(; imagestack, boxsize=7, overlap=2.0, sigma_small=1.0, sigma_large=2.0)
```

### Keyword Arguments
- `imagestack::AbstractArray{<:Real}`: The input image stack. Should be 2D or 3D.
- `boxsize::Int`: Size of the box to cut out around each local maximum (pixels).  
- `overlap::Real`: Amount of overlap allowed between boxes (pixels). 
- `sigma_small::Real`: Sigma for small Gaussian blur kernel (pixels). 
- `sigma_large::Real`: Sigma for large Gaussian blur kernel (pixels).
- `minval::Real`: Minimum value to consider as a local maximum.  
- `use_gpu::Bool`: Perform convolution and local max finding on GPU. 

### Returns
- `boxstack::AbstractArray{<:Real}`: Array with dimensions (boxsize, boxsize, nboxes). Each image in the stack contains a small subregion from imagestack centered around a local maximum.
- `boxcoords::Matrix{Float32}`: Coordinates of the upper left corner of the boxes N x (row, col, frame).
- `maxcoords::Matrix{Float32}`: Coordinates of the maxima N x (row, col, frame).

The `getboxes()` function performs a Difference of Gaussians (DoG) filter on the image stack to identify blobs and local maxima. The DoG is computed from two Gaussian kernels with standard deviations specified by `sigma_small` and `sigma_large`. After filtering, local maxima above a certain minimum value (`minval`) are identified, and boxes are cut out around each maximum, excluding overlaps. This computation can be performed either on a CPU or a GPU, depending on the `use_gpu` argument.

## Additional Tools 

In addition to the `getboxes()` function, *SMLMBoxer.jl* provides a number of lower-level tools that can be useful in processing and analyzing image stacks. These are not exported. 

- `SMLMBoxer.genlocalmaximage(imagestack, kernelsize; minval=0.0, use_gpu=false)`: Generates an image where local maxima in the original image are the only non-zero pixels. 

- `SMLMBoxer.findlocalmax(imagestack, kernelsize; minval=0.0, use_gpu=false)`: Returns the coordinates of local maxima in an image. 

- `SMLMBoxer.convolve(imagestack, kernel; use_gpu=false)`: This function convolves an image stack with a given kernel.

