"""
    SMLMBoxer

High-performance particle/blob detection in SMLM image stacks using difference-of-Gaussians
filtering with GPU acceleration and sCMOS variance-weighted filtering support.

# API Overview
For a comprehensive overview of the API, use help mode:

    ?SMLMBoxer.api

Or access the complete API documentation programmatically:

    docs = SMLMBoxer.api()
"""
module SMLMBoxer

using NNlib
using CUDA
using cuDNN  # Required for NNlib's cuDNN backend
using KernelAbstractions
using SMLMData
using Statistics: mean  # For mean gain/QE in threshold calculation

# Re-export ROIBatch and SingleROI from SMLMData for convenience
using SMLMData: ROIBatch, SingleROI
export getboxes, ROIBatch, SingleROI, BoxerConfig, BoxesInfo, recommend_batch_size

include("gpu.jl")
include("types.jl")
include("filter.jl")
include("localmax.jl")
include("coords.jl")
include("boxes.jl")
include("interface.jl")
include("api.jl")

end
