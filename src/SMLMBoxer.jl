"""
    SMLMBoxer

High-performance particle/blob detection in SMLM image stacks using difference-of-Gaussians
filtering with GPU acceleration and sCMOS variance-weighted filtering support.

Entry point: [`getboxes`](@ref), configured by keywords or a [`BoxerConfig`](@ref); it returns
an `ROIBatch` of boxes and a [`BoxesInfo`](@ref) with processing metadata.
"""
module SMLMBoxer

using NNlib: NNlib
using CUDA: CUDA, CUDABackend, CuArray
using cuDNN: cuDNN  # Required for NNlib's cuDNN backend
using KernelAbstractions: KernelAbstractions, @index, @kernel, @ndrange, CPU
using Statistics: mean  # For mean gain/QE in threshold calculation

# Re-export ROIBatch and SingleROI from SMLMData for convenience
using SMLMData: SMLMData, AbstractCamera, IdealCamera, SCMOSCamera, ROIBatch, SingleROI,
    AbstractSMLMConfig, AbstractSMLMInfo
export getboxes, ROIBatch, SingleROI, BoxerConfig, BoxesInfo, recommend_batch_size

include("gpu.jl")
include("types.jl")
include("filter.jl")
include("localmax.jl")
include("coords.jl")
include("boxes.jl")
include("interface.jl")

end
