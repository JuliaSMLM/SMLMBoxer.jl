module SMLMBoxer

using NNlib
using CUDA
using KernelAbstractions
using SMLMData

# Re-export ROIBatch and SingleROI from SMLMData for convenience
using SMLMData: ROIBatch, SingleROI
export getboxes, ROIBatch, SingleROI 

include("types.jl")
include("filter.jl")
include("localmax.jl")
include("coords.jl")
include("boxes.jl")
include("interface.jl")

end
