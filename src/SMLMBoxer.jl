module SMLMBoxer

using Flux
using CUDA
using KernelAbstractions
using SMLMData

export getboxes 

include("types.jl")
include("filter.jl")
include("localmax.jl")
include("coords.jl")
include("boxes.jl")
include("interface.jl")

end
