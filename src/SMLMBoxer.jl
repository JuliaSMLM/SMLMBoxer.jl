module SMLMBoxer

using Flux
using CUDA

export getboxes 

include("types.jl")
include("filter.jl")
include("localmax.jl")
include("coords.jl")
include("boxes.jl")
include("interface.jl")

end
