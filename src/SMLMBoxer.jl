module SMLMBoxer

using Flux
using CUDA

include("filter.jl")
include("coords.jl")
include("boxes.jl")
include("interface.jl")

end
