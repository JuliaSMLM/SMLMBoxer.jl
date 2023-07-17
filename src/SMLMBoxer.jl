module SMLMBoxer

using Flux
using CUDA

include("types.jl")
include("filter.jl")
include("coords.jl")
include("boxes.jl")
include("interface.jl")

end