using Revise
using SMLMBoxer
using Flux
using CUDA
using BenchmarkTools

args = SMLMBoxer.GetBoxesArgs(; use_gpu = true)

H, W = 128, 256
nobs = 1000



args.use_gpu = false
localmax = args.use_gpu ? CUDA.rand(Float32, H, W, 1, nobs) :
           rand(Float32, H, W, 1, nobs)
println("cpu")
@btime locmaxim  = SMLMBoxer.genlocalmaximage(localmax, args);
@btime coords = SMLMBoxer.maxima2coords(localmax, args);

args.use_gpu = true
localmax = args.use_gpu ? CUDA.rand(Float32, H, W, 1, nobs) :
           rand(Float32, H, W, 1, nobs)
println("gpu")  
@btime locmaxim  = SMLMBoxer.genlocalmaximage(localmax, args);
@btime coords = SMLMBoxer.maxima2coords(localmax, args);


