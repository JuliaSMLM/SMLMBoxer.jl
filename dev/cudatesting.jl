using Revise
using SMLMBoxer
using Flux
using CUDA
using BenchmarkTools


for i in 0:length(CUDA.devices())-1
    CUDA.device!(i)  # Switch to the first device (devices are 0-indexed)
    println("\nSwitched to device: ", CUDA.device())
    # Get the total amount of memory on the device (in GB)
    total_mem = CUDA.total_memory() / 1024^3
    println("Total memory: ", total_mem, " GB")

    # Get the amount of free memory on the device (in GB)
    free_mem = CUDA.available_memory() / 1024^3
    println("Free memory: ", free_mem, " GB")

    # Calculate and print the amount of used memory (in GB)
    used_mem = total_mem - free_mem
    println("Used memory: ", used_mem, " GB\n")
end



args = SMLMBoxer.GetBoxesArgs(; use_gpu=true)

H, W = 128, 256
nobs = 1000


args.use_gpu = false
localmax = args.use_gpu ? CUDA.rand(Float32, H, W, 1, nobs) :
           rand(Float32, H, W, 1, nobs)
data = localmax .> 0.999 .* localmax
data = dropdims(data, dims=3)

println("cpu")
# @btime locmaxim  = SMLMBoxer.genlocalmaximage(localmax, args);
# @btime coords = SMLMBoxer.maxima2coords(localmax, args);
args.imagestack = data
@btime boxes, coords = SMLMBoxer.getboxes(imagestack=data, use_gpu=false);

@btime boxes, coords = SMLMBoxer.getboxes(imagestack=data, use_gpu=true);



args.use_gpu = true
localmax = args.use_gpu ? CUDA.rand(Float32, H, W, 1, nobs) :
           rand(Float32, H, W, 1, nobs)
println("gpu")
@btime locmaxim = SMLMBoxer.genlocalmaximage(localmax, args);
@btime coords = SMLMBoxer.maxima2coords(localmax |> cpu, args);


# @profview coords = SMLMBoxer.maxima2coords(localmax, args);
