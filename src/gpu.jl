"""
GPU utilities for SMLMBoxer.

Provides backend selection and GPU memory waiting functionality following
the JuliaSMLM convention:
- backend: :cpu | :gpu | :auto
- auto_timeout: seconds to wait in Auto mode (default 300.0)
- gpu_timeout: seconds to wait in GPU mode (default Inf)
- on_wait: optional callback(elapsed, available, required)
"""

"""
    has_cuda() -> Bool

Check if CUDA is available. Wrapper for CUDA.functional().
"""
has_cuda() = CUDA.functional()

"""
    is_gpu_query_error(e) -> Bool

True when `e` comes from querying a GPU (an NVML error, or a CUDA driver error such as a
failed context creation). `:auto` falls back to the CPU on these; anything else propagates.
"""
is_gpu_query_error(e) = e isa CUDA.NVML.NVMLError || e isa CUDA.CuError

"""
    is_gpu_resource_error(e) -> Bool

True only for GPU memory errors and an unavailable device, including context-creation
races. GPU memory errors and an unavailable device are retried until the timeout;
any other GPU failure falls back to the CPU at once under `:auto`, or is raised at once
under `:gpu`; an interrupt (Ctrl-C) always stops the call.
"""
is_gpu_resource_error(e) = e isa CUDA.OutOfGPUMemoryError ||
    (e isa CUDA.CuError &&
     e.code in (CUDA.ERROR_OUT_OF_MEMORY, CUDA.ERROR_DEVICE_UNAVAILABLE))

"""
    free_gpu_arrays(arrays...; failed=false)

Explicitly free owned CuArray temporaries, ignoring CPU arrays and `nothing`. Never pass
borrowed arrays. Attempt every free even if one fails; on an already failed operation,
preserve its original exception instead of replacing it with a cleanup error.
"""
function free_gpu_arrays(arrays...; failed::Bool=false)
    first_error = nothing
    for array in arrays
        array isa CuArray || continue
        try
            CUDA.unsafe_free!(array)
        catch e
            first_error === nothing && (first_error = e)
        end
    end
    !failed && first_error !== nothing && throw(first_error)
    return nothing
end

"""
    release_gpu_memory()

Return explicitly freed GPU allocations to the driver after a successful call, without
running garbage collection. CUDA.jl 6 requires device synchronization before purging
caches, then another synchronization before trimming the pool. CUDA.jl 5 has no
`RECLAIM_PURGE`; its default `reclaim()` synchronizes and trims without a full GC.
CUDA.jl 6.5.x may retain pool memory despite this release (see CHANGELOG's known issue).
"""
function release_gpu_memory()
    if isdefined(CUDA, :RECLAIM_PURGE)
        CUDA.device_synchronize()
        CUDA.reclaim(CUDA.RECLAIM_PURGE)
        CUDA.device_synchronize()
        CUDA.trim(CUDA.memory_pool(CUDA.device()))
    else
        CUDA.reclaim()
    end
    return nothing
end

"""
    reclaim_failed_gpu_attempt()

Try to release allocations after a failed GPU attempt. A poisoned CUDA context may make
reclaim throw; preserve the processing error and allow CPU fallback in that case.
"""
function reclaim_failed_gpu_attempt()
    try
        GC.gc(false)
        CUDA.reclaim()
    catch
        # Cleanup must neither replace the original error nor prevent CPU fallback.
    end
    return nothing
end

"""
    GPU_CONTEXT_RESERVE

Bytes reserved for a CUDA context and the driver. The lab RTX 4090 reported 878.812 MiB
in use at `start` in dev/output/drop-cudnn/pool-diag2-branch.log; round up to 1 GiB.
"""
const GPU_CONTEXT_RESERVE = 1024^3

"""
    frame_fits_some_gpu(frame_bytes, totals, reserve) -> Bool

Whether any visible device can free enough memory for one frame after reserving `reserve`
bytes for its context and driver. `frame_bytes` includes the same safety margin used by
polling; `totals` contains device capacities in bytes.
An empty device list cannot hold a frame.
"""
frame_fits_some_gpu(frame_bytes, totals, reserve) =
    any(total -> total >= reserve && frame_bytes <= total - reserve, totals)

"""
    nvml_device(i) -> CUDA.NVML.Device

Return the NVML device for GPU `i`, a 0-based CUDA ordinal, without creating a context.
Resolve the CUDA device's UUID so `CUDA_VISIBLE_DEVICES` and MIG devices are respected.
"""
function nvml_device(i::Integer)
    dev = CUDA.CuDevice(i)
    uuid = CUDA.uuid(dev)
    return CUDA.NVML.Device(uuid; mig = uuid != CUDA.parent_uuid(dev))
end

"""
    nvml_free_memory(i) -> Union{Int, Nothing}

Return free memory in bytes on GPU `i`, a 0-based CUDA ordinal, using NVML without a
CUDA context. Return `nothing` if NVML is unavailable or reports
`NVML_ERROR_NOT_SUPPORTED`. Other errors propagate.
"""
function nvml_free_memory(i::Integer)
    CUDA.NVML.has_nvml() || return nothing
    try
        return CUDA.NVML.memory_info(nvml_device(i)).free
    catch e
        if !(e isa CUDA.NVML.NVMLError &&
                e.code == CUDA.NVML.NVML_ERROR_NOT_SUPPORTED)
            rethrow()
        end
        return nothing
    end
end

"""
    cuda_free_memory(i) -> Int

Return free memory in bytes on GPU `i`, a 0-based CUDA ordinal, from CUDA. Activates the
device (creating its context if needed) for the query and restores the previous device.
"""
cuda_free_memory(i::Integer) = CUDA.device!(CUDA.free_memory, i)

"""
    gpu_free_memory(i) -> Int

Return free memory in bytes on GPU `i`, a 0-based CUDA ordinal, using `nvml_free_memory`
when supported. Otherwise query CUDA on that device and restore the previous device. Other
errors propagate.
"""
gpu_free_memory(i::Integer) = @something(nvml_free_memory(i), cuda_free_memory(i))

"""
    poll_gpu_nvml(required_bytes; timeout=30.0, poll=0.5, on_wait=nothing) -> (Bool, Int)

Poll all CUDA-visible GPUs via NVML until one has sufficient free memory and low contention.
No CUDA context creation needed when NVML supports memory queries, making polling
safe under multi-process contention. Otherwise queries CUDA on that device,
restores the previous device and skips NVML contention checks for that device.

# Arguments
- `required_bytes`: Minimum bytes needed (1.5x safety margin applied internally)
- `timeout`: Maximum seconds to poll (default 30.0, use Inf for unlimited)
- `poll`: Seconds between checks (default 0.5)
- `on_wait`: Optional callback(elapsed, best_available, required) called each poll

# Returns
- `(true, device_id)` with a 0-based CUDA ordinal if a GPU became available,
  `(false, -1)` if timeout reached

# Contention Detection
A GPU is considered contended when other processes are present AND either:
- Free memory is insufficient (< required × 1.5)
- Compute utilization exceeds 90%

When other processes are present but memory is sufficient and utilization is low,
the GPU is still considered available.
"""
function poll_gpu_nvml(required_bytes::Integer;
        timeout::Real = 30.0,
        poll::Real = 0.5,
        on_wait = nothing)

    n = length(CUDA.devices())
    required_with_margin = required_bytes * 1.5

    deadline = time() + timeout
    start = time()

    while time() < deadline
        best_device = -1
        best_free = 0

        for i in 0:(n-1)
            nvml_free = nvml_free_memory(i)
            free_mem = @something(nvml_free, cuda_free_memory(i))

            # Not enough memory on this device
            free_mem < required_with_margin && continue

            # Check contention: other processes on this GPU
            if nvml_free !== nothing
                dev = nvml_device(i)
                procs = CUDA.NVML.compute_processes(dev)
                other_procs = length(procs) - (getpid() in keys(procs) ? 1 : 0)

                if other_procs > 0
                    rates = CUDA.NVML.utilization_rates(dev)
                    rates.compute > 0.9 && continue  # GPU saturated, skip
                end
            end

            # This GPU is available - track best (most free memory)
            if free_mem > best_free
                best_free = free_mem
                best_device = i
            end
        end

        if best_device >= 0
            return (true, best_device)
        end

        # Callback for progress reporting
        if on_wait !== nothing
            # Report best available across all GPUs for visibility
            max_free = maximum(gpu_free_memory(i) for i in 0:(n-1))
            on_wait(time() - start, max_free, required_bytes)
        end

        # Jittered sleep to desync competing processes
        sleep(poll * (1.0 + 0.2 * (rand() - 0.5)))
    end

    return (false, -1)
end

"""
    estimate_gpu_memory_per_frame(height, width, camera) -> Int

Estimate GPU memory required per frame.

# Arguments
- `height`: Image height in pixels
- `width`: Image width in pixels
- `camera`: Camera object (affects memory multiplier)

# Returns
- Estimated bytes needed per frame for GPU processing
"""
function estimate_gpu_memory_per_frame(height::Int, width::Int, camera)
    n_copies = camera isa SCMOSCamera ? 8 : 6
    return height * width * sizeof(Float32) * n_copies
end
