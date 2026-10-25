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
    nvml_free_memory(i) -> Union{Int, Nothing}

Return free memory in bytes on GPU `i` (0-based), using NVML without a CUDA context.
Return `nothing` if NVML is unavailable or reports `NVML_ERROR_NOT_SUPPORTED`.
Other errors propagate.
"""
function nvml_free_memory(i::Integer)
    CUDA.NVML.has_nvml() || return nothing
    try
        return CUDA.NVML.memory_info(CUDA.NVML.Device(i)).free
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

Return free memory in bytes on GPU `i` (0-based) from CUDA. Activates the device (creating
its context if needed) for the query and restores the previous device.
"""
cuda_free_memory(i::Integer) = CUDA.device!(CUDA.free_memory, i)

"""
    gpu_free_memory(i) -> Int

Return free memory in bytes on GPU `i` (0-based), using `nvml_free_memory` when supported.
Otherwise query CUDA on that device and restore the previous device. Other errors propagate.
"""
gpu_free_memory(i::Integer) = @something(nvml_free_memory(i), cuda_free_memory(i))

"""
    find_best_gpu() -> Int

Find the GPU with the most free memory and switch to it.

Returns the device index (0-based). On single-GPU systems, returns 0 immediately.

Uses NVML to query free memory on each device without creating CUDA contexts,
avoiding `cuDevicePrimaryCtxRetain` OOM errors under multi-process contention.
If NVML memory queries are not supported, temporarily activates each device to
query CUDA instead. Otherwise only calls `CUDA.device!()` on the selected device.

# Example
```julia
best = find_best_gpu()  # Switches to best GPU
# Now all CUDA operations use that GPU
```
"""
function find_best_gpu()
    !CUDA.functional() && return 0
    n = length(CUDA.devices())
    n == 1 && return 0

    # Prefer NVML to avoid creating CUDA contexts under contention
    best, maxfree = 0, 0
    for i in 0:(n-1)
        free_mem = gpu_free_memory(i)
        if free_mem > maxfree
            maxfree, best = free_mem, i
        end
    end

    CUDA.device!(best)
    @info "Selected GPU $best with $(Base.format_bytes(maxfree)) free"
    return best
end

"""
    poll_gpu_nvml(required_bytes; timeout=30.0, poll=0.5, on_wait=nothing) -> (Bool, Int)

Poll ALL GPUs via NVML until one has sufficient free memory and low contention.
No CUDA context creation needed when NVML supports memory queries, making polling
safe under multi-process contention. Otherwise queries CUDA on that device,
restores the previous device and skips NVML contention checks for that device.

# Arguments
- `required_bytes`: Minimum bytes needed (1.5x safety margin applied internally)
- `timeout`: Maximum seconds to poll (default 30.0, use Inf for unlimited)
- `poll`: Seconds between checks (default 0.5)
- `on_wait`: Optional callback(elapsed, best_available, required) called each poll

# Returns
- `(true, device_id)` if a GPU became available, `(false, -1)` if timeout reached

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
                dev = CUDA.NVML.Device(i)
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
    wait_for_gpu_memory(required_bytes; timeout=30.0, poll=0.5, on_wait=nothing) -> Bool

Wait until current GPU device has sufficient available memory.
Uses CUDA calls (requires active context). Used by :gpu mode after device selection.

# Arguments
- `required_bytes`: Minimum bytes needed (with safety margin applied internally)
- `timeout`: Maximum seconds to wait (default 30.0, use Inf for unlimited)
- `poll`: Seconds between checks (default 0.5)
- `on_wait`: Optional callback(elapsed, available, required) called each poll

# Returns
- `true` if memory became available, `false` if timeout reached
"""
function wait_for_gpu_memory(required_bytes::Integer;
        timeout::Real = 30.0,
        poll::Real = 0.5,
        on_wait = nothing)

    required_with_margin = required_bytes * 1.5

    deadline = time() + timeout
    start = time()

    while time() < deadline
        CUDA.reclaim()

        available = CUDA.free_memory()
        if available >= required_with_margin
            return true
        end

        if on_wait !== nothing
            on_wait(time() - start, available, required_bytes)
        end

        sleep(poll * (1.0 + 0.2 * (rand() - 0.5)))
    end

    return false
end

"""
    select_backend(backend::Symbol, required_bytes;
                   auto_timeout=300.0, gpu_timeout=Inf, on_wait=nothing) -> (Symbol, Int)

Select compute backend with two-layer GPU contention handling.

Layer 1 (NVML polling): Scans all GPUs via NVML without creating CUDA contexts
when supported. If NVML cannot report memory, queries CUDA on that device and
skips its NVML contention checks. Polls through timeout with jittered backoff.
First GPU with sufficient free memory and low contention wins.

Layer 2 (runtime try/catch): Applied by caller for :auto mode. If CUDA errors
occur during processing despite NVML pre-check, falls back to CPU.

# Arguments
- `backend`: :cpu, :gpu, or :auto
- `required_bytes`: Estimated GPU memory needed for processing
- `auto_timeout`: Max wait for :auto mode before CPU fallback (default 300.0)
- `gpu_timeout`: Max wait for :gpu mode (default Inf - wait forever)
- `on_wait`: Optional callback(elapsed, available, required)

# Returns
- `(backend::Symbol, device_id::Int)` - selected backend and GPU device (0-based, -1 for CPU)

# Behavior
- `:cpu` - Returns (:cpu, -1) immediately
- `:gpu` - NVML poll for device, then CUDA wait_for_gpu_memory. Errors if unavailable/timeout
- `:auto` - Poll with timeout; query errors or timeout fall back to (:cpu, -1) with warning
"""
function select_backend(backend::Symbol, required_bytes::Integer;
        auto_timeout::Real = 300.0,
        gpu_timeout::Real = Inf,
        on_wait = nothing)

    backend in (:cpu, :gpu, :auto) || error("backend must be :cpu, :gpu, or :auto")

    if backend == :cpu
        return (:cpu, -1)

    elseif backend == :gpu
        if !has_cuda() || !CUDA.functional()
            error("GPU backend requested but CUDA is not functional")
        end

        # Poll to find an available device, preferring NVML
        available, device_id = poll_gpu_nvml(required_bytes; timeout=gpu_timeout, on_wait=on_wait)
        if !available
            error("No GPU available after $(gpu_timeout)s. " *
                  "Required: $(Base.format_bytes(required_bytes))")
        end

        # Activate chosen device and confirm memory via CUDA
        CUDA.device!(device_id)
        if !wait_for_gpu_memory(required_bytes; timeout=min(gpu_timeout, 10.0), on_wait=on_wait)
            error("GPU $device_id memory not available. " *
                  "Required: $(Base.format_bytes(required_bytes)), " *
                  "Available: $(Base.format_bytes(CUDA.free_memory()))")
        end
        return (:gpu, device_id)

    else  # :auto
        if !has_cuda() || !CUDA.functional()
            return (:cpu, -1)
        end

        # Poll all GPUs, falling back to CPU if the device query fails
        available, device_id = try
            poll_gpu_nvml(required_bytes; timeout=auto_timeout, on_wait=on_wait)
        catch e
            is_gpu_query_error(e) || rethrow()
            @warn "GPU device query failed, using CPU" exception=(e, catch_backtrace())
            return (:cpu, -1)
        end
        if available
            CUDA.device!(device_id)
            return (:gpu, device_id)
        else
            @warn "No GPU available after $(auto_timeout)s, using CPU. " *
                  "Required: $(Base.format_bytes(required_bytes))"
            return (:cpu, -1)
        end
    end
end

"""
    estimate_gpu_memory(imagestack, camera) -> Int

Estimate GPU memory required for processing imagestack.

# Arguments
- `imagestack`: Input image array (H, W, ..., F)
- `camera`: Camera object (affects memory multiplier)

# Returns
- Estimated bytes needed for GPU processing

# Memory Model
Standard DoG path: 6x input size
- Input, filtered_small, filtered_large, DoG result, localmax temps, GC margin

Variance-weighted (SCMOSCamera): 8x input size
- Additional workspace for per-pixel variance weighting (in-place DoG saves one copy)
"""
function estimate_gpu_memory(imagestack::AbstractArray, camera)
    # Memory multiplier matches _getboxes_impl
    n_copies = camera isa SCMOSCamera ? 8 : 6
    return sizeof(imagestack) * n_copies
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
