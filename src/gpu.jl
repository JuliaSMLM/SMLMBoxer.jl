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
    find_best_gpu() -> Int

Find the GPU with the most free memory and switch to it.

Returns the device index (0-based). On single-GPU systems, returns 0 immediately.

Uses NVML to query free memory on each device without creating CUDA contexts,
avoiding `cuDevicePrimaryCtxRetain` OOM errors under multi-process contention.
Only calls `CUDA.device!()` once on the selected device.

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

    # Query memory via NVML (no CUDA context needed, safe under contention)
    best, maxfree = 0, 0
    for i in 0:(n-1)
        info = CUDA.NVML.memory_info(CUDA.NVML.Device(i))
        if info.free > maxfree
            maxfree, best = info.free, i
        end
    end

    CUDA.device!(best)
    @info "Selected GPU $best with $(Base.format_bytes(maxfree)) free"
    return best
end

"""
    poll_gpu_nvml(required_bytes; timeout=30.0, poll=0.5, on_wait=nothing) -> (Bool, Int)

Poll ALL GPUs via NVML until one has sufficient free memory and low contention.
No CUDA context creation needed - safe under multi-process contention.

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
            dev = CUDA.NVML.Device(i)
            info = CUDA.NVML.memory_info(dev)

            # Not enough memory on this device
            info.free < required_with_margin && continue

            # Check contention: other processes on this GPU
            procs = CUDA.NVML.compute_processes(dev)
            other_procs = length(procs) - (getpid() in keys(procs) ? 1 : 0)

            if other_procs > 0
                rates = CUDA.NVML.utilization_rates(dev)
                rates.compute > 0.9 && continue  # GPU saturated, skip
            end

            # This GPU is available - track best (most free memory)
            if info.free > best_free
                best_free = info.free
                best_device = i
            end
        end

        if best_device >= 0
            return (true, best_device)
        end

        # Callback for progress reporting
        if on_wait !== nothing
            # Report best available across all GPUs for visibility
            max_free = maximum(CUDA.NVML.memory_info(CUDA.NVML.Device(i)).free for i in 0:(n-1))
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

Layer 1 (NVML polling): Scans all GPUs via NVML without creating CUDA contexts.
Polls through timeout with jittered backoff. First GPU with sufficient free memory
and low contention wins. Safe under multi-process contention.

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
- `:auto` - NVML poll for device with timeout, falls back to (:cpu, -1) with warning
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

        # NVML poll to find available device (no CUDA context)
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

        # NVML poll all GPUs - no CUDA context creation
        available, device_id = poll_gpu_nvml(required_bytes; timeout=auto_timeout, on_wait=on_wait)
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
