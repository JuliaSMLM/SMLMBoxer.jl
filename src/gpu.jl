"""
GPU utilities for SMLMBoxer.

Provides backend selection and GPU memory waiting functionality following
the JuliaSMLM convention:
- backend: :cpu | :gpu | :auto
- auto_timeout: seconds to wait in Auto mode (default 30.0)
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
    wait_for_gpu_memory(required_bytes; timeout=30.0, poll=0.5, on_wait=nothing) -> Bool

Wait until GPU has sufficient available memory.

# Arguments
- `required_bytes`: Minimum bytes needed (with safety margin applied internally)
- `timeout`: Maximum seconds to wait (default 30.0, use Inf for unlimited)
- `poll`: Seconds between checks (default 0.5)
- `on_wait`: Optional callback(elapsed, available, required) called each poll

# Returns
- `true` if memory became available, `false` if timeout reached

# Notes
- Calls CUDA.reclaim() before each check to free cached memory
- Applies 1.5x safety margin for fragmentation
- Uses jittered polling to desync competing processes
"""
function wait_for_gpu_memory(required_bytes::Integer;
        timeout::Real = 30.0,
        poll::Real = 0.5,
        on_wait = nothing)

    # Safety margin for fragmentation (free_memory can be fragmented)
    required_with_margin = required_bytes * 1.5

    deadline = time() + timeout
    start = time()

    while time() < deadline
        # Try to free cached memory
        CUDA.reclaim()

        available = CUDA.free_memory()
        if available >= required_with_margin
            return true
        end

        # Callback for progress reporting
        if on_wait !== nothing
            on_wait(time() - start, available, required_bytes)
        end

        # Jittered sleep to desync competing allocators
        sleep(poll * (1.0 + 0.2 * (rand() - 0.5)))
    end

    return false
end

"""
    select_backend(backend::Symbol, required_bytes;
                   auto_timeout=30.0, gpu_timeout=Inf, on_wait=nothing) -> Symbol

Select compute backend with GPU memory waiting.

# Arguments
- `backend`: :cpu, :gpu, or :auto
- `required_bytes`: Estimated GPU memory needed for processing
- `auto_timeout`: Max wait for :auto mode before CPU fallback (default 30.0)
- `gpu_timeout`: Max wait for :gpu mode (default Inf - wait forever)
- `on_wait`: Optional callback(elapsed, available, required)

# Returns
- `:cpu` or `:gpu` - the actual backend to use

# Behavior
- `:cpu` - Always returns :cpu, no waiting
- `:gpu` - Requires GPU, waits up to gpu_timeout, errors if unavailable/timeout
- `:auto` - Tries GPU with auto_timeout wait, falls back to CPU with warning

# Example
```julia
backend = select_backend(:auto, 1_000_000_000;  # 1GB needed
    auto_timeout = 30.0,
    on_wait = (e, a, r) -> @info "Waiting..." elapsed=round(e,digits=1))
```
"""
function select_backend(backend::Symbol, required_bytes::Integer;
        auto_timeout::Real = 30.0,
        gpu_timeout::Real = Inf,
        on_wait = nothing)

    backend in (:cpu, :gpu, :auto) || error("backend must be :cpu, :gpu, or :auto")

    if backend == :cpu
        return :cpu

    elseif backend == :gpu
        # User explicitly requested GPU - must have it
        if !has_cuda() || !CUDA.functional()
            error("GPU backend requested but CUDA is not functional")
        end

        # Wait for memory (possibly forever)
        if !wait_for_gpu_memory(required_bytes; timeout=gpu_timeout, on_wait=on_wait)
            error("GPU memory not available after $(gpu_timeout)s. " *
                  "Required: $(Base.format_bytes(required_bytes)), " *
                  "Available: $(Base.format_bytes(CUDA.free_memory()))")
        end
        return :gpu

    else  # :auto
        # Try GPU if available, fall back to CPU
        if !has_cuda() || !CUDA.functional()
            return :cpu
        end

        if wait_for_gpu_memory(required_bytes; timeout=auto_timeout, on_wait=on_wait)
            return :gpu
        else
            @warn "GPU memory not available after $(auto_timeout)s, using CPU. " *
                  "Required: $(Base.format_bytes(required_bytes)), " *
                  "Available: $(Base.format_bytes(CUDA.free_memory()))"
            return :cpu
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

Variance-weighted (SCMOSCamera): 10x input size
- Additional workspace for per-pixel variance weighting
"""
function estimate_gpu_memory(imagestack::AbstractArray, camera)
    # Memory multiplier matches _getboxes_impl
    n_copies = camera isa SCMOSCamera ? 10 : 6
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
    n_copies = camera isa SCMOSCamera ? 10 : 6
    return height * width * sizeof(Float32) * n_copies
end
