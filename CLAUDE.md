# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

SMLMBoxer.jl is a particle/blob detection and ROI extraction library for single-molecule localization microscopy (SMLM). It finds emitters in image stacks using difference-of-Gaussians (DoG) filtering, with GPU acceleration and sCMOS camera variance-weighted filtering.

Part of the [JuliaSMLM](https://github.com/JuliaSMLM) ecosystem. SMLMBoxer handles detection; downstream fitting is done by GaussMLE.jl. The full pipeline is orchestrated by SMLMAnalysis.jl.

## Commands

```bash
# Run tests
julia --project -e 'using Pkg; Pkg.test()'

# Run tests on GPU server
ssh descent "cd ~/julia_shared_dev/SMLMBoxer && julia --project -e 'using Pkg; Pkg.test()'"

# Quick REPL check
julia --project -e 'using SMLMBoxer; println("loaded")'
```

## Architecture

### Detection Pipeline

```
imagestack → reshape → DoG filter → local maxima → overlap removal → box extraction → ROIBatch
```

### Source Files (src/)

| File | Role |
|------|------|
| `SMLMBoxer.jl` | Module definition, imports, exports |
| `types.jl` | `BoxerConfig`, `BoxesInfo`, `GetBoxesArgs` (internal); PSF-to-pixel conversion logic |
| `interface.jl` | `getboxes()` public API (2 calling conventions); batching logic; GPU retry loop |
| `filter.jl` | DoG filtering — standard path via NNlib/cuDNN, variance-weighted path via KernelAbstractions |
| `localmax.jl` | Local maxima via NNlib max pooling; sparse GPU coordinate extraction |
| `coords.jl` | Coordinate extraction (`maxima2coords`, `_gpu_maxima2coords`); `removeoverlap()` |
| `boxes.jl` | ROI patch cutting (`getboxstack`, `fillbox!`); handles image boundary cases |
| `gpu.jl` | NVML polling (context-free), CUDA memory waiting, backend selection, memory estimation |
| `api.jl` | `SMLMBoxer.api()` loads `api_overview.md` |

### Two Filtering Paths

The DoG filter routes based on camera type:

- **Standard DoG** (IdealCamera or no camera): NNlib convolution, cuDNN on GPU. Fast, simple.
- **Variance-weighted DoG** (SCMOSCamera): KernelAbstractions custom kernels. Per-pixel inverse-variance weighting. Same code runs on CPU and GPU via KA backend dispatch.

### GPU Contention Handling

Two-layer approach in `gpu.jl` + `interface.jl`:

1. **NVML polling** (`poll_gpu_nvml`) — scans all GPUs without creating a CUDA context. Checks free memory (1.5x safety margin), process count, and utilization. Jittered sleep to desync competing processes.
2. **Runtime try/catch** — if GPU processing fails (OOM), reclaims memory pool and retries with NVML polling for remaining timeout.

Backend modes: `:auto` (GPU with CPU fallback), `:gpu` (GPU required, error on timeout), `:cpu`.

### Memory Model

- Standard DoG: ~6x input size (imagestack + two convolutions + DoG result + pooling + GC margin)
- Variance-weighted sCMOS: ~8x input size (additional variance map copies)
- `_process_with_batching()` in `interface.jl` splits large stacks into memory-safe batches

### Key Types

- **`BoxerConfig`** — User config. Two parameter interfaces: PSF-aware (`psf_sigma`, `min_photons` in physical units) or direct (`sigma_small`, `sigma_large`, `minval` in pixels).
- **`BoxesInfo`** — Processing metadata (backend used, timing, batch info).
- **`GetBoxesArgs`** — Internal. Converts PSF-aware params to pixel params in its constructor.
- **`ROIBatch`** / **`SingleROI`** — From SMLMData.jl, re-exported. Output containers.

### SMLMData Integration

Imports `AbstractCamera`, `IdealCamera`, `SCMOSCamera`, `ROIBatch`, `SingleROI` from SMLMData.jl. Camera type determines the filtering path and whether calibration data (readnoise, gain, QE) is extracted per-ROI.

## Conventions

- All images converted to Float32 at entry
- NNlib expects `(H, W, C, N)` layout — imagestack is reshaped to `(H, W, 1, nframes)`
- `x_corners` = column positions, `y_corners` = row positions (follows SMLMData convention)
- `device_id = -1` means CPU; 0-based for GPUs
- KernelAbstractions kernels use `@index(Global, NTuple)` for device-agnostic indexing
