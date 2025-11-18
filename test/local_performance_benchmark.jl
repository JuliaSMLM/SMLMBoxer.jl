"""
Local Performance Benchmark for SMLMBoxer.jl

This benchmark compares CPU vs GPU performance across various configurations:
- Multiple image sizes (128×128 to 512×512)
- Different frame counts (10-50 frames)
- Various spot densities (sparse to dense)

The benchmark measures:
- Processing throughput (images/second)
- Detection accuracy (found vs expected spots)
- GPU speedup over CPU

Only runs in local testing environments (not on GitHub Actions).
"""

# All using statements must be in runtests.jl

# Configuration
const WARMUP_ITERATIONS = 2
const BENCHMARK_RUNS = 5

# Benchmark result structures
struct BenchmarkConfig
    label::String
    nx::Int
    ny::Int
    nframes::Int
    nspots_per_frame::Int
end

struct BenchmarkResult
    config::BenchmarkConfig
    expected_spots::Int
    cpu_found::Int
    gpu_found::Int
    cpu_time::Float64
    gpu_time::Float64
    cpu_throughput::Float64
    gpu_throughput::Float64
    speedup::Float64
end

"""
    create_synthetic_image(nx, ny, nframes, nspots_per_frame; sigma=2.0, intensity=100.0)

Create synthetic image stack with known number of Gaussian spots.
"""
function create_synthetic_image(nx, ny, nframes, nspots_per_frame; sigma=2.0, intensity=100.0)
    image = zeros(Float32, nx, ny, nframes)
    expected_coords = []

    for frame in 1:nframes
        for _ in 1:nspots_per_frame
            # Random position with border margin
            x = rand(10:(nx-10))
            y = rand(10:(ny-10))

            # Add Gaussian spot
            for i in max(1, x-10):min(nx, x+10)
                for j in max(1, y-10):min(ny, y+10)
                    r2 = (i-x)^2 + (j-y)^2
                    image[i, j, frame] += intensity * exp(-r2 / (2*sigma^2))
                end
            end

            push!(expected_coords, (x, y, frame))
        end
    end

    return image, length(expected_coords)
end

"""
    benchmark_getboxes(image; use_gpu=false, nwarmup=2, nruns=5)

Benchmark getboxes with warmup and multiple runs.
"""
function benchmark_getboxes(image; use_gpu=false, nwarmup=WARMUP_ITERATIONS, nruns=BENCHMARK_RUNS)
    # Warmup
    for _ in 1:nwarmup
        result = getboxes(image;
            boxsize=7,
            overlap=3.0,
            sigma_small=1.0,
            sigma_large=2.0,
            minval=5.0,
            use_gpu=use_gpu
        )
    end

    # Timed runs
    times = Float64[]
    ndetections = 0

    for _ in 1:nruns
        GC.gc()  # Force garbage collection
        if CUDA.functional() && use_gpu
            CUDA.reclaim()  # Reclaim GPU memory
        end

        t0 = time()
        result = getboxes(image;
            boxsize=7,
            overlap=3.0,
            sigma_small=1.0,
            sigma_large=2.0,
            minval=5.0,
            use_gpu=use_gpu
        )
        t1 = time()

        push!(times, t1 - t0)
        ndetections = result.metadata.ndetections
    end

    return median(times), ndetections
end

"""
    run_single_benchmark(config::BenchmarkConfig, has_cuda::Bool) -> BenchmarkResult

Run benchmark for a single configuration.
"""
function run_single_benchmark(config::BenchmarkConfig, has_cuda::Bool)
    # Create synthetic data
    image, expected_spots = create_synthetic_image(
        config.nx, config.ny, config.nframes, config.nspots_per_frame
    )

    # CPU benchmark
    cpu_time, cpu_found = benchmark_getboxes(image; use_gpu=false)
    cpu_throughput = config.nframes / cpu_time

    # GPU benchmark
    if has_cuda
        gpu_time, gpu_found = benchmark_getboxes(image; use_gpu=true)
        gpu_throughput = config.nframes / gpu_time
        speedup = cpu_time / gpu_time
    else
        gpu_time = NaN
        gpu_found = 0
        gpu_throughput = NaN
        speedup = NaN
    end

    return BenchmarkResult(
        config, expected_spots, cpu_found, gpu_found,
        cpu_time, gpu_time, cpu_throughput, gpu_throughput, speedup
    )
end

"""
    print_benchmark_table(results::Vector{BenchmarkResult}, has_cuda::Bool)

Print nicely formatted benchmark results table.
"""
function print_benchmark_table(results::Vector{BenchmarkResult}, has_cuda::Bool)
    println()
    println("="^110)
    println("SMLMBOXER PERFORMANCE BENCHMARK - CPU vs GPU")
    println("="^110)
    println("Configuration: Warmup=$WARMUP_ITERATIONS, Benchmark=$BENCHMARK_RUNS runs (median time)")
    println("Spot parameters: σ=2.0 pixels, intensity=100 ADU, min detection=5.0 ADU")
    println("Processing: boxsize=7, overlap=3.0, DoG filter (σ_small=1.0, σ_large=2.0)")
    println("="^110)
    println()

    # Table header
    if has_cuda
        println("┌────────────────┬──────────────┬──────────┬──────────┬──────────┬──────────────┬──────────────┬─────────┐")
        println("│ Configuration  │ Image Size   │ Expected │ CPU      │ GPU      │ CPU          │ GPU          │ Speedup │")
        println("│                │              │ Spots    │ Found    │ Found    │ (images/sec) │ (images/sec) │         │")
    else
        println("┌────────────────┬──────────────┬──────────┬──────────┬──────────────┐")
        println("│ Configuration  │ Image Size   │ Expected │ CPU      │ CPU          │")
        println("│                │              │ Spots    │ Found    │ (images/sec) │")
    end

    if has_cuda
        println("├────────────────┼──────────────┼──────────┼──────────┼──────────┼──────────────┼──────────────┼─────────┤")
    else
        println("├────────────────┼──────────────┼──────────┼──────────┼──────────────┤")
    end

    for r in results
        size_str = "$(r.config.nx)×$(r.config.ny)×$(r.config.nframes)"

        if has_cuda
            @printf("│ %-14s │ %-12s │ %8d │ %8d │ %8d │ %12.2f │ %12.2f │ %6.2fx │\n",
                    r.config.label, size_str, r.expected_spots, r.cpu_found,
                    r.gpu_found, r.cpu_throughput, r.gpu_throughput, r.speedup)
        else
            @printf("│ %-14s │ %-12s │ %8d │ %8d │ %12.2f │\n",
                    r.config.label, size_str, r.expected_spots, r.cpu_found, r.cpu_throughput)
        end
    end

    if has_cuda
        println("└────────────────┴──────────────┴──────────┴──────────┴──────────┴──────────────┴──────────────┴─────────┘")
    else
        println("└────────────────┴──────────────┴──────────┴──────────┴──────────────┘")
    end
    println()

    # Summary statistics
    println("SUMMARY:")
    println("─"^110)

    if has_cuda
        speedups = [r.speedup for r in results if !isnan(r.speedup)]
        println(@sprintf("  Mean GPU speedup:   %.2fx (range: %.2fx - %.2fx)",
                        mean(speedups), minimum(speedups), maximum(speedups)))
        println(@sprintf("  Median GPU speedup: %.2fx", median(speedups)))
    end

    # Detection accuracy
    cpu_accuracies = [(r.cpu_found / r.expected_spots * 100) for r in results]
    println(@sprintf("  CPU detection rate: %.1f%% ± %.1f%%", mean(cpu_accuracies), std(cpu_accuracies)))

    if has_cuda
        gpu_accuracies = [(r.gpu_found / r.expected_spots * 100) for r in results]
        println(@sprintf("  GPU detection rate: %.1f%% ± %.1f%%", mean(gpu_accuracies), std(gpu_accuracies)))

        # Check if CPU and GPU give same results
        same_detections = all(r.cpu_found == r.gpu_found for r in results)
        if same_detections
            println("  ✓ CPU and GPU detected identical number of spots in all tests")
        else
            println("  ⚠ CPU and GPU detection counts differ in some tests")
            for r in results
                if r.cpu_found != r.gpu_found
                    println(@sprintf("    - %s: CPU=%d, GPU=%d (diff=%d)",
                                    r.config.label, r.cpu_found, r.gpu_found, r.gpu_found - r.cpu_found))
                end
            end
        end
    end

    println("="^110)
    println()
end

"""
    run_comprehensive_benchmark() -> Vector{BenchmarkResult}

Main benchmark runner - tests multiple configurations.
"""
function run_comprehensive_benchmark()
    has_cuda = CUDA.functional()

    # Define benchmark configurations
    configs = [
        BenchmarkConfig("Small sparse",    128, 128, 10, 20),
        BenchmarkConfig("Small dense",     128, 128, 10, 50),
        BenchmarkConfig("Medium sparse",   256, 256, 10, 50),
        BenchmarkConfig("Medium dense",    256, 256, 10, 100),
        BenchmarkConfig("Large sparse",    512, 512, 10, 100),
        BenchmarkConfig("Large dense",     512, 512, 10, 200),
        BenchmarkConfig("Med 50-frame",    256, 256, 50, 50),
        BenchmarkConfig("Large 20-frame",  512, 512, 20, 100),
    ]

    println("\nRunning benchmarks on $(length(configs)) configurations...")
    if has_cuda
        println("GPU detected: ", CUDA.name(CUDA.device()))
        println("GPU memory: ", round(CUDA.totalmem(CUDA.device())/1e9, digits=2), " GB")
    else
        println("No GPU detected - CPU only")
    end
    println()

    # Run benchmarks
    results = BenchmarkResult[]

    for (i, config) in enumerate(configs)
        print("[$i/$(length(configs))] $(config.label) ($(config.nx)×$(config.ny)×$(config.nframes), $(config.nspots_per_frame) spots/frame)... ")

        result = run_single_benchmark(config, has_cuda)
        push!(results, result)

        println("✓")
    end

    # Print results table
    print_benchmark_table(results, has_cuda)

    return results
end
