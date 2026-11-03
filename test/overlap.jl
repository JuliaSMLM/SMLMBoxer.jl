using Test, SMLMBoxer
using Random

# Independent allocating reference for the max-pool separation and visiting order.
function removeoverlap_reference(coords::Vector{Matrix{Float32}},
        kwargs::SMLMBoxer.GetBoxesArgs)
    k = max(3, floor(Int, kwargs.boxsize - kwargs.overlap))
    separation = cld(k, 2)
    for f in eachindex(coords)
        ncoords = size(coords[f], 1)
        keep = trues(ncoords)

        for i in 1:ncoords
            if keep[i]
                ci = coords[f][i, :]
                for j in (i+1):ncoords
                    if keep[j]
                        cj = coords[f][j, :]
                        distance = maximum(abs.(ci[1:2] - cj[1:2]))
                        if distance < separation
                            if ci[4] < cj[4]
                                keep[i] = false
                                break
                            else
                                keep[j] = false
                            end
                        end
                    end
                end
            end
        end
        coords[f] = coords[f][keep, :]
    end
    return vcat(coords...)
end

overlap_args(overlap) = SMLMBoxer.GetBoxesArgs(
    imagestack = zeros(Float32, 0, 0), overlap = overlap, backend = :cpu
)

function check_overlap_reference(coords, overlap)
    args = overlap_args(overlap)
    @test SMLMBoxer.removeoverlap(deepcopy(coords), args) ==
        removeoverlap_reference(deepcopy(coords), args)
end

@testset "Overlap removal matches reference" begin
    rng = MersenneTwister(0x5a17)
    counts = [0, 1, 2, 16, 64, 600, 800, rand(rng, 0:800, 12)..., 0]
    coords = [
        hcat(
            Float32.(rand(rng, 0:16, n, 2)), fill(Float32(f), n),
            Float32.(rand(rng, 1:5, n))
        ) for (f, n) in enumerate(counts)
    ]

    # At separation 2, A shares B's window and B shares C's, but A does not share C's.
    # Equal intensities, a brighter middle box, and a brighter last box exercise
    # the tie rule, skipped boxes, and the break when i is dropped.
    chains = [
        Float32[0 0 1 2; 1 0 1 2; 2 0 1 2],
        Float32[0 0 2 1; 1 0 2 3; 2 0 2 2],
        Float32[0 0 3 2; 1 0 3 1; 2 0 3 3],
        Float32[0 0 4 2; 1 0 4 1; -1 0 4 3],
    ]
    # Exact duplicates, diagonal and axial distance boundaries, and clusters.
    clusters = [
        Float32[0 0 1 2; 0 0 1 2; 1 1 1 2; 2 0 1 3; 3 0 1 1],
        Float32[0 0 2 2; 1 0 2 3; 0 1 2 3; 1 1 2 1; 0 0 2 4],
    ]

    for overlap in (0, 1, sqrt(2.0), 2, 2.5, 3)
        check_overlap_reference(coords, overlap)
        check_overlap_reference(chains, overlap)
        check_overlap_reference(clusters, overlap)
        check_overlap_reference([zeros(Float32, 0, 4)], overlap)
        check_overlap_reference([zeros(Float32, 0, 4) for _ in 1:4], overlap)
    end

    # Noninteger Float32 positions also obey Chebyshev separation.
    fractional = [hcat(rand(rng, Float32, 100, 2) .* 10, ones(Float32, 100, 2))]
    for overlap in (sqrt(2.0), 2.5)
        check_overlap_reference(fractional, overlap)
    end

    @test SMLMBoxer.removeoverlap(deepcopy(chains), overlap_args(3)) ==
        Float32[0 0 1 2; 2 0 1 2; 1 0 2 3; 0 0 3 2; 2 0 3 3; -1 0 4 3]

    # The boundary is strict, for both odd and even max-pool windows.
    diagonal = [Float32[0 0 1 2; 1 1 1 2]]
    @test SMLMBoxer.removeoverlap(deepcopy(diagonal), overlap_args(3)) ==
        Float32[0 0 1 2]
    boundary = [Float32[0 0 1 2; 2 2 1 2]]
    for overlap in (3, 4)
        @test SMLMBoxer.removeoverlap(deepcopy(boundary), overlap_args(overlap)) ==
            boundary[1]
    end
end

@testset "Overlap removal allocations" begin
    # Keep all 600 boxes so every pair is visited and output sizes are stable.
    coords = [hcat(Float32.(1:4:2400), zeros(Float32, 600), ones(Float32, 600, 2))]
    args = overlap_args(0)
    SMLMBoxer.removeoverlap(deepcopy(coords), args)
    input = deepcopy(coords)
    @test (@allocated SMLMBoxer.removeoverlap(input, args)) < 1_000_000
end
