using SMLMBoxer
using Test

@testset "SMLMBoxer.jl" begin

    # Test image with two bright peaks
    image = zeros(Float32, 100, 100)
    image[20, 50] = 10
    image[30, 60] = 10

    # Get boxes with overlap set to 5
    # Create GetBoxArgs
    args = SMLMBoxer.GetBoxesArgs(
        imagestack=image,
        boxsize=5,
        overlap=3.0,
        sigma_small=1.0,
        sigma_large=2.0,
        minval=0.1,
        use_gpu=false
    )

    boxes, coords = SMLMBoxer.getboxes(
        imagestack=args.imagestack,
        boxsize=args.boxsize,
        overlap=args.overlap,
        sigma_small=args.sigma_small,
        sigma_large=args.sigma_large,
        minval=args.minval,
        use_gpu=args.use_gpu
    )

    # Should detect two peaks
    @test size(boxes) == (5, 5, 2)

    # Verify correct box location
    @test coords[1, :] ≈ [20, 50, 1]
    @test coords[2, :] ≈ [30, 60, 1]


    # Test image with two close bright peaks
    image = zeros(Float32, 100, 100)
    image[20, 50] = 20
    image[21, 51] = 10

    # Create GetBoxArgs
    args = SMLMBoxer.GetBoxesArgs(
        imagestack=image,
        boxsize=5,
        overlap=3.0,
        sigma_small=1.0,
        sigma_large=2.0,
        minval=0.1,
        use_gpu=false
    )


    # Get boxes with overlap set to 5
    boxes, coords = SMLMBoxer.getboxes(
        imagestack=args.imagestack,
        boxsize=args.boxsize,
        overlap=args.overlap,
        sigma_small=args.sigma_small,
        sigma_large=args.sigma_large,
        minval=args.minval,
        use_gpu=args.use_gpu
    )

    # Should detect two peaks
    @test size(boxes) == (5, 5, 1)

    # Verify correct box location
    @test coords[1, :] ≈ [20, 50, 1]

    # Verify correct box location and intensity - not correct because intentisy is after DoG
    # @test coords[1, :] ≈ [20, 50, 1, 2]


end