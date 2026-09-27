# Boxes near the image edge are shifted inward so every box lies fully inside the image and
# still contains its peak: corners are the centered corner clamped to [1, img_size - boxsize + 1].
using Test, SMLMBoxer, SMLMData

img_size = 50
boxsize = 11
peaks = [(r, c) for r in (5, 25, 45) for c in (5, 25, 45)]  # corners, edge centers, center

image = zeros(Float32, img_size, img_size)
for (r, c) in peaks
    image[r, c] = 100.0
end
camera = IdealCamera(1:(img_size + 1), 1:(img_size + 1), 0.1f0)

(roi_batch, _) = getboxes(
    image, camera;
    sigma_small = 1.0, sigma_large = 2.0, minval = 10.0,
    boxsize = boxsize, overlap = 0.5, backend = :cpu
)

@test length(roi_batch) == length(peaks)

clampcorner(p) = clamp(p - boxsize ÷ 2, 1, img_size - boxsize + 1)
for (r, c) in peaks
    i = findfirst(1:length(roi_batch)) do i
        roi_batch.y_corners[i] <= r < roi_batch.y_corners[i] + boxsize &&
            roi_batch.x_corners[i] <= c < roi_batch.x_corners[i] + boxsize
    end
    @test i !== nothing
    i === nothing && continue
    @test (roi_batch.x_corners[i], roi_batch.y_corners[i]) == (clampcorner(c), clampcorner(r))
    @test roi_batch.data[r - roi_batch.y_corners[i] + 1, c - roi_batch.x_corners[i] + 1, i] == 100.0f0
end
