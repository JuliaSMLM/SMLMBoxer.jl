"""
Test edge corner calculation to verify no off-by-one errors
"""

using SMLMBoxer
using SMLMData
using Test
using Printf

println("="^80)
println("Edge Corner Validation Test")
println("="^80)
println()

# Create test image with known peaks at various positions
img_size = 50
image = zeros(Float32, img_size, img_size)

# Test peaks at different distances from edges
test_peaks = [
    (5, 5, "near top-left corner"),
    (5, 25, "near top edge, centered horizontally"),
    (5, 45, "near top-right corner"),
    (25, 5, "near left edge, centered vertically"),
    (25, 25, "center"),
    (25, 45, "near right edge, centered vertically"),
    (45, 5, "near bottom-left corner"),
    (45, 25, "near bottom edge, centered horizontally"),
    (45, 45, "near bottom-right corner"),
]

# Place peaks
for (row, col, _) in test_peaks
    image[row, col] = 100.0
end

# Setup camera
camera = IdealCamera(1:img_size, 1:img_size, 0.1f0)

# Detect with specific parameters to ensure we get all peaks
boxsize = 11
roi_batch = getboxes(image, camera;
    sigma_small = 1.0,
    sigma_large = 2.0,
    minval = 10.0,
    boxsize = boxsize,
    overlap = 0.5,  # Tight to avoid merging
    backend = :cpu
)

println("Detected $(length(roi_batch)) peaks (expected $(length(test_peaks)))")
println()

# Verify each detection
println("Peak Position Validation:")
println("-"^80)
@printf("%-30s %10s %10s %12s %12s\n", "Location", "Peak(r,c)", "Corner(x,y)", "Expected", "Status")
println("-"^80)

for (peak_row, peak_col, desc) in test_peaks
    # Find the ROI that contains this peak
    # The peak should be within the box extracted

    found = false
    for i in 1:length(roi_batch)
        corner_x = roi_batch.corners[1, i]  # col
        corner_y = roi_batch.corners[2, i]  # row

        # Check if peak falls within this box
        # Box covers [corner_y:corner_y+boxsize-1, corner_x:corner_x+boxsize-1]
        if peak_row >= corner_y && peak_row <= corner_y + boxsize - 1 &&
           peak_col >= corner_x && peak_col <= corner_x + boxsize - 1

            # Peak found in this box
            found = true

            # Calculate expected corner for centered box
            expected_row_min = max(1, peak_row - boxsize ÷ 2)
            expected_col_min = max(1, peak_col - boxsize ÷ 2)

            # For near-far edge: clamp to fit
            if expected_row_min + boxsize - 1 > img_size
                expected_row_min = img_size - boxsize + 1
            end
            if expected_col_min + boxsize - 1 > img_size
                expected_col_min = img_size - boxsize + 1
            end

            expected_corner = (expected_col_min, expected_row_min)  # (x, y)
            actual_corner = (corner_x, corner_y)

            # Verify peak position within box
            peak_in_box_row = peak_row - corner_y + 1  # Local position (1-indexed)
            peak_in_box_col = peak_col - corner_x + 1

            # Get actual peak value in extracted box
            box_value = roi_batch.data[peak_in_box_row, peak_in_box_col, i]

            status = actual_corner == expected_corner ? "✓" : "✗ MISMATCH"

            @printf("%-30s (%3d,%3d) (%3d,%3d) (%3d,%3d) %12s\n",
                    desc, peak_row, peak_col,
                    actual_corner[1], actual_corner[2],
                    expected_corner[1], expected_corner[2],
                    status)

            # Detailed check if mismatch
            if actual_corner != expected_corner
                println("  ERROR: Corner mismatch!")
                println("    Expected corner: $(expected_corner)")
                println("    Actual corner: $(actual_corner)")
                println("    Difference: $(actual_corner .- expected_corner)")
            end

            # Verify peak is in the box
            if box_value ≈ 100.0
                # Peak correctly captured
            else
                println("  WARNING: Peak value in box = $(box_value), expected 100.0")
                println("    Peak position in box: ($peak_in_box_row, $peak_in_box_col)")
                println("    Box value range: [$(minimum(roi_batch.data[:,:,i])), $(maximum(roi_batch.data[:,:,i]))]")
            end

            break
        end
    end

    if !found
        println(@sprintf("%-30s (%3d,%3d) %10s %12s %12s\n",
                desc, peak_row, peak_col, "NOT FOUND", "", "✗ MISSING"))
    end
end

println("-"^80)
println()
println("Edge Corner Test Complete")
println("="^80)
