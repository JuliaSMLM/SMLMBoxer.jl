"""
   getboxstack(imagestack, coords, args::GetBoxesArgs)

Cut out box regions from imagestack centered on coords. After clamping corners, keep
the brightest filtered detection per corner and frame; ties keep the earlier detection.

# Arguments
- `imagestack`: Input image stack
- `coords`: Matrix of (row, col, frame, filtered intensity)
- `args`: Parameters

# Returns
- `boxstack`: Array with box crops from imagestack
- `boxcoords`: Upper left corners of boxes
"""
function getboxstack(imagestack, coords, kwargs::GetBoxesArgs)

    nrows, ncols = size(imagestack, 1), size(imagestack, 2)
    winners = Dict{NTuple{3,Int},Int}()
    keep = trues(size(coords, 1))
    for i in axes(coords, 1)
        row, col, frame = Int(coords[i, 1]), Int(coords[i, 2]), Int(coords[i, 3])
        key = (box_corner(row, kwargs.boxsize, nrows),
            box_corner(col, kwargs.boxsize, ncols), frame)
        previous = get(winners, key, 0)
        if previous == 0
            winners[key] = i
        elseif coords[i, 4] > coords[previous, 4]
            keep[previous] = false
            winners[key] = i
        else
            keep[i] = false
        end
    end
    coords = coords[keep, :]
    nboxes = size(coords, 1)
    boxstack = zeros(eltype(imagestack), kwargs.boxsize, kwargs.boxsize, nboxes)
    boxcoords = zeros(Int, nboxes, 3)

    # Cut out the boxes
    for i in 1:nboxes
        row = Int(coords[i, 1])
        col = Int(coords[i, 2])
        im = Int(coords[i, 3])
        box = @view boxstack[:, :, i]
        row_min, col_min, _ = fillbox!(box, imagestack, row, col, im, kwargs.boxsize)
        boxcoords[i,:] = [row_min, col_min, im]
    end

    return boxstack, boxcoords
end

"""Clamp a box corner to the image, using the same rule for odd and even sizes."""
box_corner(center::Int, boxsize::Int, imagesize::Int) =
    clamp(center - (boxsize - 1) ÷ 2, 1, imagesize - boxsize + 1)

"""
   fillbox!(box, imagestack, row, col, im, boxsize)

Fill a box with a crop from the imagestack.
 
# Arguments
- `box`: Array to fill with box crop
- `imagestack`: Input image stack 
- `row`, `col`, `im`: Coords for crop
- `boxsize`: Size of box

# Returns
- `boxcoords`: Upper Left corners of boxes N x (row, col, im)
"""
function fillbox!(box::AbstractArray{<:Real,2}, imagestack::AbstractArray{<:Real,4},
        row::Int, col::Int, im::Int, boxsize::Int)
    # Get the size of the image stack
    nrows, ncols = size(imagestack, 1), size(imagestack, 2)

    # Define the boundaries for the box
    row_min = box_corner(row, boxsize, nrows)
    col_min = box_corner(col, boxsize, ncols)
    row_max = row_min + boxsize - 1
    col_max = col_min + boxsize - 1

    box .= imagestack[row_min:row_max, col_min:col_max, 1, im]

    boxcoords = [row_min, col_min, im]
    return boxcoords
end
