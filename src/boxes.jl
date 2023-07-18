"""
   getboxstack(imagestack, coords, args::GetBoxesArgs) 
 
Cut out box regions from imagestack centered on coords.

# Arguments
- `imagestack`: Input image stack
- `coords`: Coords of box centers  
- `args`: Parameters

# Returns
- `boxstack`: Array with box crops from imagestack
"""
function getboxstack(imagestack, coords, kwargs::GetBoxesArgs)

    nboxes = size(coords, 1)
    boxstack = zeros(eltype(imagestack), kwargs.boxsize, kwargs.boxsize, nboxes)
    boxcoords = zeros(Int, nboxes, 3)

    # Cut out the boxes
    for i in 1:nboxes
        row = Int(coords[i, 1])
        col = Int(coords[i, 2])
        im = Int(coords[i, 3])
        box = @view boxstack[:, :, i]
        boxcoords[i,:] = fillbox!(box, imagestack, row, col, im, kwargs.boxsize)
    end

    return boxstack, boxcoords
end

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
function fillbox!(box::AbstractArray{<:Real,2}, imagestack::AbstractArray{<:Real,4}, row::Int, col::Int, im::Int, boxsize::Int)
    # Get the size of the image stack
    (nrows, ncols, ~, nimages) = size(imagestack)

    # Define the boundaries for the box
    row_min = row - boxsize ÷ 2
    row_max = row + boxsize ÷ 2
    col_min = col - boxsize ÷ 2
    col_max = col + boxsize ÷ 2

    # Adjust the boundaries if they are out of the imagestack boundaries
    if row_min < 1
        row_min = 1
        row_max = boxsize
    end

    if row_max > nrows
        row_min = nrows - boxsize + 1
        row_max = nrows
    end

    if col_min < 1
        col_min = 1
        col_max = boxsize
    end

    if col_max > ncols
        col_min = ncols - boxsize + 1
        col_max = ncols
    end

    box .= imagestack[row_min:row_max, col_min:col_max, 1, im]

    boxcoords = [row_min, col_min, im]
    return boxcoords
end