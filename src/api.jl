# Simple API overview functionality for Julia packages

# Determine the package root directory
function _package_root()
    # Get the directory of the current file (api.jl)
    src_dir = dirname(@__FILE__)
    # Go up one level to the package root
    return abspath(joinpath(src_dir, ".."))
end

# Try both api_overview.md and api.md for backward compatibility
function _find_api_file()
    root = _package_root()
    # Check api_overview.md first (preferred)
    api_overview_path = joinpath(root, "api_overview.md")
    isfile(api_overview_path) && return api_overview_path
    # Fall back to api.md for backward compatibility
    api_path = joinpath(root, "api.md")
    isfile(api_path) && return api_path
    # Return api_overview.md path for error message if neither exists
    return api_overview_path
end

# Path to the API documentation file
const _API_PATH = _find_api_file()

# Load the content of the API file if it exists
const _API_CONTENT = if isfile(_API_PATH)
    read(_API_PATH, String)
else
    """
    API documentation not found.

    Expected files (in order of preference):
    1. api_overview.md
    2. api.md (backward compatibility)

    Expected location: $(_package_root())
    """
end

"""
$(_API_CONTENT)

---
`api()` returns this documentation as a plain `String`.
"""
function api()
    return _API_CONTENT
end
