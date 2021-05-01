function create_file(filename, nx, ny, t)
    isfile(filename) && rm(filename)

    gatts = Dict(
        "Conventions" => "CF-1.5",
        "title" => "Computed tsunami solution",
        "history" => "SWE Julia",
        "institution" => "Technische Universitaet Muenchen, Department of Informatics, Chair of Scientific Computing",
        "source" => "Bathymetry and displacement data.",
        "references" => "https://github.com/tgymnich/ShallowWater.jl"
    )

    nccreate(filename, "time", "time", t+1, atts=Dict("longname" => "Time", "units" => "seconds since simulation start"), gatts=gatts, t=NC_FLOAT)

    nccreate(filename, "h", "x", collect(0.0:nx-1), "y", collect(0.0:ny-1), "time", atts=Dict("longname" => "Water Height"), t=NC_FLOAT)
    nccreate(filename, "hu", "x", "y", "time", atts=Dict("longname" => "X-component of the Momentum"), t=NC_FLOAT)
    nccreate(filename, "hv", "x", "y", "time", atts=Dict("longname" => "Y-component of the Momentum "), t=NC_FLOAT)
    nccreate(filename, "b", "x", "y", atts=Dict("longname" => "Bathymetry Data"), t=NC_FLOAT)
end