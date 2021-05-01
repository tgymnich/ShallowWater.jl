module ShallowWater

using CUDA
using Printf
using NetCDF
using ProgressMeter

include("Boundary.jl")
include("Scenario.jl")
include("WavePropagationBlock.jl")
include("WavePropagationKernels.jl")
include("FWaveKernels.jl")
include("Kernels.jl")
include("Writer.jl")

const zeroTol = 0.0000001
const dryTol = 1
const gravity = 9.81
const TILE_SIZE = 16


function run(gridSizeX, gridSizeY, numberOfCheckPoints=40, filename="/work_fast/swe.nc")
    nX = gridSizeX
    nY = gridSizeY

    scneario = DamBreakScenario()

    dX = (getBoundaryPos(scneario, RIGHT) - getBoundaryPos(scneario, LEFT)) / nX
    dY = (getBoundaryPos(scneario, TOP) - getBoundaryPos(scneario, BOTTOM)) / nY

    waveBlock = WavePropagationBlock(nX, nY, dX, dY)

    originX = getBoundaryPos(scneario, LEFT)
    originY = getBoundaryPos(scneario, BOTTOM)

    initScenario(waveBlock, scneario, originX, originY)
    
    endSimulation = endSimulationTime(scneario)
    checkPoints = Array{Float32,1}(collect(0:numberOfCheckPoints))
    checkPoints .*= endSimulation / numberOfCheckPoints

    # simulation time.
    t = 0.0

    create_file(filename, nX, nY, numberOfCheckPoints)

    # loop over checkpoints
    @showprogress 1 "Simulating..." for c = 1:numberOfCheckPoints + 1
        # do time steps until next checkpoint is reached
        while t < checkPoints[c]
            # set values in ghost cells:
            setGhostLayer(waveBlock)

            # compute numerical flux on each edge
            computeNumericalFluxes(waveBlock)

            # maximum allowed time step width.
            maxTimeStepWidth = waveBlock.maxTimestep

            # update the cell values
            updateUnknowns(waveBlock, maxTimeStepWidth)

            # update simulation time with time step width.
            t += maxTimeStepWidth
        end

        h = Array(waveBlock.hd[2:end-1,2:end-1])
        hu = Array(waveBlock.hud[2:end-1,2:end-1])
        hv = Array(waveBlock.hvd[2:end-1,2:end-1])

        ncwrite([t], filename, "time", start=[c], count=[1])
        ncwrite(reshape(h, (size(h)...,1)), filename, "h", start=[1,1,c], count=[nX,nY,1])
        ncwrite(reshape(hu, (size(hu)...,1)), filename, "hu", start=[1,1,c], count=[nX,nY,1])
        ncwrite(reshape(hv, (size(hv)...,1)), filename, "hv", start=[1,1,c], count=[nX,nY,1])
    end
    b = waveBlock.b[2:end-1,2:end-1]
    ncwrite(b, filename, "b", start=[1,1], count=[nX,nY])
end


end # module
