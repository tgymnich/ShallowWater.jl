mutable struct Block1D
    h::Array{Float32,1}
    hu::Array{Float32,1}
    hv::Array{Float32,1}
end

mutable struct WavePropagationBlock
    # grid size: number of cells (incl. ghost layer in x and y direction:
    nx::Int
    ny::Int	# < size of Cartesian arrays in y-direction
    # mesh size dx and dy:
    dx::Float32	#<  mesh size of the Cartesian grid in x-direction
    dy::Float32	#<  mesh size of the Cartesian grid in y-direction

    # define arrays for unknowns: 
    # h (water level) and u,v (velocity in x and y direction)
    # hd, ud, and vd are respective CUDA arrays on GPU
    h::Array{Float32,2}	# < array that holds the water height for each element
    hu::Array{Float32,2} # < array that holds the x-component of the momentum for each element (water height h multiplied by velocity in x-direction)
    hv::Array{Float32,2}  # < array that holds the y-component of the momentum for each element (water height h multiplied by velocity in y-direction)
    b::Array{Float32,2}  # < array that holds the bathymetry data (sea floor elevation) for each element
    
    # type of boundary conditions at LEFT, RIGHT, TOP, and BOTTOM boundary
    boundary::Dict{BoundaryEdge, BoundaryType}
    # for CONNECT boundaries: pointer to connected neighbour block
    neighbour::Array{WavePropagationBlock,1}

    # maximum time step allowed to ensure stability of the method
    maxTimestep::Float32

    # offset of current block
    offsetX::Float32	#  < x-coordinate of the origin (left-bottom corner) of the Cartesian grid
    offsetY::Float32	# < y-coordinate of the origin (left-bottom corner) of the Cartesian grid

    hd::CuArray{Float32,2}
    hud::CuArray{Float32,2}
    hvd::CuArray{Float32,2}
    bd::CuArray{Float32,2}
	
    # separate memory to hold bottom and top ghost and copy layer in main memory allowing non-strided access
    bottomLayer::Array{Float32,1}
    topLayer::Array{Float32,1}
    bottomGhostLayer::Block1D
    bottomCopyLayer::Block1D
    topGhostLayer::Block1D
    topCopyLayer::Block1D
    
    # and resp. memory on the CUDA device:
    bottomLayerDevice::CuArray{Float32,1}
    topLayerDevice::CuArray{Float32,1}


    # holds the net-updates for the water height (wave propagating to the left).
    hNetUpdatesLeftD::CuArray{Float32,2}
    # holds the net-updates for the water height (wave propagating to the right).
    hNetUpdatesRightD::CuArray{Float32,2}

    # holds the net-updates for the momentum in x-direction (wave propagating to the left).
    huNetUpdatesLeftD::CuArray{Float32,2}
    # holds the net-updates for the momentum in x-direction (wave propagating to the right).
    huNetUpdatesRightD::CuArray{Float32,2}

    # holds the net-updates for the water height (wave propagating to the top).
    hNetUpdatesBelowD::CuArray{Float32,2}
    # holds the net-updates for the water height (wave propagating to the bottom).
    hNetUpdatesAboveD::CuArray{Float32,2}

    # holds the net-updates for the momentum in y-direction (wave propagating to the top).
    hvNetUpdatesBelowD::CuArray{Float32,2}
    # holds the net-updates for the momentum in y-direction (wave propagating to the bottom).
    hvNetUpdatesAboveD::CuArray{Float32,2}

    function WavePropagationBlock(nX, nY, dX, dY)
        h = Array{Float32,2}(undef, nX+2, nY+2)
        hu = Array{Float32,2}(undef, nX+2, nY+2)
        hv = Array{Float32,2}(undef, nX+2, nY+2)
        b = Array{Float32,2}(undef, nX+2, nY+2)
        boundary = Dict(LEFT=>PASSIVE, RIGHT=>PASSIVE, TOP=>PASSIVE, BOTTOM=>PASSIVE)
        neighbour = Array{WavePropagationBlock,1}(undef, 4)
        maxTimestep = 0
        offsetX = 0
        offsetY = 0
        hd = CuArray{Float32,2}(undef, nX+2, nY+2)
        hud = CuArray{Float32,2}(undef, nX+2, nY+2)
        hvd = CuArray{Float32,2}(undef, nX+2, nY+2)
        bd = CuArray{Float32,2}(undef, nX+2, nY+2)
        bottomLayer = Array{Float32,1}(undef, (nX+2)*6)
        topLayer = Array{Float32,1}(undef, (nX+2)*6)

        size = nX+2
        bottomGhostLayer = @views Block1D(bottomLayer[1:size],bottomLayer[size+1:2size],bottomLayer[2size+1:3size])
        bottomCopyLayer = @views Block1D(bottomLayer[3size+1:4size],bottomLayer[4size+1:5size],bottomLayer[5size+1:6size])
        topGhostLayer = @views Block1D(topLayer[1:size],topLayer[size+1:2size], topLayer[2size+1:3size])
        topCopyLayer = @views Block1D(topLayer[3size+1:4size],topLayer[4size+1:5size],topLayer[5size+1:6size])

        bottomLayerDevice = CuArray{Float32,1}(undef, (nX+2)*6)
        topLayerDevice = CuArray{Float32,1}(undef, (nX+2)*6)
        hNetUpdatesLeftD = CuArray{Float32,2}(undef, nX+1, nY+1)
        hNetUpdatesRightD = CuArray{Float32,2}(undef, nX+1, nY+1)
        huNetUpdatesLeftD = CuArray{Float32,2}(undef, nX+1, nY+1)
        huNetUpdatesRightD = CuArray{Float32,2}(undef, nX+1, nY+1)
        hNetUpdatesBelowD = CuArray{Float32,2}(undef, nX+1, nY+1)
        hNetUpdatesAboveD = CuArray{Float32,2}(undef, nX+1, nY+1)
        hvNetUpdatesBelowD = CuArray{Float32,2}(undef, nX+1, nY+1)
        hvNetUpdatesAboveD = CuArray{Float32,2}(undef, nX+1, nY+1)

        if nX % TILE_SIZE != 0
            error("WARNING: nx not a multiple of TILE_SIZE  -> will lead to crashes!") 
        end
        
        if nY % TILE_SIZE != 0
            error("WARNING: ny not a multiple of TILE_SIZE  -> will lead to crashes!") 
        end

        new(nX, nY, dX, dY, h, hu, hv, b, boundary, neighbour, maxTimestep, offsetX, offsetY, hd, hud, hvd, bd, bottomLayer, topLayer,
            bottomGhostLayer, bottomCopyLayer, topGhostLayer, topCopyLayer, bottomLayerDevice, topLayerDevice, hNetUpdatesLeftD,
            hNetUpdatesRightD, huNetUpdatesLeftD, huNetUpdatesRightD, hNetUpdatesBelowD, hNetUpdatesAboveD, hvNetUpdatesBelowD, hvNetUpdatesAboveD)
    end

end


"""
Initializes the unknowns and bathymetry in all grid cells according to the given SWE_Scenario.

In the case of multiple SWE_Blocks at this point, it is not clear how the boundary conditions
should be set. This is because an isolated SWE_Block doesn't have any in information about the grid.
Therefore the calling routine, which has the information about multiple blocks, has to take care about setting
the right boundary conditions.

@param scenario scenario, which is used during the setup.
@param multipleBlocks are there multiple SWE_blocks?
"""
function initScenario(block, scenario, offsetX, offsetY, multipleBlocks=false)
	block.offsetX = offsetX
	block.offsetY = offsetY

    # initialize water height and discharge
    for i = 1:block.nx+2
        for j = 1:block.ny+2
            x = offsetX + (i - 0.5) * block.dx
            y = offsetY + (j - 0.5) * block.dy
            @inbounds block.h[j,i] = getWaterHeight(scenario,x,y)
            @inbounds block.hu[j,i] = getVeloc_u(scenario,x,y) * block.h[j,i]
            @inbounds block.hv[j,i] = getVeloc_v(scenario,x,y) * block.h[j,i] 
        end
    end

    # initialize bathymetry
    for i = 1:block.nx+2
        for j = 1:block.ny+2
            @inbounds block.b[j,i] = getBathymetry(scenario,offsetX + (i - 0.5) * block.dx, offsetY + (j - 0.5) * block.dy)
        end
    end

    # in the case of multiple blocks the calling routine takes care about proper boundary conditions.
    if multipleBlocks == false
        # obtain boundary conditions for all four edges from scenario
        setBoundaryType(block, LEFT, getBoundaryType(scenario,LEFT))
        setBoundaryType(block, RIGHT, getBoundaryType(scenario,RIGHT))
        setBoundaryType(block, BOTTOM, getBoundaryType(scenario,BOTTOM))
        setBoundaryType(block, TOP, getBoundaryType(scenario,TOP))
    end

    # perform update after external write to variables 
    synchAfterWrite(block)
end

"""
return reference to water height unknown h
"""
function getWaterHeight(block) 
    return block.h 
end

"""
return reference to discharge unknown hu
"""
function getDischarge_hu(block)
    synchDischargeBeforeRead(block)
    return block.hu
end

"""
return reference to discharge unknown hv
"""
function getDischarge_hv(block) 
    synchDischargeBeforeRead(block)
    return block.hv
end

"""
return reference to bathymetry unknown b
"""
function getBathymetry(block) 
    synchBathymetryBeforeRead(block)
    return block.b
end

"""
set the values of all ghost cells depending on the specifed 
boundary conditions
"""
function setBoundaryConditions(block)
    # Fill ghost layer corner cells
    @cuda kernelHdBufferEdges(block.hd)

    if block.boundary[LEFT] == PASSIVE || block.boundary[LEFT] == CONNECT
        # nothing to be done: 
        # ghost values are copied by SWE_BlockCUDA::synchGhostLayerAfterWrite(...)
    else
        @cuda threads=(1,TILE_SIZE) blocks=(1,Int(block.ny/TILE_SIZE)) kernelLeftBoundary(block.hd,block.hud,block.hvd,block.boundary[LEFT])
    end

    if block.boundary[RIGHT] == PASSIVE || block.boundary[RIGHT] == CONNECT
        # nothing to be done: 
        # ghost values are copied by SWE_BlockCUDA::synchGhostLayerAfterWrite(...)
    else
        @cuda threads=(1,TILE_SIZE) blocks=(1,Int(block.ny/TILE_SIZE)) kernelRightBoundary(block.hd,block.hud,block.hvd,block.boundary[RIGHT])
    end

    if block.boundary[BOTTOM] == CONNECT
        # set ghost layer data in auxiliary data structure for ghost layer:
        block.bottomGhostLayer.h = block.neighbour[BOTTOM].h
        block.bottomGhostLayer.hu = block.neighbour[BOTTOM].hu
        block.bottomGhostLayer.hv = block.neighbour[BOTTOM].hv
    elseif block.boundary[BOTTOM] == PASSIVE
        # copy ghost layer data from buffer bottomLayerDevice
        # into bottom ghost layer of unknowns
        @cuda threads=(TILE_SIZE,1) blocks=(Int(block.nx/TILE_SIZE),1) kernelBottomGhostBoundary(block.hd,block.hud,block.hvd,block.bottomLayerDevice,block.nx)
    else 
        # set simple boundary conditions (OUTFLOW, WALL) by resp. kernel:
        @cuda threads=(TILE_SIZE,1) blocks=(Int(block.nx/TILE_SIZE),1) kernelBottomBoundary(block.hd,block.hud,block.hvd,block.boundary[BOTTOM])
    end
    
    if block.boundary[TOP] == CONNECT
        # set ghost layer data in auxiliary data structure for ghost layer:
        block.topGhostLayer.h = block.neighbour[TOP].h
        block.topGhostLayer.hu = block.neighbour[TOP].hu
        block.topGhostLayer.hv = block.neighbour[TOP].hv
    elseif block.boundary[TOP] == PASSIVE
        # copy ghost layer data from buffer bottomLayerDevice
        # into bottom ghost layer of unknowns
        @cuda threads=(TILE_SIZE,1) blocks=(Int(block.nx/TILE_SIZE),1) kernelTopGhostBoundary(block.hd,block.hud,block.hvd,block.topLayerDevice,block.nx,block.ny)
    else
        # set simple boundary conditions (OUTFLOW, WALL) by resp. kernel: 
        @cuda threads=(TILE_SIZE,1) blocks=(Int(block.nx/TILE_SIZE),1) kernelTopBoundary(block.hd,block.hud,block.hvd,block.boundary[TOP])
    end
end

"""
synchronise the ghost layer content of h, hu, and hv in main memory 
with device memory and auxiliary data structures, i.e. transfer 
memory from main/auxiliary memory into device memory
"""
function synchGhostLayerAfterWrite(block)
    if block.boundary[LEFT] == PASSIVE || block.boundary[LEFT] == CONNECT
        # transfer h, hu, and hv from left ghost layer to resp. device memory
        block.hd[:,1] = @views block.h[:,1]
        block.hud[:,1] = @views block.hu[:,1]
        block.hvd[:,1] = @views block.hv[:,1]
    end

    if block.boundary[RIGHT] == PASSIVE || block.boundary[RIGHT] == CONNECT
        # transfer h, hu, and hv from right ghost layer to resp. device memory
        block.hd[:,end-1] = @views block.h[:,end-1]
        block.hud[:,end-1] = @views block.hu[:,end-1]
        block.hvd[:,end-1] = @views block.hv[:,end-1]
    end

    # bottom and top boundary ghost layers are replicated (for efficiency reasons)
    # in the  memory regions starting from bottomLayer and topLayer
    # -> these need to be transfered to device memory
    if block.boundary[BOTTOM] == PASSIVE || block.boundary[BOTTOM] == CONNECT
        # transfer bottom ghost layer 
        # (3 arrays - h, hu, hv - of nx+2 floats, consecutive in memory)
        block.bottomLayerDevice = CuArray(block.bottomLayer)
    end

    if block.boundary[TOP] == PASSIVE || block.boundary[TOP] == CONNECT
        # transfer top ghost layer
        # (3 arrays - h, hu, hv - of nx+2 floats, consecutive in memory)
        block.topLayerDevice = CuArray(block.topLayer)
    end
end

"""
Update (for heterogeneous computing) variables h, hu, hv in copy layers
before an external access to the unknowns 
(only required for PASSIVE and CONNECT boundaries)
- copy (up-to-date) content from device memory into main memory
"""
function synchCopyLayerBeforeRead(block)
    # copy values in copy layers to main memory
    # left copy layer:
    block.h[:,1] = @views block.hd[:,1]
    block.hu[:,1] = @views block.hud[:,1]
    block.hv[:,1] = @views block.hvd[:,1]
    
    # right copy layer
    block.h[:,end-1] = @views block.hd[:,end-1]
    block.hu[:,end-1] = @views block.hud[:,end-1] 
    block.hv[:,end-1] = @views block.hvd[:,end-1] 

    # bottom copy layer
    if block.boundary[BOTTOM] == PASSIVE || block.boundary[BOTTOM] == CONNECT
        @cuda threads=(TILE_SIZE,1) blocks=(Int(block.nx/TILE_SIZE),1) kernelBottomCopyLayer(block.hd,block.hud,block.hvd,block.bottomLayerDevice,block.nx)
        block.bottomLayer = CuArray(block.bottomLayerDevice)
    end

    # top copy layer
    if block.boundary[TOP] == PASSIVE || block.boundary[TOP] == CONNECT
        @cuda threads=(TILE_SIZE,1) blocks=(Int(block.nx/TILE_SIZE),1) kernelTopCopyLayer(block.hd,block.hud,block.hvd,block.topLayerDevice,block.nx,block.ny)
        block.topLayer = CuArray(block.topLayerDevice)
    end
end

"""
register the row or column layer next to a boundary as a "copy layer",
from which values will be copied into the ghost layer or a neighbour;
@return	a SWE_Block1D object that contains row variables h, hu, and hv
"""
function registerCopyLayer(block, edge)
    # for TOP and BOTTOM layer, the implementation is identical to that in SWE_Block
    # for LEFT and RIGHT layer, separate layers are used that avoid strided copies 
    # when transferring memory between host and device memory
    
    if edge == LEFT
        @views return Block1D(block.h[:,2], block.hu[:,2], block.hv[:,2])
    elseif edge == RIGHT
        @views return Block1D(block.h[:,end-1], block.hu[:,end-1], block.hv[:,end-1])
    elseif edge == BOTTOM
        # transfer bottom ghost and copy layer to extra SWE_Block1D
        block.bottomGhostLayer.h .= @views block.h[:,1]
        block.bottomGhostLayer.hu .= @views block.hu[:,1]
        block.bottomGhostLayer.hv .= @views block.hv[:,1]
        block.bottomCopyLayer.h .= @views block.h[:,2]
        block.bottomCopyLayer.hu .= @views block.hu[:,2]
        block.bottomCopyLayer.hv .= @views block.hv[:,2]

        return block.bottomCopyLayer
    elseif case BND_TOP
        # transfer bottom ghost and copy layer to extra SWE_Block1D
        block.topGhostLayer.h .= @views block.h[:,end]
        block.topGhostLayer.hu .= @views block.hu[:,end]
        block.topGhostLayer.hv .= @views block.hv[:,1]
        block.topCopyLayer.h .= @views block.h[:,end-1]
        block.topCopyLayer.hu .= @views block.hu[:,end-1]
        block.topCopyLayer.hv .= @views block.hv[:,end-1]

        return block.topCopyLayer
    end
    return nothing
end

"""
"grab" the ghost layer at the specific boundary in order to set boundary values 
in this ghost layer externally. 
The boundary conditions at the respective ghost layer is set to PASSIVE, 
such that the grabbing program component is responsible to provide correct 
values in the ghost layer, for example by receiving data from a remote 
copy layer via MPI communication. 
@param	specified edge
@return	a SWE_Block1D object that contains row variables h, hu, and hv
"""
function grabGhostLayer(block, edge)
    # for TOP and BOTTOM layer, the implementation is identical to that in SWE_Block
    # for LEFT and RIGHT layer, separate layers are used that avoid strided copies 
    # when transferring memory between host and device memory
    boundary[edge] = PASSIVE
    if edge == LEFT
        @views return Block1D(block.h[:,1], block.hu[:,1], block.hv[:,1])
    elseif edge == RIGHT
        @views return Block1D(block.h[:,end], block.hu[:,end], block.hv[:,end])
    elseif edge == BOTTOM
        return block.bottomGhostLayer
    elseif edge == TOP
        return  block.topGhostLayer
    end
    return nothing
end


# protected member functions for memory model: 
# in case of temporary variables (especial in non-local memory, for 
# example on accelerators), the main variables h, hu, hv, and b 
# are not necessarily updated after each time step.
# The following methods are called to synchronise before or after 
# external read or write to the variables.

"""
Update all temporary and non-local (for heterogeneous computing) variables
after an external update of the main variables h, hu, hv, and b.
"""
function synchAfterWrite(block)
    # update h, hu, hv, b in device memory
    synchWaterHeightAfterWrite(block)
    synchDischargeAfterWrite(block)
    synchBathymetryAfterWrite(block)

    # update the auxiliary data structures for copy and ghost layers 
    # at bottom (and top, see below) boundaries
    #  -> only required for PASSIVE and CONNECT boundaries
    if block.boundary[BOTTOM] == PASSIVE || block.boundary[BOTTOM] == CONNECT
        # transfer bottom ghost and copy layer to extra SWE_Block1D
        block.bottomGhostLayer.h .= @views block.h[:,1]
        block.bottomGhostLayer.hu .= @views block.hu[:,1]
        block.bottomGhostLayer.hv .= @view block.hv[:,1]
        block.bottomCopyLayer.h .= @view block.h[:,2]
        block.bottomCopyLayer.hu .= @view block.hu[:,2]
        block.bottomCopyLayer.hv .= @view block.hv[:,2]
    end
  
    if block.boundary[TOP] == PASSIVE || block.boundary[TOP] == CONNECT
        # transfer bottom ghost and copy layer to extra SWE_Block1D
        block.topGhostLayer.h .= @view block.h[:,end]
        block.topGhostLayer.hu .= @view block.hu[:,end]
        block.topGhostLayer.hv .= @view block.hv[:,end]
        block.topCopyLayer.h .= @view block.h[:,end-1]
        block.topCopyLayer.hu .= @view block.hu[:,end-1]
        block.topCopyLayer.hv .= @view block.hv[:,end-1]
    end
end

"""
Update temporary and non-local (for heterogeneous computing) variables
after an external update of the water height h
"""
function synchWaterHeightAfterWrite(block)
    block.hd = CuArray(block.h)
end

"""
Update temporary and non-local (for heterogeneous computing) variables
after an external update of the discharge variables hu and hv
"""
function synchDischargeAfterWrite(block)
    block.hud = CuArray(block.hu)
    block.hvd = CuArray(block.hv)
end

"""
Update temporary and non-local (for heterogeneous computing) variables
after an external update of the bathymetry b
"""
function synchBathymetryAfterWrite(block)
    block.bd = CuArray(block.b) 
end

"""
Update the main variables h, hu, hv, and b before an external read access:
copies current content of the respective device variables hd, hud, hvd, bd
"""
function synchBeforeRead(block)
   synchWaterHeightBeforeRead(block)
   synchDischargeBeforeRead(block)
   synchBathymetryBeforeRead(block)
end

"""
Update temporary and non-local (for heterogeneous computing) variables
before an external access to the water height h
"""
function synchWaterHeightBeforeRead(block)
    block.h = Array(block.hd)
  
    # only required for visualisation: set values in corner ghost cells
    block.h[1,1] = block.h[2,2]
    block.h[end-1,1] = block.h[end-1,2]
    block.h[1,end-1] = block.h[2,end-1]
    block.h[end-1,end-1] = block.h[end-1,end-1]
end

"""
Update temporary and non-local (for heterogeneous computing) variables
before an external access to the discharge variables hu and hv
"""
function synchDischargeBeforeRead(block)
    block.hu = Array(block.hud)
    block.hv = Array(block.hvd)
end

"""
Update temporary and non-local (for heterogeneous computing) variables
before an external access to the bathymetry b
"""
function synchBathymetryBeforeRead(block)
    block.b = Array(block.bd)
end

"""
Compute a single global time step of a given time step width.
Remark: The user has to take care about the time step width. No additional check is done. The time step width typically available
after the computation of the numerical fluxes (hidden in this method).

First the net-updates are computed.
Then the cells are updated with the net-updates and the given time step width.
"""
function simulateTimestep(block, dT)
    computeNumericalFluxes(block)
    updateUnknowns(block, dT)
end

"""
perform forward-Euler time steps, starting with simulation time tStart,:
until simulation time tEnd is reached; 
device-global variables hd, hud, hvd are updated;
unknowns h, hu, hv in main memory are not updated.
Ghost layers and bathymetry sources are updated between timesteps.
intended as main simulation loop between two checkpoints
"""
function simulate(block, tStart, tEnd) 
    t = tStart
  
    while t < tEnd
        # set values in ghost cells:
        setGhostLayer(block)
     
        # Compute the numerical fluxes/net-updates in the wave propagation formulation.
        computeNumericalFluxes(block)

        # Update the unknowns with the net-updates.
        updateUnknowns(block, maxTimestep)
	 
	    t += maxTimestep
        println("Simulation at time t")
    end

    return t
end

"""
Set the boundary type for specific block boundary.
@param edge location of the edge relative to the SWE_block.
@param boundaryType type of the boundary condition.
"""
function setBoundaryType(block, edge, boundaryType)
	block.boundary[edge] = boundaryType

	if boundaryType == OUTFLOW || boundaryType == WALL
		# One of the boundary was changed to OUTFLOW or WALL
		# -> Update the bathymetry for this boundary
		setBoundaryBathymetry(block)
    end
end

"""
Sets the bathymetry on OUTFLOW or WALL boundaries.
Should be called every time a boundary is changed to a OUTFLOW or
WALL boundary or the bathymetry changes.
"""
function setBoundaryBathymetry(block)
    # set bathymetry values in the ghost layer, if necessary
	if block.boundary[LEFT] == OUTFLOW || block.boundary[LEFT] == WALL
        block.b[:,1] = @views block.b[:,2]
    end
	if block.boundary[RIGHT] == OUTFLOW || block.boundary[RIGHT] == WALL
        block.hd[:,end-1] = @views block.h[:,end]
    end
	if block.boundary[BOTTOM] == OUTFLOW || block.boundary[BOTTOM] == WALL
		block.b[:,1] = @views block.b[:,2]
	end
	if block.boundary[TOP] == OUTFLOW || block.boundary[TOP] == WALL
		block.b[:,end] = @views block.b[:,end-1]
	end


	# set corner values
    block.b[1,1] = block.b[2,2]
    block.b[end,1] = block.b[end-1,2]
    block.b[1,end] = block.b[2,end-1]
    block.b[end,end] = block.b[end-1,end-1]

    # synchronize after an external update of the bathymetry
	synchBathymetryAfterWrite(block)
end

"""
Compute the numerical fluxes (net-update formulation here) on all edges.

The maximum wave speed is computed within the net-updates kernel for each CUDA-block.
To finalize the method the Thrust-library is called, which does the reduction over all blockwise maxima.
In the wave speed reduction step the actual cell width in x- and y-direction is not taken into account.
"""
function computeNumericalFluxes(block)
    # definition of one CUDA-block. Typical size are 8*8 or 16*16
    dimGrid = (Int(block.nx/TILE_SIZE), Int(block.ny/TILE_SIZE))
    dimBlock = (TILE_SIZE, TILE_SIZE)

    # 2D array which holds the blockwise maximum wave speeds
    sizeMaxWaveSpeeds = (dimGrid[1] + 1) * (dimGrid[2] + 1) - 1
    maximumWaveSpeedsD = CuArray{Float32}(undef, sizeMaxWaveSpeeds)

    @cuda threads=dimBlock blocks=dimGrid computeNetUpdatesKernel_inlined(block.hd, block.hud, block.hvd, block.bd, block.hNetUpdatesLeftD, block.hNetUpdatesRightD, 
                                                                  block.huNetUpdatesLeftD, block.huNetUpdatesRightD, block.hNetUpdatesBelowD,
                                                                  block.hNetUpdatesAboveD, block.hvNetUpdatesBelowD, block.hvNetUpdatesAboveD,
                                                                  maximumWaveSpeedsD)

    # compute the "remaining" net updates (edges "simulation domain"/"top ghost layer" and "simulation domain"/"right ghost layer" 
    # edges between cell nx and ghost layer nx+1

    dimRightBlock = (TILE_SIZE, 1)
    dimRightGrid = (1, Int(block.ny/TILE_SIZE))
    
    @cuda threads=dimRightBlock blocks=dimRightGrid computeNetUpdatesKernel_inlined(block.hd, block.hud, block.hvd, block.bd, block.hNetUpdatesLeftD, block.hNetUpdatesRightD, block.huNetUpdatesLeftD, block.huNetUpdatesRightD,
                block.hNetUpdatesBelowD, block.hNetUpdatesAboveD, block.hvNetUpdatesBelowD, block.hvNetUpdatesAboveD, maximumWaveSpeedsD, dimGrid[1], 0)

    # edges between cell ny and ghost layer ny+1
    dimTopBlock = (1, TILE_SIZE)
    dimTopGrid = (Int(block.nx/TILE_SIZE), 1)

    @cuda threads=dimTopBlock blocks=dimTopGrid computeNetUpdatesKernel_inlined(block.hd, block.hud, block.hvd, block.bd, block.hNetUpdatesLeftD, block.hNetUpdatesRightD, block.huNetUpdatesLeftD, block.huNetUpdatesRightD,
                block.hNetUpdatesBelowD, block.hNetUpdatesAboveD, block.hvNetUpdatesBelowD, block.hvNetUpdatesAboveD, maximumWaveSpeedsD, 0, dimGrid[2])

    # Finalize (max reduction of the maximumWaveSpeeds-array.)
    
    # get the result from the device
    maximumWaveSpeed = maximum(maximumWaveSpeedsD)

    block.maxTimestep = 0.4 * min(block.dx / maximumWaveSpeed, block.dy / maximumWaveSpeed)
end


"""
Update the cells with a given global time step.
@param deltaT time step size.
"""
function updateUnknowns(block, deltaT)
    # definition of one CUDA-block. Typical size are 8*8 or 16*16
    dimBlock = (TILE_SIZE, TILE_SIZE)
    dimGrid = (Int(block.nx/TILE_SIZE), Int(block.ny/TILE_SIZE))

    # compute the update width.
    updateWidthX = deltaT / block.dx
    updateWidthY = deltaT / block.dy

    # update the unknowns (global time step)
    @cuda threads=dimBlock blocks=dimGrid updateUnknownsKernel(block.hNetUpdatesLeftD, block.hNetUpdatesRightD, block.huNetUpdatesLeftD, block.huNetUpdatesRightD, block.hNetUpdatesBelowD, block.hNetUpdatesAboveD,
                block.hvNetUpdatesBelowD, block.hvNetUpdatesAboveD, block.hd, block.hud, block.hvd, updateWidthX, updateWidthY)

    #synchCopyLayerBeforeRead(block)
end

"""
set the values of all ghost cells depending on the specifed boundary conditions;
if the ghost layer replicates the variables of a remote SWE_Block, 
the values are copied
"""
function setGhostLayer(block)
    # call to virtual function to set ghost layer values 
    setBoundaryConditions(block)
    # left boundary
    if block.boundary[LEFT] == CONNECT
        block.h[:,1] = block.neighbour[LEFT].h
        block.hu[:,1] = block.neighbour[LEFT].hu
        block.hv[:,1] = block.neighbour[LEFT].hv
    end
  
    # right boundary
    if block.boundary[RIGHT] == CONNECT
        block.h[:,end] = block.neighbour[RIGHT].h
        block.hu[:,end] = block.neighbour[RIGHT].hu
        block.hv[:,end] = block.neighbour[RIGHT].hv
    end

    # bottom boundary
    if block.boundary[BOTTOM] == CONNECT
        block.h[1,:] = block.neighbour[BOTTOM].h
        block.hu[1,:] = block.neighbour[BOTTOM].hu
        block.hv[1,:] = block.neighbour[BOTTOM].hv
    end

    # top boundary
    if block.boundary[TOP] == CONNECT
        block.h[end,:] = block.neighbour[TOP].h
        block.hu[end,:] = block.neighbour[TOP].hu
        block.hv[end,:] = block.neighbour[TOP].hv
    end

    # synchronize the ghost layers (for PASSIVE and CONNECT conditions)
    # with accelerator memory
    synchGhostLayerAfterWrite(block)
end