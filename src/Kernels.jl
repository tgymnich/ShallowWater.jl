function kernelHdBufferEdges(hd)	
    @inbounds hd[1,1] = hd[2,2]
    @inbounds hd[end,1] = hd[end-1,2]
    @inbounds hd[1,end] = hd[2,end-1]
    @inbounds hd[end,end] = hd[end-1,end-1]
    return
end


"""
Kernel to set left boundary layer for conditions WALL & OUTFLOW
blockIdx.y and threadIdx.y loop over the boundary elements
SWE_Block size ny is assumed to be a multiple of the TILE_SIZE
"""
function kernelLeftBoundary(hd, hud, hvd, bound)
    j = 2 + TILE_SIZE * (blockIdx().y - 1) + (threadIdx().y - 1)
    @inbounds hd[j,1] = hd[j,2]
    @inbounds hud[j,1] = (bound == WALL) ? -hud[j,2] : hud[j,2]
    @inbounds hvd[j,1] = hvd[j,2]
    return
end


"""
Kernel to set right boundary layer for conditions WALL & OUTFLOW
blockIdx.y and threadIdx.y loop over the boundary elements
SWE_Block size ny is assumed to be a multiple of the TILE_SIZE
"""
function kernelRightBoundary(hd, hud, hvd, bound)
    j = 2 + TILE_SIZE * (blockIdx().y - 1) + (threadIdx().y - 1)
    
    @inbounds hd[j,end] = hd[j,end-1]
    @inbounds hud[j,end] = (bound == WALL) ? -hud[j,end-1] : hud[j,end-1]
    @inbounds hvd[j,end] = hvd[j,end-1]
    return
end


"""
Kernel to set bottom boundary layer for conditions WALL & OUTFLOW
blockIdx.x and threadIdx.x loop over the boundary elements
SWE_Block size ny is assumed to be a multiple of the TILE_SIZE
"""

function kernelBottomBoundary(hd, hud, hvd, bound)
    i = 2 + TILE_SIZE + (blockIdx().x - 1) + (threadIdx().x - 1)

    @inbounds hd[1,i] = hd[2,i]
    @inbounds hud[1,i] = hud[2,i]
    @inbounds hvd[1,i] = (bound == WALL) ? -hvd[2,i] : hvd[2,i]
    return
end


"""
Kernel to set bottom boundary layer for conditions WALL & OUTFLOW
blockIdx.x and threadIdx.x loop over the boundary elements
"""
function kernelTopBoundary(hd, hud, hvd, bound)
    i = 2 + TILE_SIZE * (blockIdx().x - 1) + (threadIdx().x - 1)
    
    @inbounds hd[end,i] = hd[end-1,i]
    @inbounds hud[end,i] = hud[end-1,i]
    @inbounds hvd[end,i] = (bound == WALL) ? -hvd[end-1,i] : hvd[end-1,i]
    return
end


"""
Kernel to set bottom boundary layer according to the external 
ghost layer status (conditions PASSIVE and CONNECT)
blockIdx.x and threadIdx.x loop over the boundary elements.
Note that diagonal elements are currently not copied!
SWE_Block size ny is assumed to be a multiple of the TILE_SIZE
"""
function kernelBottomGhostBoundary(hd, hud, hvd, bottomGhostLayer, nx)
    i = 2 + TILE_SIZE * (blockIdx().x - 1) + (threadIdx().x - 1)

    @inbounds hd[1,i]  = bottomGhostLayer[i]
    @inbounds hud[1,i] = bottomGhostLayer[(nx+2)+i]
    @inbounds hvd[1,i] = bottomGhostLayer[2*(nx+2)+i]
    return
end


"""
Kernel to set top boundary layer according to the external 
ghost layer status (conditions PASSIVE and CONNECT)
blockIdx.x and threadIdx.x loop over the boundary elements
Note that diagonal elements are currently not copied!
SWE_Block size ny is assumed to be a multiple of the TILE_SIZE
"""
function kernelTopGhostBoundary(hd, hud, hvd, topGhostLayer, nx, ny)
    i = 2 + TILE_SIZE * (blockIdx().x - 1) + (threadIdx().x - 1)
  
    @inbounds hd[end,i] = topGhostLayer[i]
    @inbounds hud[end,i] = topGhostLayer[(nx+2)+i]
    @inbounds hvd[end,i] = topGhostLayer[2*(nx+2)+i]
    return
end


"""
Kernel to update bottom copy layer according 
(for boundary conditions PASSIVE and CONNECT)
blockIdx.x and threadIdx.x loop over the boundary elements.
Note that diagonal elements are currently not copied!
SWE_Block size ny is assumed to be a multiple of the TILE_SIZE
"""
function kernelBottomCopyLayer(hd, hud, hvd, bottomCopyLayer, nx)
    i = 2 + TILE_SIZE * (blockIdx().x - 1) + (threadIdx().x - 1)
  
    @inbounds bottomCopyLayer[i] = hd[2,i]
    @inbounds bottomCopyLayer[(nx+2)+i] = hud[2,i]
    @inbounds bottomCopyLayer[2*(nx+2)+i] = hvd[2,i]
    return
end


"""
Kernel to set top boundary layer according to the external 
ghost layer status (conditions PASSIVE and CONNECT)
blockIdx.x and threadIdx.x loop over the boundary elements
Note that diagonal elements are currently not copied!
SWE_Block size ny is assumed to be a multiple of the TILE_SIZE
"""
function kernelTopCopyLayer(hd, hud, hvd, topCopyLayer, nx, ny)
    i = 2 + TILE_SIZE * (blockIdx().x - 1) + (threadIdx().x - 1)
    
    @inbounds opCopyLayer[i] = hd[end-1,i]
    @inbounds topCopyLayer[(nx+2)+i] = hud[end-1,i]
    @inbounds topCopyLayer[2*(nx+2)+i] = hvd[end-1,i]
    return
end
