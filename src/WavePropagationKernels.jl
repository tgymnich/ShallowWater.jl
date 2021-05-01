"""
The compute net-updates kernel calls the solver for a defined CUDA-Block and does a reduction over the computed wave speeds within this block.
Remark: In overall we have nx+1 / ny+1 edges. Therefore the edges "simulation domain"/"top ghost layer" and "simulation domain"/"right ghost layer"
will not be computed in a typical call of the function:
To reduce the effect of branch-mispredictions the kernel provides optional offsets, which can be used to compute the missing edges.
"""
function computeNetUpdatesKernel(h, hu, hv, b, 
    hNetUpdatesLeftD, hNetUpdatesRightD, 
    huNetUpdatesLeftD, huNetUpdatesRightD, 
    hNetUpdatesBelowD, hNetUpdatesAboveD, 
    hvNetUpdatesBelowD, hvNetUpdatesAboveD, maximumWaveSpeeds,
    offsetX=0, offsetY=0, blockOffSetX=0, blockOffSetY=0)

    #initialize shared maximum wave speed with zero
    maxWaveSpeedShared = @cuStaticSharedMem(Float32, (TILE_SIZE,TILE_SIZE))    
    maxWaveSpeedShared[threadIdx().x, threadIdx().y] = 0

    # index (l_cellIndexI,l_cellIndexJ) of the cell lying on the right side of the edge/above the edge where the thread works on.
    # initialize the indices l_cellIndexI and l_cellIndexJ with the given offset
    # compute (l_cellIndexI,l_cellIndexJ) for the cell lying right/above of the edge
    cellIndexI = offsetX + blockDim().y * (blockIdx().x - 1) + (threadIdx().y - 1) + 2
    cellIndexJ = offsetY + blockDim().x * (blockIdx().y - 1) + (threadIdx().x - 1) + 2

    # array which holds the thread local net-updates.
    #netUpdates = MArray{Tuple{5},Float32}(undef)
    netUpdates = @cuStaticSharedMem(Float32, (5))    


    # Computation of horizontal net-updates
    if offsetY == 0
        # compute the net-updates
        fWaveComputeNetUpdates(9.81, h[cellIndexJ,cellIndexI-1], h[cellIndexJ,cellIndexI], hu[cellIndexJ,cellIndexI-1], hu[cellIndexJ,cellIndexI], b[cellIndexJ,cellIndexI-1], b[cellIndexJ,cellIndexI], netUpdates)
        
        # store the horizontal net-updates (thread local net-updates -> device net-updates)
        hNetUpdatesLeftD[cellIndexJ,cellIndexI-1] = netUpdates[1]
        hNetUpdatesRightD[cellIndexJ,cellIndexI-1] = netUpdates[2]
        huNetUpdatesLeftD[cellIndexJ,cellIndexI-1] = netUpdates[3]
        huNetUpdatesRightD[cellIndexJ,cellIndexI-1] = netUpdates[4]

        # store the maximum wave speed in the shared array
        maxWaveSpeedShared[threadIdx().x, threadIdx().y] = netUpdates[5]
    end

    # synchronize the threads before the vertical edges (optimization)
    sync_threads()

    # Computation of vertical net-updates
    if offsetX == 0
        # compute the net-updates
        fWaveComputeNetUpdates(9.81, h[cellIndexJ-1,cellIndexI], h[cellIndexJ,cellIndexI], hv[cellIndexJ-1,cellIndexI], hv[cellIndexJ,cellIndexI], b[cellIndexJ-1,cellIndexI], b[cellIndexJ,cellIndexI], netUpdates)

        # store the vertical net-updates (thread local net-updates -> device net-updates)
        hNetUpdatesBelowD[cellIndexJ-1,cellIndexI]  = netUpdates[1]
        hNetUpdatesAboveD[cellIndexJ-1,cellIndexI]  = netUpdates[2]
        hvNetUpdatesBelowD[cellIndexJ-1,cellIndexI] = netUpdates[3]
        hvNetUpdatesAboveD[cellIndexJ-1,cellIndexI] = netUpdates[4]

        # store the maximum wave speed in the shared array
        maxWaveSpeedShared[threadIdx().x, threadIdx().y] = max(maxWaveSpeedShared[threadIdx().x, threadIdx().y], netUpdates[5])
    end

    # Compute the maximum observed wave speed
    sync_threads()

    # initialize reduction block size with the original block size
    reductionBlockDimY = blockDim().y
    reductionBlockDimX = blockDim().x

    # do the reduction
    while reductionBlockDimY != 1 || reductionBlockDimX != 1
        # reduction partner for a thread
        reductionPartner = 1

        # split the block in the x-direction (size in x-dir. > 1) or y-direction (size in x-dir. == 1, size in y-dir. > 1)
        if reductionBlockDimX != 1
            reductionBlockDimX >>= 1
            reductionPartner = computeOneDPositionKernel((threadIdx().y-1), (threadIdx().x-1) + reductionBlockDimX, blockDim().x)
        elseif reductionBlockDimY != 1
            reductionBlockDimY >>= 1
            reductionPartner = computeOneDPositionKernel((threadIdx().y-1) + reductionBlockDimY, (threadIdx().x-1), blockDim().x)
        end

    
        if (threadIdx().y-1) < reductionBlockDimY && (threadIdx().x-1) < reductionBlockDimX
            #execute the reduction routine (maximum)
            maxWaveSpeedShared[threadIdx().x, threadIdx().y] = max(maxWaveSpeedShared[threadIdx().x, threadIdx().y], maxWaveSpeedShared[reductionPartner])
        end

        sync_threads()
    end

    if (threadIdx().y-1) == 0 && (threadIdx().x-1) == 0
        # Position of the maximum wave speed in the global device array.
        maxWaveSpeedDevicePosition = computeOneDPositionKernel(blockOffSetX + (blockIdx().x-1), blockOffSetY + (blockIdx().y-1), max(blockOffSetY + 1, gridDim().y + 1))

        # write the block local maximum wave speed to the device array
        maximumWaveSpeeds[maxWaveSpeedDevicePosition] = maxWaveSpeedShared[1]
    end

    return
end

function computeNetUpdatesKernel_inlined(h, hu, hv, b, 
    hNetUpdatesLeftD, hNetUpdatesRightD, 
    huNetUpdatesLeftD, huNetUpdatesRightD, 
    hNetUpdatesBelowD, hNetUpdatesAboveD, 
    hvNetUpdatesBelowD, hvNetUpdatesAboveD, maximumWaveSpeeds,
    offsetX=0, offsetY=0, blockOffSetX=0, blockOffSetY=0)

    #initialize shared maximum wave speed with zero
    maxWaveSpeedShared = @cuStaticSharedMem(Float32, (TILE_SIZE,TILE_SIZE))    
    @inbounds maxWaveSpeedShared[threadIdx().x, threadIdx().y] = 0

    # index (l_cellIndexI,l_cellIndexJ) of the cell lying on the right side of the edge/above the edge where the thread works on.
    # initialize the indices l_cellIndexI and l_cellIndexJ with the given offset
    # compute (l_cellIndexI,l_cellIndexJ) for the cell lying right/above of the edge
    cellIndexI = offsetX + blockDim().y * (blockIdx().x - 1) + (threadIdx().y - 1) + 2
    cellIndexJ = offsetY + blockDim().x * (blockIdx().y - 1) + (threadIdx().x - 1) + 2
    
    # Computation of horizontal net-updates
    if offsetY == 0
        # compute the net-updates
        # reset net updates
        netUpdates1 = 0 # hUpdateLeft
        netUpdates2 = 0 # hUpdateRight
        netUpdates3 = 0 # huUpdateLeft
        netUpdates4 = 0 # huUpdateRight

        # reset the maximum wave speed
        netUpdates5 = 0 # maxWaveSpeed

        hLeft = @inbounds h[cellIndexJ,cellIndexI-1]
        hRight = @inbounds h[cellIndexJ,cellIndexI]
        huLeft = @inbounds hu[cellIndexJ,cellIndexI-1] 
        huRight = @inbounds hu[cellIndexJ,cellIndexI]
        bLeft = @inbounds b[cellIndexJ,cellIndexI-1]
        bRight = @inbounds b[cellIndexJ,cellIndexI]
        
        skip = false

        # determine the wet dry state and corr. values, if necessary.
        if !(hLeft > dryTol && hRight > dryTol)
            if hLeft < dryTol && hRight < dryTol
                skip = true
            elseif hLeft < dryTol
                hLeft = hRight
                huLeft = -huRight
                bLeft = bRight
            else
                hRight = hLeft
                huRight = -huLeft
                bRight = bLeft
            end
        end

        if !skip
            # velocity on the left side of the edge
            uLeft = huLeft / hLeft
            
            # velocity on the right side of the edge
            uRight = huRight / hRight
        
            # wave speeds of the f-waves
            ### compute the wave speeds
            # compute eigenvalues of the jacobian matrices in states Q_{i-1} and Q_{i}
            characteristicSpeed1 = uLeft - sqrt(gravity * hLeft)
            characteristicSpeed2 = uRight + sqrt(gravity * hRight)
            
            # compute "Roe speeds"
            hRoe = 0.5 * (hRight + hLeft)
            uRoe = (uLeft * sqrt(hLeft) + uRight * sqrt(hRight)) / (sqrt(hLeft) + sqrt(hRight))
            
            roeSpeed1 = uRoe - sqrt(gravity * hRoe)
            roeSpeed2 = uRoe + sqrt(gravity * hRoe)
            
            # computer eindfeldt speeds
            einfeldtSpeed1 = min(characteristicSpeed1, roeSpeed1)
            einfeldtSpeed2 = max(characteristicSpeed2, roeSpeed2)
            
            # set wave speeds
            waveSpeed1 = einfeldtSpeed1
            waveSpeed2 = einfeldtSpeed2
        
            ### end  
        
            ### compute the decomposition into f-waves
            lambdaDif = waveSpeed2 - waveSpeed1
        
            # compute the inverse matrix R^{-1}
        
            oneDivLambdaDif = 1 / lambdaDif
            Rinv11 = oneDivLambdaDif * waveSpeed2
            Rinv12 = -oneDivLambdaDif
        
            Rinv21 = oneDivLambdaDif * -waveSpeed1
            Rinv22 = oneDivLambdaDif
        
            # calculate modified (bathymetry!) flux difference
            # f(Q_i) - f(Q_{i-1})
            fDif1 = huRight - huLeft
            fDif2 = huRight * uRight + 0.5 * gravity * hRight * hRight -(huLeft * uLeft + 0.5 * gravity * hLeft * hLeft)
        
            # δx \ Psi[2]
            psi = -gravity * 0.5 * (hRight + hLeft) * (bRight - bLeft)
            fDif2 -= psi
        
            # solve linear equations
            beta1 = Rinv11 * fDif1 + Rinv12 * fDif2
            beta2 = Rinv21 * fDif1 + Rinv22 * fDif2
        
            # return f-waves
            fWave11 = beta1
            fWave12 = beta1 * waveSpeed1
        
            fWave21 = beta2
            fWave22 = beta2 * waveSpeed2
        
            ### end
        
            # compute the net-updates
            # 1st wave family
            if waveSpeed1 < -zeroTol
                netUpdates1 += fWave11
                netUpdates3 += fWave12
            elseif waveSpeed1 > zeroTol
                netUpdates2 +=  fWave11
                netUpdates4 += fWave12
            else
                netUpdates1 += 0.5 * fWave11
                netUpdates3 += 0.5 * fWave12
                netUpdates2 += 0.5 * fWave11
                netUpdates4 += 0.5 * fWave12
            end
        
            # 2nd wave family
            if waveSpeed2 < -zeroTol
                netUpdates1 +=  fWave21
                netUpdates3 += fWave22
            elseif waveSpeed2 > zeroTol
                netUpdates2 += fWave21
                netUpdates4 += fWave22
            else
                netUpdates1 += 0.5 * fWave21 # hUpdateLeft
                netUpdates3 += 0.5 * fWave22 # huUpdateLeft
                netUpdates2 += 0.5 * fWave21 # hUpdateRight
                netUpdates4 += 0.5 * fWave22 # huUpdateRight
            end
        
            # compute maximum wave speed (-> CFL-condition)
            netUpdates5 = max(abs(waveSpeed1), abs(waveSpeed2))

        end

        # store the horizontal net-updates (thread local net-updates -> device net-updates)
        @inbounds hNetUpdatesLeftD[cellIndexJ,cellIndexI-1] = netUpdates1
        @inbounds hNetUpdatesRightD[cellIndexJ,cellIndexI-1] = netUpdates2
        @inbounds huNetUpdatesLeftD[cellIndexJ,cellIndexI-1] = netUpdates3
        @inbounds huNetUpdatesRightD[cellIndexJ,cellIndexI-1] = netUpdates4

        # store the maximum wave speed in the shared array
        @inbounds maxWaveSpeedShared[threadIdx().x, threadIdx().y] = netUpdates5
    end

    # synchronize the threads before the vertical edges (optimization)
    sync_threads()

    # Computation of vertical net-updates
    if offsetX == 0
        # compute the net-updates

        # reset net updates
        netUpdates1 = 0 # hUpdateLeft
        netUpdates2 = 0 # hUpdateRight
        netUpdates3 = 0 # huUpdateLeft
        netUpdates4 = 0 # huUpdateRight

        # reset the maximum wave speed
        netUpdates5 = 0 # maxWaveSpeed

        hLeft = @inbounds h[cellIndexJ-1,cellIndexI]
        hRight = @inbounds h[cellIndexJ,cellIndexI]
        huLeft = @inbounds hv[cellIndexJ-1,cellIndexI]
        huRight = @inbounds hv[cellIndexJ,cellIndexI]
        bLeft = @inbounds b[cellIndexJ-1,cellIndexI]
        bRight = @inbounds b[cellIndexJ,cellIndexI]
        
        skip = false
        
        # determine the wet dry state and corr. values, if necessary.
        if  !(hLeft > dryTol && hRight > dryTol)
            if hLeft < dryTol && hRight < dryTol
                skip = true
            elseif hLeft < dryTol
                hLeft = hRight
                huLeft = -huRight
                bLeft = bRight
            else
                hRight = hLeft
                huRight = -huLeft
                bRight = bLeft
            end
        end

        if !skip
        
            # velocity on the left side of the edge
            uLeft = huLeft / hLeft

            # velocity on the right side of the edge
            uRight = huRight / hRight
        
            # wave speeds of the f-waves
            ### compute the wave speeds
            # compute eigenvalues of the jacobian matrices in states Q_{i-1} and Q_{i}
            characteristicSpeed1 = uLeft - sqrt(gravity * hLeft)
            characteristicSpeed2 = uRight + sqrt(gravity * hRight)
            
            # compute "Roe speeds"
            hRoe = 0.5 * (hRight + hLeft)
            uRoe = (uLeft * sqrt(hLeft) + uRight * sqrt(hRight)) / (sqrt(hLeft) + sqrt(hRight))
            
            roeSpeed1 = uRoe - sqrt(gravity * hRoe)
            roeSpeed2 = uRoe + sqrt(gravity * hRoe)
            
            # computer eindfeldt speeds
            einfeldtSpeed1 = min(characteristicSpeed1, roeSpeed1)
            einfeldtSpeed2 = max(characteristicSpeed2, roeSpeed2)
            
            # set wave speeds
            waveSpeed1 = einfeldtSpeed1
            waveSpeed2 = einfeldtSpeed2
        
            ### end  
        
            ### compute the decomposition into f-waves
            lambdaDif = waveSpeed2 - waveSpeed1
        
            # compute the inverse matrix R^{-1}
        
            oneDivLambdaDif = 1 / lambdaDif
            Rinv11 = oneDivLambdaDif * waveSpeed2
            Rinv12 = -oneDivLambdaDif
        
            Rinv21 = oneDivLambdaDif * -waveSpeed1
            Rinv22 = oneDivLambdaDif
        
            # calculate modified (bathymetry!) flux difference
            # f(Q_i) - f(Q_{i-1})
            fDif1 = huRight - huLeft
            fDif2 = huRight * uRight + 0.5 * gravity * hRight * hRight -(huLeft * uLeft + 0.5 * gravity * hLeft * hLeft)
        
            # δx \ Psi[2]
            psi = -gravity * 0.5 * (hRight + hLeft) * (bRight - bLeft)
            fDif2 -= psi
        
            # solve linear equations
            beta1 = Rinv11 * fDif1 + Rinv12 * fDif2
            beta2 = Rinv21 * fDif1 + Rinv22 * fDif2
        
            # return f-waves
            fWave11 = beta1
            fWave12 = beta1 * waveSpeed1
        
            fWave21 = beta2
            fWave22 = beta2 * waveSpeed2
        
            ### end
        
            # compute the net-updates
            # 1st wave family
            if waveSpeed1 < -zeroTol
                netUpdates1 += fWave11
                netUpdates3 += fWave12
            elseif waveSpeed1 > zeroTol
                netUpdates2 +=  fWave11
                netUpdates4 += fWave12
            else
                netUpdates1 += 0.5 * fWave11
                netUpdates3 += 0.5 * fWave12
                netUpdates2 += 0.5 * fWave11
                netUpdates4 += 0.5 * fWave12
            end
        
            # 2nd wave family
            if waveSpeed2 < -zeroTol
                netUpdates1 +=  fWave21
                netUpdates3 += fWave22
            elseif waveSpeed2 > zeroTol
                netUpdates2 += fWave21
                netUpdates4 += fWave22
            else
                netUpdates1 += 0.5 * fWave21 # hUpdateLeft
                netUpdates3 += 0.5 * fWave22 # huUpdateLeft
                netUpdates2 += 0.5 * fWave21 # hUpdateRight
                netUpdates4 += 0.5 * fWave22 # huUpdateRight
            end
        
            # compute maximum wave speed (-> CFL-condition)
            netUpdates5 = max(abs(waveSpeed1), abs(waveSpeed2))

        end

        # store the vertical net-updates (thread local net-updates -> device net-updates)
        @inbounds hNetUpdatesBelowD[cellIndexJ-1,cellIndexI]  = netUpdates1
        @inbounds hNetUpdatesAboveD[cellIndexJ-1,cellIndexI]  = netUpdates2
        @inbounds hvNetUpdatesBelowD[cellIndexJ-1,cellIndexI] = netUpdates3
        @inbounds hvNetUpdatesAboveD[cellIndexJ-1,cellIndexI] = netUpdates4

        # store the maximum wave speed in the shared array
        @inbounds maxWaveSpeedShared[threadIdx().x, threadIdx().y] = max(maxWaveSpeedShared[threadIdx().x, threadIdx().y], netUpdates5)
    end

    # Compute the maximum observed wave speed
    sync_threads()

    # initialize reduction block size with the original block size
    reductionBlockDimY = blockDim().y
    reductionBlockDimX = blockDim().x

    # do the reduction
    while reductionBlockDimY != 1 || reductionBlockDimX != 1
        # reduction partner for a thread
        reductionPartner = 1

        # split the block in the x-direction (size in x-dir. > 1) or y-direction (size in x-dir. == 1, size in y-dir. > 1)
        if reductionBlockDimX != 1
            reductionBlockDimX >>= 1
            reductionPartner = computeOneDPositionKernel((threadIdx().y-1), (threadIdx().x-1) + reductionBlockDimX, blockDim().x)
        elseif reductionBlockDimY != 1
            reductionBlockDimY >>= 1
            reductionPartner = computeOneDPositionKernel((threadIdx().y-1) + reductionBlockDimY, (threadIdx().x-1), blockDim().x)
        end

    
        if (threadIdx().y-1) < reductionBlockDimY && (threadIdx().x-1) < reductionBlockDimX
            #execute the reduction routine (maximum)
            @inbounds maxWaveSpeedShared[threadIdx().x, threadIdx().y] = max(maxWaveSpeedShared[threadIdx().x, threadIdx().y], maxWaveSpeedShared[reductionPartner])
        end

        sync_threads()
    end

    if (threadIdx().y-1) == 0 && (threadIdx().x-1) == 0
        # Position of the maximum wave speed in the global device array.
        maxWaveSpeedDevicePosition = computeOneDPositionKernel(blockOffSetX + (blockIdx().x-1), blockOffSetY + (blockIdx().y-1), max(blockOffSetY + 1, gridDim().y + 1))

        # write the block local maximum wave speed to the device array
        @inbounds maximumWaveSpeeds[maxWaveSpeedDevicePosition] = maxWaveSpeedShared[1]
    end

    return
end


"""
The "update unknowns"-kernel updates the unknowns in the cells with precomputed net-updates.
"""
function updateUnknownsKernel(hNetUpdatesLeftD, hNetUpdatesRightD, huNetUpdatesLeftD, huNetUpdatesRightD, hNetUpdatesBelowD, hNetUpdatesAboveD,
    hvNetUpdatesBelowD, hvNetUpdatesAboveD, h, hu, hv, updateWidthX, updateWidthY)
    
    # compute the thread local cell indices (start at cell (1,1))
    cellIndexI = blockDim().y * (blockIdx().x-1) + (threadIdx().y-1) + 2
    cellIndexJ = blockDim().x * (blockIdx().y-1) + (threadIdx().x-1) + 2

    # update the cell values
    @inbounds h[cellIndexJ, cellIndexI] -= updateWidthX * (hNetUpdatesRightD[cellIndexJ, cellIndexI-1] + hNetUpdatesLeftD[cellIndexJ, cellIndexI]) + updateWidthY * (hNetUpdatesAboveD[cellIndexJ-1, cellIndexI] + hNetUpdatesBelowD[cellIndexJ, cellIndexI])
    @inbounds hu[cellIndexJ, cellIndexI] -= updateWidthX * (huNetUpdatesRightD[cellIndexJ, cellIndexI-1] + huNetUpdatesLeftD[cellIndexJ, cellIndexI])
    @inbounds hv[cellIndexJ, cellIndexI] -= updateWidthY * (hvNetUpdatesAboveD[cellIndexJ-1, cellIndexI] + hvNetUpdatesBelowD[cellIndexJ, cellIndexI])
    return
end


"""
Compute the position of 2D coordinates in a 1D array. array[i][j] -> i * ny + j
"""
function computeOneDPositionKernel(i, j, ny)
  return 1 + (i * ny + j)
end
