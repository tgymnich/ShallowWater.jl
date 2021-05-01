"""
Compute net updates for the cell on the left/right side of the edge

The order of o_netUpdates is given by:
    0: hUpdateLeft   - Net-update for the height of the cell on the left side of the edge.
    1: hUpdateRight  - Net-update for the height of the cell on the right side of the edge.
    2: huUpdateLeft  - Net-update for the momentum of the cell on the left side of the edge
    3: huUpdateRight - Net-update for the momentum of the cell on the right side of the edge.
    4: maxWaveSpeed  - Maximum (linearized) wave speed -> Should be used in the CFL-condition.
"""
function fWaveComputeNetUpdates(gravity, hLeft, hRight, huLeft, huRight, bLeft,  bRight, netUpdates)
    # reset net updates
    netUpdates[1] = 0 # hUpdateLeft
    netUpdates[2] = 0 # hUpdateRight
    netUpdates[3] = 0 # huUpdateLeft
    netUpdates[4] = 0 # huUpdateRight

    # reset the maximum wave speed
    netUpdates[5] = 0 # maxWaveSpeed

    # determine the wet dry state and corr. values, if necessary.
    if  !(hLeft > dryTol && hRight > dryTol)
        if hLeft < dryTol && hRight < dryTol
            return
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

    # velocity on the left side of the edge
    uLeft = huLeft / hLeft
    
    # velocity on the right side of the edge
    uRight = huRight / hRight

    # wave speeds of the f-waves
    # waveSpeeds = MArray{Tuple{2},Float32}(undef)
    waveSpeeds = @cuStaticSharedMem(Float32, (2))    


    # compute the wave speeds
    fWaveComputeWaveSpeeds(gravity, hLeft, hRight, huLeft, huRight, uLeft, uRight, bLeft, bRight, waveSpeeds)

    #fWaves = MArray{Tuple{2,2},Float32}(undef)
    fWaves = @cuStaticSharedMem(Float32, (2,2))    

    # compute the decomposition into f-waves
    fWaveComputeWaveDecomposition(gravity, hLeft, hRight, huLeft, huRight, uLeft, uRight, bLeft, bRight, waveSpeeds, fWaves)

    # compute the net-updates
    # 1st wave family
    if waveSpeeds[1] < -zeroTol
        netUpdates[1] += fWaves[1,1]
        netUpdates[3] += fWaves[1,2]
    elseif waveSpeeds[1] > zeroTol
        netUpdates[2] +=  fWaves[1,1]
        netUpdates[4] += fWaves[1,2]
    else
        netUpdates[1] += 0.5 * fWaves[1,1]
        netUpdates[3] += 0.5 * fWaves[1,2]
        netUpdates[2] += 0.5 * fWaves[1,1]
        netUpdates[4] += 0.5 * fWaves[1,2]
    end

    # 2nd wave family
    if waveSpeeds[2] < -zeroTol
        netUpdates[1] +=  fWaves[2,1]
        netUpdates[3] += fWaves[2,2]
    elseif waveSpeeds[2] > zeroTol
        netUpdates[2] += fWaves[2,1]
        netUpdates[4] += fWaves[2,2]
    else
        netUpdates[1] += 0.5 * fWaves[2,1] # hUpdateLeft
        netUpdates[3] += 0.5 * fWaves[2,2] # huUpdateLeft
        netUpdates[2] += 0.5 * fWaves[2,1] # hUpdateRight
        netUpdates[4] += 0.5 * fWaves[2,2] # huUpdateRight
    end

    # compute maximum wave speed (-> CFL-condition)
    netUpdates[5] = max(abs(waveSpeeds[1]), abs(waveSpeeds[2]))
    return
end

"""
Compute the edge local eigenvalues.
"""
function fWaveComputeWaveSpeeds(gravity, hLeft, hRight, huLeft, huRight, uLeft, uRight, bLeft, bRight, waveSpeeds)
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
    waveSpeeds[1] = einfeldtSpeed1
    waveSpeeds[2] = einfeldtSpeed2
    return
end

"""
Compute the decomposition into f-Waves.
"""
function fWaveComputeWaveDecomposition(gravity, hLeft, hRight, huLeft, huRight, uLeft, uRight, bLeft, bRight, waveSpeeds, fWaves)

    lambdaDif = waveSpeeds[2] - waveSpeeds[1]

    # compute the inverse matrix R^{-1}

    oneDivLambdaDif = 1 / lambdaDif
    Rinv11 = oneDivLambdaDif * waveSpeeds[2]
    Rinv12 = -oneDivLambdaDif

    Rinv21 = oneDivLambdaDif * -waveSpeeds[1]
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
    fWaves[1,1] = beta1
    fWaves[1,2] = beta1 * waveSpeeds[1]

    fWaves[2,1] = beta2
    fWaves[2,2] = beta2 * waveSpeeds[2]
    return
end