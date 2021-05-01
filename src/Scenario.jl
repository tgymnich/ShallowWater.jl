struct DamBreakScenario end

function getBathymetry(scenario::DamBreakScenario, x, y)
    return 0
end

function getVeloc_u(scenario::DamBreakScenario, x, y)
    return 0
end

function getVeloc_v(scneario::DamBreakScenario, x, y)
    return 0
end

function getWaterHeight(scneario::DamBreakScenario, x, y)
    return sqrt((x-500)*(x-500) + (y-500)*(y-500)) < 100 ? 15 : 10
end

function endSimulationTime(scneario::DamBreakScenario)
    return 30
end

function waterHeightAtRest(scneario::DamBreakScenario)
    return 10
end

function getBoundaryType(scneario::DamBreakScenario, edge)
    return OUTFLOW
end

function getBoundaryPos(scneario::DamBreakScenario, edge)
    if edge == LEFT
        return 0
    elseif edge == RIGHT
        return 1000
    elseif edge == BOTTOM
        return 0
    elseif edge == TOP
        return 1000
    else
        error("invalid edge")
    end
end
