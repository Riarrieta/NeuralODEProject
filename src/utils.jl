
# Cost function to minimize
function node_cost(yhat::Vector{T},y::Vector{T}) where T
    # sum of squares
    cost = sum(sum((ai-bi)^2 for (ai,bi) in zip(a,b))  
                   for (a,b) in zip(yhat,y)) / length(yhat)
    return cost
end
 
function node_cost_withgradient(yhat,y)
    c,cg = Flux.withgradient((x) -> node_cost(x,y), yhat)
    return c,cg[1]  # C, ∂C/∂yhat
end

# ODE solver using Tsit5
solveTsit5(p;kwargs...) = OrdinaryDiffEq.solve(p, Tsit5();kwargs...)

const DEFAULT_ODE_SOLVER = solveTsit5