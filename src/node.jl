
struct NeuralODE{A<:Chain,B,T}
    nn::A
    params::B
    nparams::Int64
    sizeparams::Vector{Tuple{Int64,Vararg{Int64}}}
    lengthparams::Vector{Int64}
    ndim::Int64
    r0::Vector{T}   # backward ODE initial condition r0 = [μ₁,μ₂,...,λ]
    bstate_indices::Vector{UnitRange{Int64}}  # backward state indices
end
Base.show(io::IO,nn::NeuralODE) = print(io,"NeuralODE()")
(nn::NeuralODE)(u) = nn.nn(u)

# adjoint variables
μstate(nn::NeuralODE,rstate) = @view rstate[1:end-nn.ndim]
λstate(nn::NeuralODE,rstate) = @view rstate[end-nn.ndim+1:end]

function obtain_backward_state_indices(nparams,lengthparams,ndim)
    index1 = 1
    index2 = lengthparams[1]
    # μ states
    bstate_indices = [index1:index2]
    for i in 2:nparams
        index1 = index2+1
        index2 += lengthparams[i]
        push!(bstate_indices,index1:index2)
    end
    # λ state
    index1 = index2+1
    index2 += ndim
    push!(bstate_indices,index1:index2)
    return bstate_indices
end

function NeuralODE(nn)
    params = Flux.params(nn)
    nparams = length(params)
    sizeparams = [size(w) for w in params]
    lengthparams = [length(w) for w in params]
    ndim = size(layers(nn)[1].weight, 2)
    T = layers(nn)[1].weight |> eltype
    r0 = zeros(T,ndim+sum(lengthparams))
    bstate_indices = obtain_backward_state_indices(nparams,lengthparams,ndim)
    return NeuralODE(nn,params,nparams,sizeparams,lengthparams,
                     ndim,r0,bstate_indices)
end

function forward_pass(nn::NeuralODE,u0,tspan;odesolver=DEFAULT_ODE_SOLVER)
    probNN = ODEProblem((u,p,t) -> p(u), u0, tspan, nn)
    ufunc = odesolver(probNN)
    return ufunc
end

# to be used with OrdinaryDiffEq.solve
function backward_ode(dr,r,p,t)
    nn,ufunc = p  # unpack parameters (neural network and forward pass)
    uvalue = ufunc(t)
    # add uvalue to nn.params
    push!(nn.params,uvalue)
    # pullback
    _,pullback_func = Flux.pullback(() -> nn(uvalue),nn.params)  
    λ = λstate(nn,r)  # λ state
    pullback = pullback_func(λ) 
    # flattened version of the pullback, for weight #i
    flattened_pullback(i::Int64) = pullback[nn.params[i]] |> vec
    for (i,state_index) in enumerate(nn.bstate_indices)
        dr[state_index] = -flattened_pullback(i)
    end
    # remove uvalue from nn.params
    delete!(nn.params,uvalue)
    return nothing
end

function backward_ode_callbacks(prob_backward,tarray,
                                cost_ugrad;odesolver=DEFAULT_ODE_SOLVER)
    tstops = collect(tarray)
    tlength = tstops[end]-tstops[1]
    # modify endpoints of tstops, assumes tarray is sorted
    if tstops[1] == prob_backward.tspan[2]
        tstops[1] += tlength*1e-6
    end
    if tstops[end] == prob_backward.tspan[1]
        tstops[end] -= tlength*1e-6
    end
    # generate callbacks to add jumps to λ state
    index = nothing
    function condition(u,t,integrator)
        index = findfirst(x -> x==t, tstops)
        return !isnothing(index)
    end
    function affect!(integrator)
        nn,_ = integrator.p  # unpack parameters (neural network and forward pass)
        λ = λstate(nn,integrator.u)
        λ .+= cost_ugrad[index] 
    end
    callback = DiscreteCallback(condition,affect!)
    return callback,tstops
end

function backward_pass(nn::NeuralODE,ufunc,tspan,tarray,
                       cost_ugrad;odesolver=DEFAULT_ODE_SOLVER)
    # setup ODEProblem
    tspan_inv = (tspan[2],tspan[1])  # reverse time
    r0 = nn.r0
    prob_backward = ODEProblem(backward_ode,r0,tspan_inv,(nn,ufunc))
    # solve backward ode
    callback,tstops = backward_ode_callbacks(prob_backward,tarray,cost_ugrad;odesolver)
    rfunc = odesolver(prob_backward;callback,tstops)
    return rfunc
end

function update_weights!(nn::NeuralODE,cost_pgrad,optim)
    # assemble gradient with correct shapes
    μ = μstate(nn,cost_pgrad)
    grads = IdDict(w => reshape(μ[index],s) 
                   for (w,s,index) in zip(nn.params,nn.sizeparams,nn.bstate_indices))
    Flux.update!(optim,nn.params,grads)
end

function node_weights_gradient(nn::NeuralODE,u0,tspan,tarray,
                               utrue;odesolver=DEFAULT_ODE_SOLVER)
    # forward pass
    ufunc = forward_pass(nn,u0,tspan;odesolver)
    # cost
    umodel = ufunc.(tarray)
    cost, cost_ugrad = node_cost_withgradient(umodel,utrue)
    # backward pass
    rfunc = backward_pass(nn,ufunc,tspan,tarray,cost_ugrad;odesolver)
    cost_pgrad = rfunc(tspan|>first)  # gradient of cost w/r to parameters
    return cost_pgrad,ufunc,cost
end
function node_weights_gradient(nn::NeuralODE,ode::ODEStruct;
                               odesolver=DEFAULT_ODE_SOLVER)
    tspan = (first(ode.tarray),last(ode.tarray))
    return node_weights_gradient(nn,ode.u0,tspan,ode.tarray,ode.utrue;odesolver)                      
end

function node_train!(nn::NeuralODE,u0,tspan,tarray,utrue,
                     optim;odesolver=DEFAULT_ODE_SOLVER)
    cost_pgrad,ufunc,cost = node_weights_gradient(nn,u0,tspan,tarray,utrue;odesolver)
    # update weights
    update_weights!(nn,cost_pgrad,optim)
    return ufunc,cost
end
function node_train!(nn::NeuralODE,ode::ODEStruct,optim;odesolver=DEFAULT_ODE_SOLVER)
    tspan = (first(ode.tarray),last(ode.tarray))
    return node_train!(nn::NeuralODE,ode.u0,tspan,ode.tarray,ode.utrue,optim;odesolver)
end
function node_train!(nn::NeuralODE,odelist::Vector{<:ODEStruct},
                     optim;odesolver=DEFAULT_ODE_SOLVER)
    cost_pgrad = zero(nn.r0)     # average of gradients
    cost = zero(nn.r0 |> eltype) # average cost
    for ode in odelist
        ode_pgrad,_,ode_cost = node_weights_gradient(nn,ode;odesolver)
        cost_pgrad .+= ode_pgrad/length(odelist)
        cost += ode_cost/length(odelist)
    end
    # update weights
    update_weights!(nn,cost_pgrad,optim)
    return nothing,cost
end
