struct ODEStruct
    f        # derivative function
    u0       # initial condition
    tarray   # initial condition
    soltrue  # solution
    utrue    # solution samples
end
Base.show(io::IO,ode::ODEStruct) = print(io,"ODEStruct()")

function ODEStruct(f,u0,tarray,noise=0f0;odesolver=DEFAULT_ODE_SOLVER)
    ftrue(du,u,p,t) = f(du,u)
    tspan = (first(tarray), last(tarray))
    probtrue = ODEProblem(ftrue, u0, tspan)
    soltrue = odesolver(probtrue; saveat=tarray)
    @assert tarray == soltrue.t
    utrue = soltrue.u 
    # add noise
    if !iszero(noise)
        T = eltype(eltype(utrue))
        s = size(first(utrue))
        utrue = copy(utrue) + [noise*randn(T,s) for _ in utrue]
    end
    return ODEStruct(f,u0,tarray,soltrue,utrue)
end

function ode_forward_pass(f,u0,tspan;odesolver=DEFAULT_ODE_SOLVER)
    ftrue(du,u,p,t) = f(du,u)
    probtrue = ODEProblem(ftrue,u0,tspan)
    soltrue = odesolver(probtrue)
    return soltrue
end