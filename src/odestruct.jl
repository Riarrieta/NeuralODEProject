struct ODEStruct
    f        # derivative function
    u0       # initial condition
    tarray   # initial condition
    soltrue  # solution
    utrue    # solution samples
end
Base.show(io::IO,ode::ODEStruct) = print(io,"ODEStruct()")

function ODEStruct(f,u0,tarray;odesolver=DEFAULT_ODE_SOLVER)
    ftrue(du,u,p,t) = f(du,u)
    tspan = (first(tarray), last(tarray))
    probtrue = ODEProblem(ftrue, u0, tspan)
    soltrue = odesolver(probtrue; saveat=tarray)
    @assert tarray == soltrue.t
    utrue = soltrue.u 
    return ODEStruct(f,u0,tarray,soltrue,utrue)
end

function ode_forward_pass(f,u0,tspan;odesolver=DEFAULT_ODE_SOLVER)
    ftrue(du,u,p,t) = f(du,u)
    probtrue = ODEProblem(ftrue,u0,tspan)
    soltrue = odesolver(probtrue)
    return soltrue
end