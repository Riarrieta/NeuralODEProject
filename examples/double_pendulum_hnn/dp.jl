using LinearAlgebra
using Flux
using OrdinaryDiffEq
using Plots
using NeuralODEProject
using DiffEqFlux
using ReverseDiff
const NO = NeuralODEProject

## Generate data
const m1,m2,l1,l2 = (1f0, 1f0, 1f0, 1f0)
const g = 9.8
function dp_hamiltonian(θ1,θ2,p1,p2)
    H = m2*l2^2*p1^2 + (m1+m2)*l1^2*p2^2-2*m2*l1*l2*p1*p2*cos(θ1-θ2)
    H /= 2*m2*l1^2*l2^2*(m1+m2*sin(θ1-θ2)^2)
    H += -(m1+m2)*g*l1*cos(θ1)-m2*g*l2*cos(θ2)
    return H
end 
function ftrue(du,u)
    # double pendulum
    θ1,θ2,p1,p2 = u
    dhθ1,dhθ2,dhp1,dhp2 = Flux.gradient(dp_hamiltonian,θ1,θ2,p1,p2)
    du[1] = dhp1  # θ1  
    du[2] = dhp2  # θ2
    du[3] = -dhθ1 # p1
    du[4] = -dhθ2 # p2
end
u0 = Float32[1.0,1.0,0.0,0.0]
tarray  = 0.0f0:0.25f0:10f0
odelist = [NO.ODEStruct(ftrue,u0,tarray) for _ in 1:1]

## Neural network
nn = Chain(Dense(2 => 64,tanh),
           Dense(64 => 2)) |> NO.NeuralODE
## save
@save joinpath(@__DIR__,"data.bson") nn
## load
@load joinpath(@__DIR__,"lotka_volterra.bson") nn
## Optimiser
optim = Flux.Adam(1e-2)  # Optimiser
## Training v1
costlist = zeros(length(odelist))
ufunclist = Vector(undef,length(odelist))
for _ in 1:1000
    for (i,ode_) in enumerate(odelist)
        ufunc,cost = NO.node_train!(nn,ode_,optim)
        ufunclist[i] = ufunc
        costlist[i] = cost
    end
    @show maximum(costlist)
end

## Training v2
costlist = zeros(length(odelist))
ufunclist = Vector(undef,length(odelist))
for (i,ode_) in enumerate(odelist)
    for _ in 1:30
        ufunc,cost = NO.node_train!(nn,ode_,optim)
        ufunclist[i] = ufunc
        costlist[i] = cost
    end
    @show maximum(costlist)
end

## Plot savefig(joinpath(@__DIR__,"fig.png"))
#index = 5
ode = odelist[1]
ufunc = NO.forward_pass(nn,ode.u0,(first(tarray),last(tarray)))
obs1,obs2 = zip(ode.utrue...)
obs1,obs2 = collect(obs1),collect(obs2)

plot(ode.soltrue,label=["Ground truth prey" "Ground truth predator"],color=:green,dpi=600,legend=:topleft)
plot!(ufunc,label=["Prediction prey" "Prediction predator"],linestyle=:dash,color=:blue)
scatter!(tarray,obs1,label="Observation prey",color=:green)
scatter!(tarray,obs2,label="Observation predator",color=:green)
xlabel!("Time")

## Extrapolation
tinf,tsup = 0f0,50f0

tspan2 = (0.0f0, tsup)
ufunc2 = NO.forward_pass(nn,u0,tspan2)
soltrue2 = NO.ode_forward_pass(ftrue,u0,tspan2)

tspan3 = (0.0f0, tinf)
ufunc3 = NO.forward_pass(nn,u0,tspan3)
soltrue3 = NO.ode_forward_pass(ftrue,u0,tspan3)

## Plot savefig(joinpath(@__DIR__,"fig.png"))
plot(soltrue3,label=["Ground truth prey" "Ground truth predator"],color=:green,legend=:bottomright,dpi=600)
plot!(ufunc3,label=["Prediction prey" "Prediction predator"],color=:blue,linestyle=:dash)
plot!(soltrue2,label="",color=:green)
plot!(ufunc2,label="",color=:blue,linestyle=:dash)
xlims!(tinf,tsup)
title!("Extrapolation")
xlabel!("Time")
