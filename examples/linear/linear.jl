using LinearAlgebra
using Flux
using OrdinaryDiffEq
using Plots
using NeuralODEProject
using BSON: @save,@load
const NO = NeuralODEProject

## Generate data
A = [-0.1f0 2.0f0; -2.0f0 -0.1f0]
ftrue(du,u) = mul!(du,A,u)  # du = A*u
tarray  = 0.0f0:0.1f0:1f0
ntraining = 4
σ = 4
odelist = [NO.ODEStruct(ftrue,σ*randn32(2),tarray) for _ in 1:ntraining]
append!(odelist,[NO.ODEStruct(ftrue,σ*randn32(2),tarray) for _ in 1:ntraining])

## Neural network
nn = Chain(Dense(2 => 64,tanh),
           Dense(64 => 2)) |> NO.NeuralODE
## save
@save joinpath(@__DIR__,"data.bson") nn
## load
@load joinpath(@__DIR__,"linear.bson") nn
## Optimiser
optim = Flux.Adam(1e-2)  # Optimiser
## Training v1
costlist = zeros(length(odelist))
ufunclist = Vector(undef,length(odelist))
for _ in 1:50
    for (i,ode_) in enumerate(odelist)
        ufunc,cost = NO.node_train!(nn,ode_,optim)
        ufunclist[i] = ufunc
        costlist[i] = cost
    end
    @show maximum(costlist)
end

## Training v2
for _ in 1:300
    _,cost = NO.node_train!(nn,odelist,optim)
    @show cost
end

## Training v3
for _ in 1:20
    ode = NO.ODEStruct(ftrue,σ*randn32(2),tarray)
    optim = Flux.Adam(1e-2)
    cost = 0
    for _ in 1:50
        ufunc,cost = NO.node_train!(nn,ode,optim)
    end
    @show cost
end

## Plot savefig(joinpath(@__DIR__,"fig.png"))
#index = 5
u0 = σ*randn32(2)
ode = NO.ODEStruct(ftrue,u0,tarray) 
ufunc = NO.forward_pass(nn,ode.u0,(first(tarray),last(tarray)))
soltrue1 = NO.ode_forward_pass(ftrue,u0,(first(tarray),last(tarray)))

plot(soltrue1,label=["Ground truth u[1]" "Ground truth u[2]"],color=:green,dpi=600)
plot!(ufunc,label=["Prediction u[1]" "Prediction u[2]"],linestyle=:dash,color=:blue)
xlabel!("Time")

## Extrapolation
tinf,tsup = -30f0,10f0

tspan2 = (0.0f0, tsup)
ufunc2 = NO.forward_pass(nn,u0,tspan2)
soltrue2 = NO.ode_forward_pass(ftrue,u0,tspan2)

tspan3 = (0.0f0, tinf)
ufunc3 = NO.forward_pass(nn,u0,tspan3)
soltrue3 = NO.ode_forward_pass(ftrue,u0,tspan3)

## Plot savefig(joinpath(@__DIR__,"fig.png"))
plot(soltrue3,label=["Ground truth u[1]" "Ground truth u[2]"],color=:green,legend=true,dpi=600)
plot!(ufunc3,label=["Prediction u[1]" "Prediction u[2]"],color=:blue,linestyle=:dash)
plot!(soltrue2,label="",color=:green)
plot!(ufunc2,label="",color=:blue,linestyle=:dash)
xlims!(tinf,tsup)
title!("Extrapolation")
xlabel!("Time")

## Compare nn forward map angle
t = range(0,1,100)
u = [Float32[cos(2π*t),sin(2π*t)] for t in t]
dutrue = [A*u for u in u]
dunn = [nn(u) for u in u]

plot(t,[du[1] for du in dutrue],label="Ground truth u[1]",color=:green,legend=true,dpi=600)
plot!(t,[du[2] for du in dutrue],label="Ground truth u[2]",color=:green)
plot!(t,[du[1] for du in dunn],label="Prediction u[1]",color=:blue,linestyle=:dash)
plot!(t,[du[2] for du in dunn],label="Prediction u[2]",color=:blue,linestyle=:dash)
xlabel!("θ")

## Compare nn forward map radius
t = range(0,1,100)
r = 25
rt = r*t
ui = randn32(2)
u = [r*ui*t for t in t]
dutrue = [A*u for u in u]
dunn = [nn(u) for u in u]

plot(rt,[du[1] for du in dutrue],label="Ground truth u[1]",color=:green,legend=true,dpi=600)
plot!(rt,[du[2] for du in dutrue],label="Ground truth u[2]",color=:green)
plot!(rt,[du[1] for du in dunn],label="Prediction u[1]",color=:blue,linestyle=:dash)
plot!(rt,[du[2] for du in dunn],label="Prediction u[2]",color=:blue,linestyle=:dash)
xlabel!("r")