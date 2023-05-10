using LinearAlgebra
using Flux
using OrdinaryDiffEq
using Plots
using NeuralODEProject
const NO = NeuralODEProject

## Generate data
const A = [-0.1f0 2.0f0; -2.0f0 -0.1f0]
ftrue(du,u) = mul!(du,A,u)  # du = A*u
tarray  = 0.0f0:0.1f0:1.0f0
σ = 5 
odelist = [NO.ODEStruct(ftrue,σ*randn32(2),tarray) for _ in 1:4]

## Neural network
nn = Chain(Dense(2 => 50,tanh),
           Dense(50 => 2)) |> NO.NeuralODE
#optim = Flux.Descent(1e-2)  # Optimiser
optim = Flux.Adam(1e-2)  # Optimiser

## Training v1
costlist = zeros(length(odelist))
ufunclist = Vector(undef,length(odelist))
for _ in 1:300
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

## Plot
#index = 5
u0 = σ*randn32(2)
ode = NO.ODEStruct(ftrue,u0,tarray) #odelist[index]
ufunc = NO.forward_pass(nn,ode.u0,(first(tarray),last(tarray)))# ufunclist[index]

plot(ode.soltrue,label=["Ground truth u[1]" "Ground truth u[2]"])
plot!(ufunc,label=["Prediction u[1]" "Prediction u[2]"],linestyle=:dash)
#scatter!([u[1] for u in soltrue.u], [u[2] for u in soltrue.u], label="Observations")

## Extrapolation
tinf,tsup = -5f0,50f0

tspan2 = (0.0f0, tsup)
ufunc2 = NO.forward_pass(nn,u0,tspan2)
soltrue2 = NO.ode_forward_pass(ftrue,u0,tspan2)

tspan3 = (0.0f0, tinf)
ufunc3 = NO.forward_pass(nn,u0,tspan3)
soltrue3 = NO.ode_forward_pass(ftrue,u0,tspan3)

## Plot
plot(soltrue3,label="Ground truth",color=:green,legend=false)
plot!(ufunc3,label="Prediction",color=:blue,linestyle=:dash)
plot!(soltrue2,label="Ground truth",color=:green)
plot!(ufunc2,label="Prediction",color=:blue,linestyle=:dash)
xlims!(tinf,tsup)
title!("Extrapolation")

## Compare nn forward map angle
t = range(0,1,100)
u = [Float32[cos(2π*t),sin(2π*t)] for t in t]
dutrue = [A*u for u in u]
dunn = [nn(u) for u in u]

plot(t,[du[1] for du in dutrue],label="Ground truth u[1]",color=:green,legend=false)
plot!(t,[du[2] for du in dutrue],label="Ground truth u[2]",color=:green)
plot!(t,[du[1] for du in dunn],label="Prediction u[1]",color=:blue,linestyle=:dash)
plot!(t,[du[2] for du in dunn],label="Prediction u[2]",color=:blue,linestyle=:dash)

## Compare nn forward map radius
t = range(0,1,100)
r = 25
ui = randn32(2)
u = [r*ui*t for t in t]
dutrue = [A*u for u in u]
dunn = [nn(u) for u in u]

plot(t,[du[1] for du in dutrue],label="Ground truth u[1]",color=:green,legend=false)
plot!(t,[du[2] for du in dutrue],label="Ground truth u[2]",color=:green)
plot!(t,[du[1] for du in dunn],label="Prediction u[1]",color=:blue,linestyle=:dash)
plot!(t,[du[2] for du in dunn],label="Prediction u[2]",color=:blue,linestyle=:dash)