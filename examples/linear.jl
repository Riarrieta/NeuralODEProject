using LinearAlgebra
using Flux
using OrdinaryDiffEq
using Plots
using NeuralODEProject
const NO = NeuralODEProject

## Generate data
const A = [-0.1f0 2.0f0; -2.0f0 -0.1f0]
ftrue(du,u) = mul!(du,A,u)  # du = A*u
u0 = [2f0,0f0]
tarray  = 0.0f0:0.1f0:1.0f0 
ode = NO.ODEStruct(ftrue,u0,tarray)

## Neural network
nn = Chain(Dense(2 => 50,tanh),
           Dense(50 => 2)) |> NO.NeuralODE
#optim = Flux.Descent(1e-2)  # Optimiser
optim = Flux.Adam(1e-2)  # Optimiser

## Training
for _ in 1:1000
    global ufunc,cost = NO.node_train!(nn,ode,optim)
    @show cost
end

## Plot
plot(ode.soltrue,label=["Ground truth u[1]" "Ground truth u[2]"])
plot!(ufunc,label=["Prediction u[1]" "Prediction u[2]"],linestyle=:dash)
#scatter!([u[1] for u in soltrue.u], [u[2] for u in soltrue.u], label="Observations")

## Extrapolation
tinf,tsup = -10f0,10f0

tspan2 = (0.0f0, tsup)
ufunc2 = NO.forward_pass(nn,u0,tspan2)
soltrue2 = NO.ode_forward_pass(ftrue,u0,tspan2)

tspan3 = (0.0f0, tinf)
ufunc3 = NO.forward_pass(nn,u0,tspan3)
soltrue3 = NO.ode_forward_pass(ftrue,u0,tspan3)

## Plot
plot(soltrue3,label="Ground truth",color=:green)
plot!(ufunc3,label="Prediction",color=:blue,linestyle=:dash)
plot!(soltrue2,label="Ground truth",color=:green)
plot!(ufunc2,label="Prediction",color=:blue,linestyle=:dash)
xlims!(tinf,tsup)
title!("Extrapolation")

