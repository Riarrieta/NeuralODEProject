using LinearAlgebra
using Flux
using OrdinaryDiffEq
using Plots
using NeuralODEProject
using BSON: @save,@load
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
u0 = Float32[0.05,-0.05,0.0,0.0]
tarray  = 0.0f0:0.1f0:2.5f0
odelist = [NO.ODEStruct(ftrue,u0,tarray) for _ in 1:1]

## Neural network
nn = Chain(Dense(4 => 64,tanh),
           Dense(64 => 4)) |> NO.NeuralODE
## save
@save joinpath(@__DIR__,"data.bson") nn
## load
@load joinpath(@__DIR__,"dp.bson") nn
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

## Plot savefig(joinpath(@__DIR__,"fig.png"))
#index = 5
ode = odelist[1]
ufunc = NO.forward_pass(nn,ode.u0,(first(tarray),last(tarray)))

# extract solution data
t = ode.soltrue.t
θ1 = [u[1] for u in ode.soltrue.u]
θ2 = [u[2] for u in ode.soltrue.u]
p1 = [u[3] for u in ode.soltrue.u]
p2 = [u[4] for u in ode.soltrue.u]
tn = ufunc.t
θ1n = [u[1] for u in ufunc.u]
θ2n = [u[2] for u in ufunc.u]
p1n = [u[3] for u in ufunc.u]
p2n = [u[4] for u in ufunc.u]

plot(t,θ1,label="Ground truth θ1",color=:lightblue,dpi=600,legend=:bottomleft)
plot!(t,θ2,label="Ground truth θ2",color=:darkblue)
plot!(tn,θ1n,label="Prediction θ1",color=:lightgreen,linestyle=:dash)
plot!(tn,θ2n,label="Prediction θ2",color=:darkgreen,linestyle=:dash)
xlabel!("Time")

## Extrapolation
tinf,tsup = 0f0,200f0

tspan2 = (0.0f0, tsup)
ufunc2 = NO.forward_pass(nn,u0,tspan2)
soltrue2 = NO.ode_forward_pass(ftrue,u0,tspan2)

# extract solution data
t_e = soltrue2.t
θ1_e = [u[1] for u in soltrue2.u]
θ2_e = [u[2] for u in soltrue2.u]
p1_e = [u[3] for u in soltrue2.u]
p2_e = [u[4] for u in soltrue2.u]
tn_e = ufunc2.t
θ1n_e = [u[1] for u in ufunc2.u]
θ2n_e = [u[2] for u in ufunc2.u]
p1n_e = [u[3] for u in ufunc2.u]
p2n_e = [u[4] for u in ufunc2.u]

## Plot savefig(joinpath(@__DIR__,"fig.png"))
plot(t_e,θ1_e,label="Ground truth θ1",color=:lightblue,dpi=600,legend=:bottomleft)
plot!(t_e,θ2_e,label="Ground truth θ2",color=:darkblue)
plot!(tn_e,θ1n_e,label="Prediction θ1",color=:lightgreen,linestyle=:dash)
plot!(tn_e,θ2n_e,label="Prediction θ2",color=:darkgreen,linestyle=:dash)
title!("Extrapolation")
xlabel!("Time")
xlims!(190,200)

## Hamiltonian
htrue = dp_hamiltonian.(θ1_e,θ2_e,p1_e,p2_e)
hnn = dp_hamiltonian.(θ1n_e,θ2n_e,p1n_e,p2n_e)

plot(t_e,htrue,label="Ground truth H",color=:blue,dpi=600,legend=:bottomleft)
plot!(tn_e,hnn,label="Prediction H",color=:green,linestyle=:dash)
xlabel!("Time")
ylabel!("H")