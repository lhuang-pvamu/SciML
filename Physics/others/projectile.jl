cd(@__DIR__)
#using Pkg;
#Pkg.activate(".");
#Pkg.instantiate();

using DifferentialEquations
using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra, DiffEqSensitivity, Optim
using DiffEqFlux
using Flux
using Plots
using Zygote
using HDF5
gr()

output_figures="Figures/"
output_models="Models/"

g=9.8
#v0=10
#θ=45

function projectile_ode(u, p, t)
    v0 = p[1]
    θ = p[2]
    [v0*cos(θ), v0*sin(θ)-g*t]
end

U0 = Float32[0.0, 0.0]
tspan = (0.0,5.0)
Δt = 0.01
p = Float32[20.0,pi/3]

prob = ODEProblem(projectile_ode, U0, tspan, p)
sol = solve(prob, Tsit5(), saveat=Δt)
data = Array(sol)

plot(sol)

scatter(data[1,:],data[2,:]) #, xlims=(0,30))

plt = scatter((data[1,1],data[2,1]), xlims=(0,60), ylims=(-15,15))

anim = @animate for i ∈ 1:5:length(data[1,:])
     push!(plt, (data[1,i],data[2,i]))
end
gif(anim, "anim_fps30.gif", fps = 10)
