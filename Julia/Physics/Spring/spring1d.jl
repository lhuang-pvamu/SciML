cd(@__DIR__)
#using Pkg;
#Pkg.activate(".");
#Pkg.activate("~/.julia/environments/v1.5/Project.toml");
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
using BSON: @save,@load
using Surrogates
gr()

output_figures="Figures/"
output_models="Models/"

g = 9.81
#k = 9.0
#m = 30.0
#damping = 10.0
#anchorY = 200.0

function spring_ode_1d(u, p, t)
    position = u[1]
    velocity = u[2]
    (k,m,damping,y0) = p
    springForceY = -k * (position - y0)
    dampingForceY = damping * velocity
    forceY = springForceY + m*g - dampingForceY
    accelerationY = forceY/m
    [velocity, accelerationY]
end


U0 = [250,0]
tspan = (0.0,10.0)
Δt = 0.1
p = Float32[100.0,5.0,1.0,200.0]
prob = ODEProblem(spring_ode_1d, U0, tspan, p)
sol = solve(prob, Tsit5(), saveat=Δt)
data = Array(sol)

plot(data[1,:])
plot(data[1,:], data[2,:])

y = data[1,:]

anim = @animate for i ∈ 1:2:length(y)
    plot([(100,200),(100,y[i])], marker = :hex, xlims=(0,200), ylims=(100,300), yflip = true,
    title="$(round((i-1)*0.1,digits=1)) second", legend = false)
end

gif(anim, output_figures*"anim_torque_10.gif", fps = 5)

#----------
# Spring Mass 2D Problem
#---------

function spring_ode_2d(u, p, t)
    positionX = u[1]
    positionY = u[2]
    velocityX = u[3]
    velocityY = u[4]
    (k,m,damping,position0) = p
    springForceX = -k * (positionX - position0[1])
    springForceY = -k * (positionY - position0[2])
    dampingForceX = damping * velocityX
    dampingForceY = damping * velocityY
    forceX = springForceX - dampingForceX
    forceY = springForceY + m*g - dampingForceY
    accelerationX = forceX / m
    accelerationY = forceY / m
    [velocityX, velocityY, accelerationX, accelerationY]
end

U0 = [150,220,0,0]
tspan = (0.0,30.0)
Δt = 0.1
p = [7.0,30.0,1.0,[100.0,200.0]]
prob = ODEProblem(spring_ode_2d, U0, tspan, p)
sol = solve(prob, Tsit5(), saveat=Δt)
data = Array(sol)

scatter(data[1,:],data[2,:])

x = data[1,:]
y = data[2,:]

anim = @animate for i ∈ 1:2:length(y)
    plot([(100,200),(x[i],y[i])], marker = :hex, xlims=(0,200), ylims=(200,350), yflip = true,
    title="$(round((i-1)*0.1,digits=1)) second", legend = false)
end

gif(anim, output_figures*"anim_torque_10.gif", fps = 5)

#-----------
# 2 springs
#-----------
