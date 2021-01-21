cd(@__DIR__)
using Pkg;
Pkg.activate("..");
Pkg.instantiate();

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

############
## Create a simple spring case, use Neural network to learn its force/position
############

k = 1.0
m=1.0

force(dx,x,k,t) = -k*x + 0.1sin(x)

acceleration(dx,x,k,t) = force(dx,x,k,t)/m
prob = SecondOrderODEProblem(acceleration,1.0,0.0,(0.0,10.0),k)
sol = solve(prob)
plot(sol,label=["Velocity" "Position"])


plot_t = 0:0.01:10
data_plot = sol(plot_t)
positions_plot = [state[2] for state in data_plot]
force_plot = [force(state[1],state[2],k,t) for state in data_plot]

# Generate the dataset
t = 0:3.3:10
dataset = sol(t)
position_data = [state[2] for state in sol(t)]
force_data = [force(state[1],state[2],k,t) for state in sol(t)]

plot(plot_t,force_plot,xlabel="t",label="True Force")
scatter!(t,force_data,label="Force Measurements")

# Train a network to estimate the force
NNForce = Chain(x -> [x],
           Dense(1,32,tanh),
           Dense(32,1),
           first)

loss() = sum(abs2,NNForce(position_data[i]) - force_data[i] for i in 1:length(position_data))
loss()

opt = Flux.Descent(0.01)
data = Iterators.repeated((), 5000)
iter = 0
cb = function () #callback function to observe training
  global iter += 1
  if iter % 500 == 0
    display(loss())
  end
end
display(loss())
Flux.train!(loss, Flux.params(NNForce), data, opt; cb=cb)

learned_force_plot = NNForce.(positions_plot)

plot(plot_t,force_plot,xlabel="t",label="True Force")
plot!(plot_t,learned_force_plot,label="Predicted Force")
scatter!(t,force_data,label="Force Measurements")


force2(dx,x,k,t) = -k*x
acceleration2(dx,x,k,t) = force2(dx,x,k,t)/m
prob_simplified = SecondOrderODEProblem(acceleration2,1.0,0.0,(0.0,10.0),k)
sol_simplified = solve(prob_simplified)
plot(sol,label=["Velocity" "Position"])
plot!(sol_simplified,label=["Velocity Simplified" "Position Simplified"])

random_positions = [2rand()-1 for i in 1:100] # random values in [-1,1]
loss_ode() = sum(abs2,NNForce(x) - (-k*x) for x in random_positions)
loss_ode()

λ = 0.1
composed_loss() = loss() + λ*loss_ode()

Flux.train!(composed_loss, Flux.params(NNForce), data, opt; cb=cb)

learned_force_plot = NNForce.(positions_plot)

plot(plot_t,force_plot,xlabel="t",label="True Force")
plot!(plot_t,learned_force_plot,label="Predicted Force")
scatter!(t,force_data,label="Force Measurements")


#####################
#  ODE for 1 spring
#####################


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

gif(anim, output_figures*"anim_torque_10.gif", fps = 10)

#-----------
# 4 springs one mass
#         ~
#  |~~~~~||||~~~~~~|
#         ~
#-----------

x0=1.0
y0=1.0
k1=1.0
k2=1.0
k3=1.0
k4=1.0
d=0.1
m=1.0

function springs_eq(u, p, t)
    positionX = u[1]
    positionY = u[2]
    velocityX = u[3]
    velocityY = u[4]
    (k1,k2,k3,k4,m,damping,x0,y0) = p
    springForceX1 = -k1 * (positionX - x0)
    springForceY1 = -k1 * (positionY - y0)
    springForceX2 = k2 * -(positionX - x0)
    springForceY2 = k2 * -(positionY - y0)
    springForceX3 = -k3 * (positionX - x0)
    springForceY3 = -k3 * (positionY - y0)
    springForceX4 = k4 * -(positionX - x0)
    springForceY4 = k4 * -(positionY - y0)
    dampingForceX = damping * velocityX
    dampingForceY = damping * velocityY
    forceX = (springForceX1 + springForceX2 + springForceX3 + springForceX4) - dampingForceX
    forceY = (springForceY1 + springForceY2 + springForceY3 + springForceY4) - dampingForceY
    accelerationX = forceX/m
    accelerationY = forceY/m
    [velocityX, velocityY, accelerationX, accelerationY]
end


U0 = [1.5,1.5,1.0,2.0]
tspan = (0.0,30.0)
Δt = 0.1
p = [k1,k2,k3,k4,m,d,x0,y0]
prob = ODEProblem(springs_eq, U0, tspan, p)
sol = solve(prob, Tsit5(), saveat=Δt)
data = Array(sol)
time =tspan[1]:Δt:tspan[2]

plot3d(time, data[1,:],data[2,:])

plt = plot3d(
    1,
    zlim = (0, 2),
    ylim = (0, 2),
    xlim = (0, 30),
    title = "springs",
    marker = 2,
)

@gif for i=1:301
    push!(plt, time[i], data[1,i], data[2,i])
end every 10

plot(data[1,:],data[2,:])

plt = plot(
    1,
    xlim = (0, 2),
    ylim = (0, 2),
    title = "springs",
    marker = 2,
)

anim = @animate for i=1:301
    #push!(plt, plot((0,0)(data[1,i], data[2,i])))
    plot([(0,1),(data[1,i],data[2,i])], marker = :hex, xlims=(0,2), ylims=(0,2),legend=false)
    plot!([(data[1,i],data[2,i]),(2,1)], marker = :hex, xlims=(0,2), ylims=(0,2),legend=false)
    plot!([(data[1,i],data[2,i]),(1,2)], marker = :hex, xlims=(0,2), ylims=(0,2),legend=false)
    plot!([(data[1,i],data[2,i]),(1,0)], marker = :hex, xlims=(0,2), ylims=(0,2),legend=false)
end

gif(anim, output_figures*"anim_4springs_10.gif", fps = 10)


##  Hooke's law
#  F = k(||xa - xb||2 - l0) (xa-xb)/||xa-xb||2
#  where k is the spring stiffness, F is spring force,
#  xa and xb are the positions of two mass, and l0 is the rest length
##

 
