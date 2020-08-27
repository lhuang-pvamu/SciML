cd(@__DIR__)
#using Pkg;
#Pkg.activate(".");
#Pkg.instantiate();

using DifferentialEquations
using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra, DiffEqSensitivity, Optim
using DiffEqFlux, Flux
using Plots
#using BSON: @load, @save
using Zygote
using JLD2
using HDF5
gr()

output_figures="Figures/"
output_models="Models/"

g=9.8
L=2.0
mu=0.1
mu1 = 0.1
mu2 = 0.5
mu3 = 1.3
results=[]

function poly_friction(θ_dot)
    -(mu1*θ_dot^2 + mu2*θ_dot + mu3)
end

function linear_friction(θ_dot)
    -mu*θ_dot
end

function get_double_theta_dot(θ, θ_dot)
    ploy_friction(θ_dot) - (g/L)*sin(θ)
end

function pendulum_solver(θ, θ_dot, t)
    Δt = 0.01
    for t in 0:Δt:t
        append!(results, θ)
        θ_double_dot = get_double_theta_dot(θ, θ_dot)
        θ += θ_dot * Δt
        θ_dot += θ_double_dot * Δt
    end
    θ
end

theta = pendulum_solver(pi/2, 0, 20.0)

plot(results)
savefig(output_figures*"plot_fd.png")
#############
# use ODE solver
###########


function pendulum_ode(u, p, t)
    θ = u[1]
    θ_dot = u[2]
    mu = p[1]
    L = p[2]
    [θ_dot, poly_friction(θ_dot) - (g/L)*sin(θ)]
end

#U0 = convert(Array{Float32}, [pi/3,0])
U0 = Float32[pi/2, 0]
tspan = (0.0,3.0)
Δt = 0.01
p = Float32[0.1,2.0]

prob = ODEProblem(pendulum_ode, U0, tspan, p, saveat=Δt)
sol = solve(prob, Tsit5(), saveat=Δt)
res = Array(sol)

plot(res[1,:])
#plot!(res[2,:])
savefig(output_figures*"plot_ode.png")
scatter(res[1,:], res[2,:])
savefig(output_figures*"scatter_ode.png")
##################################
#### Neural Network
##########################

ann = FastChain(FastDense(1, 32, tanh),FastDense(32, 32, tanh),FastDense(32, 1))
p_ann = initial_params(ann)

function nn_ode(u, p, t)
    θ = u[1]
    θ_dot = u[2]
    L = 2.0
    resist = ann([θ_dot], p)
    [θ_dot, -resist[1] - (g/L)*sin(θ)]
end

#p = [2.0, p_ann]
#p = p_ann

prob_nn = ODEProblem(nn_ode, U0, tspan, p_ann)
s = solve(prob_nn, Tsit5(), saveat = Δt)

plot(s)
plot!(sol)

function predict(θ)
    Array(solve(prob_nn, Vern7(), u0=U0, p=θ, saveat = Δt,
                         abstol=1e-6, reltol=1e-6,
                         sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
end

# No regularisation right now
function loss(θ)
    pred = predict(θ)
    sum(abs2, res .- pred), pred # + 1e-5*sum(sum.(abs, params(ann)))
end

loss(p_ann)

const losses = []
callback(θ,l,pred) = begin
    push!(losses, l)
    if length(losses)%50==0
        println(losses[end])
    end
    false
end

res1 = DiffEqFlux.sciml_train(loss, p_ann, ADAM(0.01), cb=callback, maxiters = 100)
res2 = DiffEqFlux.sciml_train(loss, res1.minimizer, BFGS(initial_stepnorm=0.01), cb=callback, maxiters = 10000)

# Plot the losses
plot(losses, yaxis = :log, xaxis = :log, xlabel = "Iterations", ylabel = "Loss")
savefig(output_figures*"loss.png")

# Plot the data and the approximation
NNsolution = predict(res2.minimizer)
# Trained on noisy data vs real solution
plot(NNsolution')
plot!(res')
savefig(output_figures*"plot_nn.png")
scatter(NNsolution[1,:],NNsolution[2,:])
savefig(output_figures*"scatter_nn.png")
weights = res2.minimizer
#@save output_models*"pendulum_nn.jld2" ann weights
#@load output_models*"pendulum_nn.jld2" ann weights

fid=h5open(output_models*"pendulum_nn.h5","w")
fid["weights"] = weights
close(fid)

NNsolution = predict(weights)
plot(NNsolution')
plot!(res')

ann = FastChain(FastDense(1, 32, tanh),FastDense(32, 32, tanh),FastDense(32, 1))
#weights = initial_params(ann)

#@load output_models*"pendulum_nn.jld2" ann weights
weights = h5read(output_models*"fwi_1d_nn.h5", "weights")

U0 = Float32[pi-0.01,0]
tspan = (0.0,40.0)
Δt = 0.01
p = Float32[0.1,2.0]

prob_nn = ODEProblem(nn_ode, U0, tspan, weights)
s = solve(prob_nn, Tsit5(), saveat = Δt)
res = Array(s)
plot(res[1,:])
scatter(res[1,:],res[2,:])
