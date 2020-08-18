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
using RecursiveArrayTools
#using BSON: @load, @save
using Zygote
using JLD2
gr()


g=9.8
L=2.0
mu=0.1
results=[]

function get_double_theta_dot(θ, θ_dot)
    -mu*θ_dot - (g/L)*sin(θ)
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

theta = pendulum_solver(pi/2, 0, 10.0)

plot(results)

#############
# use ODE solver
###########


function pendulum_ode(u, p, t)
    θ = u[1]
    θ_dot = u[2]
    mu = p[1]
    L = p[2]
    [θ_dot, -mu*θ_dot - (g/L)*sin(θ)]
end

#U0 = convert(Array{Float32}, [pi/3,0])
U0 = Float32[pi/3,0]
tspan = (0.0,5.0)
Δt = 0.01
p = Float32[0.1,2.0]

prob = ODEProblem(pendulum_ode, U0, tspan, p, saveat=Δt)
sol = solve(prob, Tsit5(), saveat=Δt)
res = Array(sol)

plot(res[1,:])

plot(res[2,:])

scatter(res[1,:], res[2,:])

##################################
#### Neural Network
##########################

ann = FastChain(FastDense(1, 32, tanh),FastDense(32, 1))
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

# Plot the data and the approximation
NNsolution = predict(res2.minimizer)
# Trained on noisy data vs real solution
plot(NNsolution')
plot!(res')
weights = res2.minimizer
@save "pendulum_nn.jld2" ann weights
@load "pendulum_nn.jld2" ann weights

NNsolution = predict(weights)
plot(NNsolution')
plot!(res')

ann = FastChain(FastDense(1, 32, tanh),FastDense(32, 1))
weights = initial_params(ann)

@load "pendulum_nn.jld2" ann weights

U0 = Float32[pi-0.01,0]
tspan = (0.0,60.0)
Δt = 0.01
p = Float32[0.1,2.0]

prob_nn = ODEProblem(nn_ode, U0, tspan, weights)
s = solve(prob_nn, Tsit5(), saveat = Δt)
res = Array(s)
plot(res[1,:])
scatter(res[1,:],res[2,:])
