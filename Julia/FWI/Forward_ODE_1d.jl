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
using JLD2
#using RecursiveArrayTools
gr()
include("Forward_1d.jl")

output_figures="Figures/"
output_models="Models/"

#config = Dict()
#config["dx"] = 0.005
#config["x"] = 0.0:config["dx"]:1.0

println("Threads: ",Threads.nthreads())

c, c0 = velocity_model()

plot(c)

set_config!(config, c)

js = argmin(abs.(config["x"] .- config["x_s"]))
S = zero(c)
S[2:200]
w = ricker(0.1, 10)
S[js] = 1/ config["dx"]
function setS(t)
    S .* ricker(t,10)
end

NS = length(config["x"])-1
setS(0.2)

function set_matrics_ode(c)
    NS = size(c,1)-1
    dx = config["dx"]
    cR = c[2:NS]
    #M1 = Diagonal(1 ./(cR.^2))
    M = zeros(NS-1,NS-1)
    for i in 1:NS-1
        M[i,i] = 1/cR[i]^2
    end
    K = (Diagonal(2*ones(NS-1))+diagm(1 => ones(NS - 2) * -1, -1 => ones(NS - 2) * -1))/(dx*dx)
    K = hcat(zeros(NS-1,1),K, zeros(NS-1,1))
    #display(K)
    K[1,1] = -1.0/(dx*dx)
    K[NS-1,NS+1] = -1.0/(dx*dx)
    #display(K)
    MI = inv(M)
    #display(MI)
    M, K, MI
end


U0 = zeros(2*NS)
dx = config["dx"]
function wave(u, p, t)
    (c0, M, K, MI) = p
    NS = size(c,1)-1
    U = u[1:NS+1]
    V = u[NS+2:end]
    S1 = setS(t)
    du0dt = c[1]*(2/dx*(U[2]-U[1]) - V[1]/c[2])
    duNdt = c[NS+1]*(2/dx*(U[NS]-U[NS+1]) - V[NS-1]/c[NS])
    W = MI * (S1[2:NS] - (K * U))
    vcat([du0dt], V, [duNdt], W)
end

function forward_ODE_driver(c)
    tspan = (0.0,3.0)
    M, K, MI = set_matrics_ode(c)
    p = (c, M, K, MI)
    prob = ODEProblem(wave, U0, tspan, p, saveat=config["dt"])
    #sol = solve(prob, Tsit5(), saveat=config["dt"])
    sol = solve(prob, Vern7(), saveat=config["dt"])
    res = Array(sol)
    Z = transpose(res[1:NS+1,:])
    #heatmap(Z)
    traces = record_data(Z)
    Z, traces
end

U, traces = forward_ODE_driver(c)
heatmap(U)
savefig(output_figures*"heatmap_ODE.png")
plot(traces[1,:])
plot!(traces[2,:])
plot!(traces[3,:])
savefig(output_figures*"plot_ODE_traces.png")

###### SciML using ANN #######

ann = FastChain(FastDense(3, 32, tanh),FastDense(32, 32, tanh),FastDense(32, 1))
p = initial_params(ann)
M, K, MI = set_matrics_ode(c)

function wave_ann(u, p, t)
    #(c0, M, K, MI, ap) = p
    NS = size(c,1)-1
    U = u[1:NS+1]
    V = u[NS+2:end]
    S1=setS(t)
    du0dt = c[1]*(2/dx*(U[2]-U[1])) - ann(Float32[V[1],c[1],c[2]],p)[1]  #c[1]*(2/dx*(U[2]-U[1]) - V[1]/c[2])
    duNdt = c[NS+1]*(2/dx*(U[NS]-U[NS+1]) - V[NS-1]/c[NS])
    W = MI * (S1[2:NS] - (K * U))
    vcat(du0dt, V, duNdt, W)
end

NS = size(c,1)-1
U0 = zeros(2*NS)
tspan = (0.0,1.0)
prob_nn = ODEProblem(wave_ann, U0, tspan, p)

function forward_ann(θ)
    res = Array(solve(prob_nn, Tsit5(), u0=U0, p=θ, saveat = config["dt"],
                         abstol=1e-6, reltol=1e-6,
                         sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
    Z0 = transpose(res[1:NS+1,:])
    tra = record_data(Z0)
    Z0, tra
end

Z0, tr = forward_ann(p)
heatmap(Z0)
plot(traces[1,:])
plot!(tr[1,:])

# No regularisation right now
function loss(θ)
    U0, t0 = forward_ann(θ)
    #print(size(t0),size(traces))
#    sum(abs2, traces[:,size(t0,2)] .- t0), t0 # + 1e-5*sum(sum.(abs, params(ann)))
    sum(abs2, U[1:size(U0,1),1:size(U0,2)] .- U0),U0 # + 1e-5*sum(sum.(abs, params(ann)))
end

loss(p)

const losses1 = []
callback(θ,l,pred) = begin
    push!(losses1, l)
    if length(losses1)%1==0
        println(length(losses1), ": ", losses1[end])
    end
    false
end

res1 = DiffEqFlux.sciml_train(loss, p, ADAM(0.01), cb=callback, maxiters = 100)
weights = res1.minimizer
@save output_models*"fwi_1d_nn.jld2" ann weights
Z0, tr = forward_ann(res1.minimizer)
heatmap(Z0)
savefig(output_figures*"heatmap_ODE_ann_1.png")
plot(tr[1,:])
plot!(tr[2,:])
plot!(tr[3,:])
savefig(output_figures*"plot_ODE_ann_traces_1.png")
# Plot the losses
plot(losses1, yaxis = :log, xaxis = :log, xlabel = "Iterations", ylabel = "Loss")
savefig(output_figures*"loss.png")


res2 = DiffEqFlux.sciml_train(loss, res1.minimizer, BFGS(initial_stepnorm=0.01), cb=callback, maxiters = 10000)
# Plot the losses
plot(losses1, yaxis = :log, xaxis = :log, xlabel = "Iterations", ylabel = "Loss")
savefig(output_figures*"loss.png")

display(res2.minimizer)
Z0, tr = forward_ann(res2.minimizer)
heatmap(Z0)
savefig(output_figures*"heatmap_ODE_ann.png")
plot(tr[1,:])
plot!(tr[2,:])
plot!(tr[3,:])
savefig(output_figures*"plot_ODE_ann_traces.png")

weights = res2.minimizer
@save output_models*"fwi_1d_nn.jld2" ann weights
@load output_models*"fwi_1d_nn.jld2" ann weights

Z0, tr = forward_ann(weights)
heatmap(Z0)

####### Optimization ############

function F_ODE(c0)
    NS = size(c0,1)-1
    U0 = zeros(2*NS)
    M, K, MI = set_matrics_ode(c0)
    tspan = (0.0,3.0)
    p = (c0, M, K, MI)
    prob = ODEProblem(wave, U0, tspan, p, saveat=config["dt"])
    sol = solve(prob, Tsit5())
    res = Array(sol)
    Z0 = transpose(res[1:NS+1,:])
    tr = record_data(Z0)
    sum((traces.-tr).^2)
end

F_ODE(c0)

plot(c0)

res = optimize(
    F_ODE,
    c0,
    LBFGS(),
    Optim.Options(show_trace = true, iterations = 5, store_trace=true, f_tol=1e-8),
)

summary(res)
Optim.minimum(res)
plot(res.minimizer)
plot!(c)
savefig(output_figures*"plot_ODE_optim.png")
