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
L=2.0
mu=0.1
mu1 = 0.1
mu2 = 0.5
mu3 = 1.3
results=[]

function poly_friction(θ_dot)
    -(mu1*θ_dot^2 + mu2*θ_dot)
end

function linear_friction(θ_dot)
    -mu*θ_dot
end

function get_double_theta_dot(θ, θ_dot)
    poly_friction(θ_dot) - (g/L)*sin(θ)
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

####################
# use ODE solver
####################

function pendulum_ode(u, p, t)
    θ = u[1]
    θ_dot = u[2]
    mu = p[1]
    L = p[2]
    [θ_dot, poly_friction(θ_dot) - (g/L)*sin(θ)]
end

#U0 = convert(Array{Float32}, [pi/3,0])
U0 = Float32[pi/2, 0.0]
tspan = (0.0,10.0)
Δt = 0.01
p = Float32[0.1,2.0]

prob = ODEProblem(pendulum_ode, U0, tspan, p)
sol = solve(prob, Tsit5(), saveat=Δt)
data = Array(sol)

plot(data[1,:])
#plot!(res[2,:])
savefig(output_figures*"plot_ode.png")
scatter(data[1,:], data[2,:])
savefig(output_figures*"scatter_ode.png")

##################################
#### Neural Network
##################################

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
    sum(abs2, data .- pred), pred # + 1e-5*sum(sum.(abs, params(ann)))
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
plot!(data')
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
plot!(data')

ann = FastChain(FastDense(1, 32, tanh),FastDense(32, 32, tanh),FastDense(32, 1))
#weights = initial_params(ann)

#@load output_models*"pendulum_nn.jld2" ann weights
weights = h5read(output_models*"pendulum_nn.h5", "weights")

U0 = Float32[pi/2,0.0]
tspan = (0.0,10.0)
Δt = 0.01
p = Float32[0.1,2.0]

prob_nn = ODEProblem(nn_ode, U0, tspan, weights)
s = solve(prob_nn, Tsit5(), saveat = Δt)
res = Array(s)
plot(res[1,:])
savefig(output_figures*"plot_nn.png")
scatter(res[1,:],res[2,:])
savefig(output_figures*"scatter_nn.png")

##################################
##  Optimize for initial values
##################################

u0 = Float32[pi, 0.0]
pa = Flux.params(u0)
p= Float32[0.1, 2.0]

function predict_rd() # Our 1-layer "neural network"
  Array(solve(prob,Tsit5(),u0=u0, p=p,saveat=Δt)) # override with new parameters
end

loss_rd() = sum(abs2,  predict_rd() - data) # loss function

it = Iterators.repeated((), 1000)
opt = ADAM(0.01)
cb = function () #callback function to observe training
  display(loss_rd())
  # using `remake` to re-create our `prob` with current parameters `p`
  display(plot(solve(remake(prob,u0=u0,p=p),Tsit5(),saveat=Δt)[1,:],ylim=(0,8)))
end

# Display the ODE with the initial parameter values.
cb()

Flux.train!(loss_rd, pa, it, opt, cb = cb)
display(u0)
res = predict_rd()
plot(res[1,:])
plot!(data[1,:])

###########################################
## Train a Neural ODE network to fit data
###########################################

dudt = Chain(Dense(2,32,tanh),Dense(32,32,tanh),Dense(32,2))
n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=Δt,reltol=1e-7,abstol=1e-9)
ps = Flux.params(n_ode)

function predict_n_ode()
  n_ode(U0)
end
loss_n_ode() = sum(abs2,data .- predict_n_ode())

t = tspan[1]:Δt:tspan[2]

it = Iterators.repeated((), 1000)
opt = ADAM(0.01)
cb = function () #callback function to observe training
  display(loss_n_ode())
  # plot current prediction against data
  cur_pred = predict_n_ode()
  pl = scatter(t, data[1,:],label="data")
  scatter!(pl,t, cur_pred[1,:],label="prediction")
  display(plot(pl))
end

# Display the ODE with the initial parameter values.
cb()

Flux.train!(loss_n_ode, ps, it, opt, cb = cb)


function solve(stacks)::Array{Tuple{Int, Int}}
	sol = []
	moves = [(1,2),(1,3),(2,3),(3,2),(2,1),(3,1)]
	print("Start")
	#what to do?
	while !iscomplete(stacks)
        new_mv = false
		for mv in moves
			new_stacks = move(stacks,mv[1],mv[2])
			if islegal(new_stacks)
				stacks = new_stacks
				push!(sol,mv)
                new_mv = true
				break
			end
		end
        if new_mv == false
            print("no move")
            break
        end
	end
	return sol
end
