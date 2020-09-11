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
using BSON: @save,@load
gr()

output_figures="Figures/"
output_models="Models/"

g = 9.81                            # gravitational acceleration [m/s²]
L=2.0
mu=0.1
mu1 = 0.01
mu2 = 0.2
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
tspan = (0.0,30.0)
Δt = 0.1
p = Float32[0.1,L]

prob = ODEProblem(pendulum_ode, U0, tspan, p)
sol = solve(prob, Tsit5(), saveat=Δt)
data = Array(sol)

plot(data[1,:])
#plot!(res[2,:])
savefig(output_figures*"plot_ode.png")
scatter(data[1,:], data[2,:])
savefig(output_figures*"scatter_ode.png")

x = L*sin.(data[1,:])
y = -L*cos.(data[1,:])
anim = @animate for i ∈ 1:1:length(x)
    plot([(0,0),(x[i],y[i])], marker = :hex, xlims=(-3,3), ylims=(-2.5,0.5),
    title="$(round((i-1)*Δt,digits=1)) second", legend = false)
end
gif(anim, "anim_fps30.gif", fps = 10)


####################################
# Nonhomogeneous Equations
# M is an external torque (say by a wind or motor)
# L: length, m: mass, g: gravity
####################################

m = 1.0                             # mass[m]

function pendulum!(du,u,p,t)
    du[1] = u[2]                    # θ'(t) = ω(t)
    du[2] = poly_friction(du[1]) -3g/(2L)*sin(u[1]) + 3/(m*L^2)*p(u[1]) # ω'(t) = -3g/(2l) sin θ(t) + 3/(ml^2)M(t)
end

θ₀ = pi/2                           # initial angular deflection [rad]
ω₀ = 0.0                            # initial angular velocity [rad/s]
u₀ = [θ₀, ω₀]                       # initial state vector
tspan = (0.0,10.0)                  # time interval
Δt = 0.1
M = t->2.0*cos(t)                    # external torque [Nm]

prob = ODEProblem(pendulum!,u₀,tspan,M)
sol = solve(prob, Tsit5(), saveat=Δt)
data = Array(sol)
plot(sol,linewidth=2,xaxis="t",label=["θ [rad]" "ω [rad/s]"],layout=(2,1))
scatter(data[1,:], data[2,:])

x = L*sin.(data[1,:])
y = -L*cos.(data[1,:])

anim = @animate for i ∈ 1:1:length(x)
    plot([(0,0),(x[i],y[i])], marker = :hex, xlims=(-3,3), ylims=(-2.5,2.5),
    title="$(round((i-1)*0.1,digits=1)) second", legend = false)
end

gif(anim, "anim_fps10.gif", fps = 10)

#############################################
#### Universal Neural Differential Equation
#### Learn the torque (time independant) and the polynomial friction
#############################################

ann = FastChain(FastDense(4, 64, tanh),FastDense(64, 64, tanh),FastDense(64, 1))
p_ann = initial_params(ann)
U0 = Float32[pi/2, 0.0]
m = 1.0
function nn_ode(u, p, t)
    θ = u[1]
    θ_dot = u[2]
    torque = ann([θ_dot, m, L, θ], p)
    [θ_dot, torque[1] - 3g/(2L)*sin(θ)]
end

#p = [2.0, p_ann]
#p = p_ann

prob_nn = ODEProblem(nn_ode, U0, tspan, p_ann)
s = solve(prob_nn, Tsit5(), saveat = Δt)

plot(Array(s)')
plot!(data')

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
    if length(losses)%10==0
        println(losses[end])
        pl = plot(data')
        plot!(pl, pred')
        display(plot(pl))
    end
    false
end

# train using 10-second data with Δt=0.1
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
plot(NNsolution',linewidth=2,xaxis="t",label=["θ [rad]" "ω [rad/s]"],layout=(2,1))
plot!(data',linewidth=2,xaxis="t",label=["θ [rad]" "ω [rad/s]"],layout=(2,1))


fid=h5open(output_models*"pendulum_nn.h5","w")
fid["weights"] = weights
close(fid)

NNsolution = predict(weights)
plot(NNsolution')
plot!(data')

ann = FastChain(FastDense(4, 64, tanh),FastDense(64, 64, tanh),FastDense(64, 1))
#weights = initial_params(ann)

#@load output_models*"pendulum_nn.jld2" ann weights
weights = h5read(output_models*"pendulum_nn.h5", "weights")

# extend to 20 seconds with Δt=0.01

U0 = Float32[pi/2,0.0]
tspan = (0.0,20.0)
Δt = 0.01
p = Float32[0.1,2.0]

prob_nn = ODEProblem(nn_ode, U0, tspan, weights)
s = solve(prob_nn, Tsit5(), saveat = Δt)
res = Array(s)
plot(res[1,:])
savefig(output_figures*"plot_nn.png")
scatter(res[1,:],res[2,:])
savefig(output_figures*"scatter_nn.png")

prob = ODEProblem(pendulum!,U0,tspan,M)
sol = solve(prob, Tsit5(), saveat=Δt)
data = Array(sol)

# results should match data
plot(res',linewidth=2,xaxis="t",label=["θ [rad]" "ω [rad/s]"],layout=(2,1))
plot!(data',linewidth=2,xaxis="t",label=["θ [rad]" "ω [rad/s]"],layout=(2,1))

# display what the neural network ann learned

θ_dot_range = -3:0.1:3.0
θ_range = -1.5:0.1:1.5

torque_ann(x,y) =  ann([y, m, L, x], weights)[1]
torque(x,y) = poly_friction(y) + 3/(m*L^2)*2.0*cos(x)
torque_diff(x,y) = torque(x,y) - torque_ann(x,y)
surface(θ_range, θ_dot_range, torque)
surface!(θ_range, θ_dot_range, torque_ann)
surface(θ_range, θ_dot_range, torque_diff)

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
opt = ADAM(0.001)
cb = function () #callback function to observe training
  display(loss_rd())
  # using `remake` to re-create our `prob` with current parameters `p`
  #display(plot(solve(remake(prob,u0=u0,p=p),Tsit5(),saveat=Δt)[1,:],ylim=(-5,5)))
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

dudt = Chain(x->[x[1],x[2],-g/L],Dense(3,32,tanh),Dense(32,2)) #, y->[y[1],y[2],g/L])
n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=Δt,reltol=1e-7,abstol=1e-9)
if isfile(output_models*"pendulum_n_ode_weights.bson")
    @load output_models*"pendulum_n_ode_weights.bson" ps
    Flux.loadparams!(n_ode,ps)
end
ps = Flux.params(n_ode)

UO_node = [pi/2, 0.0]

function predict_n_ode()
  n_ode(UO_node)
end
loss_n_ode() = sum(abs2,data .- predict_n_ode())

t = tspan[1]:Δt:tspan[2]

it = Iterators.repeated((), 100)
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

res = n_ode(Float32[pi/2,0.0])
d = hcat(res.u)
scatter(d[1,:], d[2,:])
plot(d[1,:])
plot!(data[1,:])

@save output_models*"pendulum_dudt.bson" dudt
@save output_models*"pendulum_n_ode.bson" n_ode
@save output_models*"pendulum_n_ode_weights.bson" ps

L=2.0
tspan = (0.0,10.0)
dudt1 = Chain(x->[x[1],x[2],-g/L],Dense(3,32,tanh),Dense(32,2)) #, y->[y[1],y[2],g/L])
n_ode1 = NeuralODE(dudt,tspan,Tsit5(),saveat=Δt,reltol=1e-7,abstol=1e-9)

@load output_models*"pendulum_n_ode_weights.bson" ps

Flux.loadparams!(n_ode1,ps)

res1 = n_ode1([pi/3, 0.0])
d1 = hcat(res1.u)
scatter(d1[1,:], d1[2,:])
plot(d1[1,:])
plot!(data[1,:])
