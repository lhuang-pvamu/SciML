cd(@__DIR__)
using Pkg;
Pkg.activate("..");
#Pkg.activate("~/.julia/environments/v1.5/Project.toml");
Pkg.instantiate();

using DifferentialEquations, Flux, Optim, DiffEqFlux, DiffEqSensitivity
using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra
using Plots
using Zygote
using HDF5
using BSON: @save,@load
using Surrogates
using NPZ
using ImageFiltering
gr()

output_figures="Figures/"
output_models="Models/"

g = 9.81                            # gravitational acceleration [m/s²]
L=2.0
mu=0.1
mu1 = 0.5
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
    #poly_friction(θ_dot) -
    (g/L)*sin(θ)
end

function pendulum_solver(θ, θ_dot, t)
    Δt = 0.001
    for t in 0:Δt:t
        append!(results, θ)
        θ_double_dot = get_double_theta_dot(θ, θ_dot)
        θ += θ_dot * Δt
        θ_dot += θ_double_dot * Δt
    end
    θ
end

theta = pendulum_solver(pi/2, 0.0, 20.0)

plot(results)
savefig(output_figures*"plot_fd.png")

#-----------
# get θ and θ̂ from video data
#-----------

L = 0.57
Δt = 1/60.0
angles0 = npzread("Data/angles.npy")
angles0_hat = [(angles0[i]-angles0[i-1])/Δt for i=2:length(angles0)]
ker = ImageFiltering.Kernel.gaussian((3,))
angles0_hat = imfilter(angles0_hat, ker)
pushfirst!(angles0_hat, angles0_hat[1])
angles0 = (hcat(angles0,angles0_hat))'
plot(angles0')
#θ = angles0[29:300]
#θ̂ = [(θ[i]-θ[i-1])/Δt for i=2:length(θ)]
#pushfirst!(θ̂, 0.0)
#angles = (hcat(θ,θ̂))'
angles = angles0[:,29:300]
plot(angles')
print(angles[1])
tspan = (0.0, length(angles[1,:])/60.0)
u0 = angles[:,1]

##################################
# use ODE solver with friction
#################################

function pendulum_ode(u, p, t)
    θ = u[1]
    θ_dot = u[2]
    mu = p[1]
    L = p[2]
    [θ_dot, poly_friction(θ_dot) - (g/L)*sin(θ)]
end

#U0 = convert(Array{Float32}, [pi/3,0])
U0 = Float32[pi/2, 0.0]
U0 = u0
tspan = (0.0,10.0)
Δt = 0.1
p = Float32[0.1,L]

prob = ODEProblem(pendulum_ode, U0, tspan, p)
sol = solve(prob, Tsit5(), saveat=Δt)
data = Array(sol)

plot(data')
plot!(angles0[:,29:end]')
#plot!(res[2,:])
savefig(output_figures*"plot_ode.png")
plot(data[1,:], data[2,:], marker = :hex)
savefig(output_figures*"scatter_ode.png")

x = L*sin.(data[1,:])
y = -L*cos.(data[1,:])
anim = @animate for i ∈ 1:1:length(x)
    plot([(0,0),(x[i],y[i])], marker = :hex, xlims=(-3,3), ylims=(-2.5,0.5),
    title="$(round((i-1)*Δt,digits=1)) second", legend = false)
end
gif(anim, "anim_friction.gif", fps = 10)


####################################
# Nonhomogeneous Equations
# M is an external torque (say by a wind or motor)
# L: length, m: mass, g: gravity
####################################

m = 1.0                             # mass[m]

function pendulum!(du,u,p,t)
    du[1] = u[2]                    # θ'(t) = ω(t)
    du[2] = poly_friction(du[1]) -(g/L)*sin(u[1]) + 3/(m*L^2)*p(u[1]) # ω'(t) = -3g/(2l) sin θ(t) + 3/(ml^2)M(t)
end

θ₀ = pi/2                           # initial angular deflection [rad]
ω₀ = 0.0                            # initial angular velocity [rad/s]
u₀ = [θ₀, ω₀]                       # initial state vector
tspan = (0.0,10.0)                  # time interval
Δt = 0.01
M = θ->2.0*cos(θ)                    # external torque [Nm]

prob = ODEProblem(pendulum!,u₀,tspan,M)
#sol = solve(prob, Tsit5(), saveat=Δt)
sol = solve(prob, Vern7(), saveat=Δt)
data = Array(sol)
plot(sol,linewidth=2,xaxis="t",label=["θ [rad]" "ω [rad/s]"],layout=(2,1))
savefig(output_figures*"pendulum_torque_10_01.png")
#scatter(data[1,:], data[2,:])
plot(data[1,:], data[2,:], xaxis="θ", yaxis="ω", label="phase", marker = :hex)
savefig(output_figures*"scatter_torque_10_01.png")

@userplot CirclePlot
@recipe function f(cp::CirclePlot)
    x, y, i = cp.args
    n = length(x)
    inds = i:min(i+25,n)
    xlims := (-4,10)
    ylims := (-5,5)
    grid --> true
    linewidth --> range(0, 5, length = 25)
    seriesalpha --> range(0, 1, length = 25)
    aspect_ratio --> 1
    label --> false
    x[inds], y[inds]
end

anim = @animate for i ∈ 1:length(data[1,:])
    circleplot(data[1,:], data[2,:], i, cbar = false, framestyle = :zerolines)
end
gif(anim, output_figures*"phase_fps10.gif", fps = 10)

plt = plot(1, xlim=(-4,10), ylim=(-5,5),
                title = "Pendulum Phase Space", marker = 2,legend = false, framestyle = :zerolines)

anim = @animate for i ∈ 1:1:length(data[1,:])
    push!(plt, data[1,i], data[2,i])
end

gif(anim, output_figures*"anim_phase_10.gif", fps = 10)

x = L*sin.(data[1,:])
y = -L*cos.(data[1,:])

anim = @animate for i ∈ 1:1:length(x)
    plot([(0,0),(x[i],y[i])], marker = :hex, xlims=(-3,3), ylims=(-2.5,2.5),
    title="$(round((i-1)*0.1,digits=1)) second", legend = false)
end

gif(anim, output_figures*"anim_torque_10.gif", fps = 10)

#############################################
#### Universal Neural Differential Equation
#### Learn the torque (time independant) and the polynomial friction
#############################################

#ann = FastChain(FastDense(4, 64, tanh),FastDense(64, 64, tanh),FastDense(64, 1))
ann = FastChain(FastDense(2,64,tanh), FastDense(64, 64, tanh),FastDense(64,1))
if isfile(output_models*"pendulum_nn.h5")
    p_ann = h5read(output_models*"pendulum_nn.h5", "weights")
else
    p_ann = initial_params(ann)
end

#U0 = Float32[pi/2, 0.0]
U0 = angles[:,1]
m = 1.0
last = 200
data = angles0[:,29:last]
function nn_ode(u, p, t)
    θ = u[1]
    θ_dot = u[2]
    #torque = ann([θ_dot, m, L, θ], p)
    torque = ann([θ_dot, L], p)
    [θ_dot, torque[1] - (g/L)*sin(θ)]
end

#p = [2.0, p_ann]
#p = p_ann
Δt = 1/60.0
tspan = (0.0, length(angles0[1,29:last])/60.0)
prob_nn = ODEProblem(nn_ode, U0, tspan, p_ann)
s = solve(prob_nn, Tsit5(), saveat = Δt)

#plot(Array(s)')
#plot!(data')
plot(Array(s)',linewidth=2,xaxis="t",label=["θ [rad]" "ω [rad/s]"],layout=(2,1))
plot!(angles0[:,29:last]',linewidth=2,xaxis="t",label=["θ [rad]" "ω [rad/s]"],layout=(2,1))
savefig(output_figures*"initial_nn.png")

function predict(θ)
    Array(solve(prob_nn, Tsit5(), u0=U0, p=θ, saveat = Δt,
                         abstol=1e-6, reltol=1e-6,
                         sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
end

# No regularisation right now
function loss(θ)
    pred = predict(θ)
    sum(abs2, data .- pred[:,1:length(data[1,:])]), pred # + 1e-5*sum(sum.(abs, params(ann)))
end

loss(p_ann)

const losses = []
callback(θ,l,pred) = begin
    push!(losses, l)
    if length(losses)%5==0
        println(losses[end])
        pl = plot(data')
        plot!(pl, pred')
        display(plot(pl))
    end
    false
end

# train using 10-second data with Δt=0.1
res1 = DiffEqFlux.sciml_train(loss, p_ann, ADAM(0.01), cb=callback, maxiters = 1000)
res2 = DiffEqFlux.sciml_train(loss, res1.minimizer, BFGS(initial_stepnorm=0.01), cb=callback, maxiters = 10000)

# Plot the losses
plot(losses, yaxis = :log, xaxis = :log, xlabel = "Iterations", ylabel = "Loss", legend = false)
savefig(output_figures*"loss.png")

# Plot the data and the approximation
NNsolution = predict(res2.minimizer)
# Trained on noisy data vs real solution
#plot(NNsolution')
#plot!(data')
plot(s.t, NNsolution',linewidth=2,xaxis="t",label=["θ [rad]" "ω [rad/s]"],layout=(2,1))
savefig(output_figures*"plot_nn_10s_01.png")
plot(NNsolution[1,:],NNsolution[2,:],marker=:hex)
savefig(output_figures*"scatter_nn_10s_01.png")
weights = res2.minimizer
#@save output_models*"pendulum_nn.jld2" ann weights
#@load output_models*"pendulum_nn.jld2" ann weights
plot(NNsolution',linewidth=2,xaxis="t",label=["θ [rad]" "ω [rad/s]"],layout=(2,1))
plot!(data',linewidth=2,xaxis="t",label=["θ [rad]" "ω [rad/s]"],layout=(2,1))
savefig(output_figures*"comp_ode_nn_10s_01.png")

fid=h5open(output_models*"pendulum_nn.h5","w")
fid["weights"] = weights
close(fid)

NNsolution = predict(weights)
plot(NNsolution')
plot!(data')

ann = FastChain(FastDense(2, 64, tanh),FastDense(64, 64, tanh),FastDense(64, 1))
#weights = initial_params(ann)

#@load output_models*"pendulum_nn.jld2" ann weights
weights = h5read(output_models*"pendulum_nn.h5", "weights")

# extend to 30 seconds with Δt=0.01

#U0 = Float32[pi/2,0.0]
U0 = angles[:,1]
tspan = (0.0,10.0)
#Δt = 0.01
#p = Float32[0.1,2.0]

prob_nn = ODEProblem(nn_ode, U0, tspan, weights)
s = solve(prob_nn, Tsit5(), saveat = Δt)
#s = solve(prob_nn, Vern7(), saveat = Δt)
res = Array(s)
plot(res[1,:])
plot(res',linewidth=2,xaxis="t",label=["θ [rad]" "ω [rad/s]"],layout=(2,1))
plot!(angles0[:,29:end]',linewidth=2,xaxis="t",label=["θ [rad]" "ω [rad/s]"],layout=(2,1))
savefig(output_figures*"plot_nn_30s_001.png")
plot(res[1,:],res[2,:], marker=:hex)
savefig(output_figures*"scatter_nn_30s_001.png")

prob = ODEProblem(pendulum!,U0,tspan,M)
sol = solve(prob, Tsit5(), saveat=Δt)
data = Array(sol)

# results should match data
plot(sol.t, res',linewidth=2,xaxis="t",label=["predicted θ [rad]" "predicted ω [rad/s]"],layout=(2,1))
plot!(sol.t, data',linewidth=2,xaxis="t",label=["True θ [rad]" "True ω [rad/s]"],layout=(2,1))
savefig(output_figures*"comp_data_nn_30s_001.png")
# display what the neural network ann learned

θ_dot_range = -3:0.1:3.0
θ_range = -1.5:0.1:1.5

torque_ann(x,y) =  ann([y, m, L, x], weights)[1]
torque(x,y) = poly_friction(y) + 3/(m*L^2)*2.0*cos(x)
torque_diff(x,y) = torque(x,y) - torque_ann(x,y)
surface(θ_range, θ_dot_range, torque)
surface!(θ_range, θ_dot_range, torque_ann)
savefig(output_figures*"comp_torque_nn.png")
surface(θ_range, θ_dot_range, torque_diff)
savefig(output_figures*"diff_torque_nn.png")

DX = Array(sol(sol.t, Val{1}))
plot(DX')
plot!(data')

# Create a Basis
@variables u[1:2]
# Lots of polynomials
polys = Operation[1]
for i ∈ 1:3
    push!(polys, u[1]^i)
    push!(polys, u[2]^i)
    for j ∈ i:3
        if i != j
            push!(polys, (u[1]^i)*(u[2]^j))
            push!(polys, u[2]^i*u[1]^i)
        end
    end
end

X = zeros(length(θ_range), length(θ_dot_range))
X = [[x, y] for x in θ_range, y in θ_dot_range][:]
X[:,:]
Y = [torque(x[1], x[2]) for x in X]
X = hcat(X)
Y[:]
# And some other stuff
h = [cos.(u)...; sin.(u)...; polys...]
basis = Basis(h, u)

# Create an optimizer for the SINDY problem
opt = SR3()
# Create the thresholds which should be used in the search process
λ =  exp10.(-6:0.1:2)
# Target function to choose the results from; x = L0 of coefficients and L2-Error of the model
#f_target(x, w) = iszero(x[1]) ? Inf : norm(w.*x, 2)
opt = STRRidge(0.1)

# Test on original data and without further knowledge
Ψ = SINDy(X, Y, basis, opt, maxiter = 100)
println(Ψ)
print_equations(Ψ)
get_error(Ψ)
print(parameters(Ψ))

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
tspan = (0.0,10.0)
u0 = Float32[pi/3, 0.0]
Δt = 0.1
L = 2.0
m = 1.0
prob = ODEProblem(pendulum!,u0,tspan,M)
sol = solve(prob, Tsit5(), saveat=Δt)
data = Array(sol)


# Use video data
tspan = (0.0, length(angles[1,:])/60.0)
#dudt = Chain(x->[x[1],x[2],L, m],Dense(4,32,tanh),Dense(32,2)) #, y->[y[1],y[2],g/L])
#x->[x[1],x[2],-g/L],
dudt = Chain(x->[x[1],x[2],-g/L],Dense(3,32,tanh),Dense(32,2))
u0 = angles[:,1]
n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=Δt,reltol=1e-6,abstol=1e-7)
if isfile(output_models*"pendulum_n_ode_weights.bson")
    @load output_models*"pendulum_n_ode_weights.bson" ps
    Flux.loadparams!(n_ode,ps)
end
ps = Flux.params(n_ode)

function predict_n_ode()
  n_ode(u0)
end
d = predict_n_ode()
length(angles[1,:])
Array(d)
loss_n_ode() = sum(abs2, angles .- Array(predict_n_ode())[:,1:length(angles[1,:])])

t = tspan[1]:Δt:tspan[2]

it = Iterators.repeated((), 1000)
opt = ADAM(0.001)
cb = function () #callback function to observe training
  display(loss_n_ode())
  # plot current prediction against data
  cur_pred = predict_n_ode()
  #pl = scatter(angles[1,:],angles[2,:],label="data")
  #scatter!(pl,cur_pred[1,:], cur_pred[2,:],label="prediction")
  pl = plot(angles',label="data")
  plot!(pl,cur_pred',label="prediction")
  display(plot(pl))
end

# Display the ODE with the initial parameter values.
cb()

Flux.train!(loss_n_ode, ps, it, opt, cb = cb)

tspan = (0.0, length(angles0[1,:])/60.0)
res = n_ode(u0)
d = hcat(res.u)
scatter(d[1,:], d[2,:])
plot(d')
plot!(angles0[:,29:end]')

@save output_models*"pendulum_dudt.bson" dudt
@save output_models*"pendulum_n_ode.bson" n_ode
@save output_models*"pendulum_n_ode_weights.bson" ps

L=2.0
m=1.0
tspan = (0.0,10.0)
#Δt = 0.01
#dudt1 = Chain(x->[x[1],x[2],L, m],Dense(4,32,tanh),Dense(32,2)) #, y->[y[1],y[2],g/L])
n_ode1 = NeuralODE(dudt,tspan,Tsit5(),saveat=Δt,reltol=1e-7,abstol=1e-9)

@load output_models*"pendulum_n_ode_weights.bson" ps

Flux.loadparams!(n_ode1,ps)

#u0 = Float32[pi/2.0, 0.0]
res1 = n_ode1(u0)
d1 = hcat(res1.u)
scatter(d1[1,:], d1[2,:])
plot(res1.t, d1[1,:])
plot(d1')
plot!(angles0[:,29:end]')


prob = ODEProblem(pendulum!,u0,tspan,M)
sol = solve(prob, Tsit5(), saveat=Δt)
data = Array(sol)
plot!(sol.t,data[1,:])
