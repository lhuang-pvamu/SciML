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
gr()
using Zygote
using HDF5
using BSON: @save,@load
using Surrogates
using NPZ
using ImageFiltering
using StaticArrays
#using Makie

output_figures="Figures/"
output_models="Models/"
include("../../Utils/SymmetryNet.jl")

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

θ_dot_range = -2.5:0.1:2.5
θ_range = -0.8:0.1:0.8

plot(θ_dot_range, poly_friction)

#--------
# phase space for vector
#-------

meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x)))
xvalues, yvalues = meshgrid(-10:0.5:10, -5:0.5:5)
yvalues
xdot = yvalues
ydot = (-g/L*sin.(xvalues))
#(xdot,ydot)
#c = collect(-100:100)
quiver(xvalues, yvalues, quiver=(xdot,ydot)./10, c=:blue)

#-----------
# get θ and θ̂ from video data
#-----------

#L = 0.57
#m = 0.02965
fd_coff=[1/280,-4/105,1/5,-4/5,0,4/5,-1/5,4/105,-1/280]

L = 0.635
m = 0.06932
Δt = 1/60.0
angles0_raw = npzread("Data/new_angles.npy")
angles0_dot_raw = [sum(fd_coff.*angles0_raw[i-4:i+4])/Δt for i=5:length(angles0_raw)-4]
plot(angles0_raw)
plot(angles0_dot_raw)
#pushfirst!(angles0_dot_raw, angles0_dot_raw[1])
#angles0_raw = (hcat(angles0_raw,angles0_dot_raw))'
ker = ImageFiltering.Kernel.gaussian((3,))
angles0_dot = imfilter(angles0_dot_raw, ker)
angles0 = (hcat(angles0_raw[5:length(angles0_raw)-4],angles0_dot))'
plot(angles0')
#θ = angles0[29:300]
#θ̂ = [(θ[i]-θ[i-1])/Δt for i=2:length(θ)]
#pushfirst!(θ̂, 0.0)
#angles = (hcat(θ,θ̂))'
#start = 29  # the first video
start = 64   # the second long video
angles = angles0[:,start:start+300]
plot(angles')
print(angles[:,1])
tspan = (0.0, length(angles[1,:])/60.0)
u0 = angles[:,1]


## Adjust the recorded θ and θ_dot based on the Hamitonian law

θ0 = angles0[1,:]
θ_dot0 = angles0[2,:]
plot(θ0,θ_dot0)
#θ_dot0 = [(θ0[i]-θ0[i-1])/Δt for i=2:length(θ0)]
#pushfirst!(θ_dot0, θ_dot0[1])

#Hamiltonian(θ_dot) = θ_dot0 .^2 ./ 2 .+ g/L .* (1 .- cos.(θ))
Hamiltonian(θ,θ_dot) = θ_dot .^2 ./ 2 .+ g/L .* (1 .- cos.(θ))
H0 = Hamiltonian(θ0,θ_dot0)
plot(H0)
ker = ImageFiltering.Kernel.gaussian((11,))
H0_smooth = imfilter(H0, ker)
plot!(H0_smooth)

function loss(x,y)
    ŷ = Hamiltonian(x)
    Flux.Losses.mse(y,ŷ)
end

p = params(θ0,θ_dot0)
opt = ADAM(0.01)
for epoch in 1:100
    gs = gradient(() -> loss([θ0,θ_dot0],H0_smooth), p)
    Flux.Optimise.update!(opt,p, gs)
    println(loss((θ0,θ_dot0),H0_smooth))
end

H1 = Hamiltonian(θ0,θ_dot0)
plot(H1)

ker = ImageFiltering.Kernel.gaussian((1,))
θ0_smooth = imfilter(θ0, ker)
θ_dot0_smooth = imfilter(θ_dot0, ker)
plot(θ0, θ_dot0)
plot(θ0_smooth, θ_dot0_smooth)
plot(θ0)
plot!(θ0_smooth)
plot(θ_dot0)
plot!(θ_dot0_smooth)

angles0 = (hcat(θ0_smooth,θ_dot0_smooth))'
npzwrite("Data/new_adjusted.npy", angles0)

plot(angles0_raw)
plot!(θ0)

plot(angles0_dot_raw)
plot!(θ_dot0)
plot!(θ_dot0_smooth)

angles = angles0[:,start:start+300]
#Ŵ = gs[θ_dot0]
#display(Ŵ)
##################################
# use ODE solver with friction
#################################

function pendulum_ode(u, p, t)
    θ = u[1]
    θ_dot = u[2]
    mu = p[1]
    L = p[2]
    #poly_friction(θ_dot)
    #(0.06*(θ_dot^2) -0.03*θ_dot)
    [θ_dot, - (g/L)*sin(θ)]
end

#U0 = convert(Array{Float32}, [pi/3,0])
#U0 = Float32[pi/2, 0.0]
U0 = u0
tspan = (0.0,60.0)
#Δt = 0.01
p = Float32[0.1,L]

prob = ODEProblem(pendulum_ode, U0, tspan, p)
sol = solve(prob, Feagin14(), saveat=Δt, reltol=1.0e-20)
data = Array(sol)

plot3d(sol.t,data[1,:],data[2,:])
plot(data[1,:])
plot!(angles0[1,:])
plot!(angles')
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

θ = data[1,:]
θ̇ = data[2,:]
# Hamiltonian energy should be conserved
H = θ̇.^2 ./ 2 .+ g/L .* (1 .- cos.(θ))
H .= abs.(H/H[1] .- 1)
plot(H)
savefig(output_figures*"simple_pendulum_hamiltonian.png")

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
#
ann = FastChain(FastDense(2,8,tanh), FastDense(8, 1))
snet = SymmetryNet(ann,sym=1)
#ann = FastChain(FastDense(4,64,σ), FastDense(64,64,σ), FastDense(64,1))
#ann = FastChain(FastDense(4, 64, tanh),FastDense(64, 64, tanh), FastDense(64, 1))
#ann = FastChain(FastDense(4,64,tanh), FastDense(64, 64, tanh),FastDense(64,1))
if isfile(output_models*"pendulum_n.h5")
    p_ann = h5read(output_models*"pendulum_nn.h5", "weights")
else
    p_ann = initial_params(ann)
end
#p_ann = snet.p

#U0 = Float32[pi/2, 0.0]
U0 = angles[:,1]
last = 400
data = angles #0[:,29:last]
function nn_ode(u, p, t)
    θ = u[1]
    θ_dot = u[2]
    torque = ann([θ,θ_dot], p) #*θ_dot θ_dot^2, θ_dot,
    #torque = ann([θ_dot, θ_dot^2, θ, cos(θ)], p) #*θ_dot
    [θ_dot, torque[1] - (g/L)*sin(θ)]
end

#p = [2.0, p_ann]
#p = p_ann
#Δt = 1/60.0
tspan = (0.0, length(angles[1,:])/60.0)
prob_nn = ODEProblem(nn_ode, U0, tspan, p_ann)
s = solve(prob_nn, Vern7(), saveat = Δt, abstol=1e-9, reltol=1e-12)

#plot(Array(s)')
#plot!(data')
plot(Array(s)',linewidth=2,xaxis="t",label=["θ [rad]" "ω [rad/s]"],layout=(2,1))
plot!(angles',linewidth=2,xaxis="t",label=["θ [rad]" "ω [rad/s]"],layout=(2,1))
savefig(output_figures*"initial_nn.png")

#Vern7()
function predict(θ)
    Array(solve(prob_nn, Vern7(), u0=U0, p=θ, saveat = Δt,
                         abstol=1e-9, reltol=1e-12))
                         #sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
end

# No regularisation right now
function loss(θ)
    pred = predict(θ) 
    sum(abs2, data[1,:] .- pred[1,1:length(data[1,:])]), pred # + 1e-5*sum(sum.(abs, params(ann)))
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
batch = 1024
angles0
for i=1:1
    start = 64 #rand(64:3500)
    println("iteration: ", i, " starting: ", start)
    data = angles0[:,start:end]
    U0 = data[:,1]
    tspan = (0.0, length(data[1,:])/60.0)
    prob_nn = ODEProblem(nn_ode, U0, tspan, p_ann)
    res1 = DiffEqFlux.sciml_train(loss, p_ann, ADAM(0.01), cb=callback, maxiters = 100)
    p_ann = res1.minimizer
    res2 = DiffEqFlux.sciml_train(loss, res1.minimizer, BFGS(initial_stepnorm=0.01), cb=callback, maxiters = 100)
    p_ann = res2.minimizer
end

# Plot the losses
plot(losses, yaxis = :log, xaxis = :log, xlabel = "Iterations", ylabel = "Loss", legend = false)
savefig(output_figures*"loss.png")
# Plot the data and the approximation
NNsolution = predict(p_ann)
# Trained on noisy data vs real solution
#plot(NNsolution')
#plot!(data')
plot(NNsolution',linewidth=2,xaxis="t",label=["θ [rad]" "ω [rad/s]"],layout=(2,1))
savefig(output_figures*"plot_nn_10s_01.png")
plot(NNsolution[1,:],NNsolution[2,:],marker=:hex)
savefig(output_figures*"scatter_nn_10s_01.png")
weights = p_ann
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

#ann = FastChain(FastDense(4, 64, tanh),FastDense(64, 64, tanh), FastDense(64, 1))
#ann = FastChain(FastDense(4,16,tanh), FastDense(16, 1))
#ann = FastChain(FastDense(2,16,σ), FastDense(16, 1))
#ann = FastChain(FastDense(2,4,σ), FastDense(4,8,σ), FastDense(8,4,σ), FastDense(4, 1))
#ann = FastChain(FastDense(3,1))
#ann = FastChain(FastDense(3,64,σ), FastDense(64,1))#weights = initial_params(ann)
#ann = FastChain(FastDense(2,64,σ), FastDense(64,64,σ), FastDense(64,1))
#@load output_models*"pendulum_nn.jld2" ann weights
weights = h5read(output_models*"pendulum_nn.h5", "weights")

# extend to 30 seconds with Δt=0.01

#U0 = Float32[pi/2,0.0]
start = 64
U0 = angles0[:,start]
tspan = (0.0,60.0)
#Δt = 0.01
#p = Float32[0.1,2.0]

prob_nn = ODEProblem(nn_ode, U0, tspan, weights)
#s = solve(prob_nn, Tsit5(), saveat = Δt)
s = solve(prob_nn, Vern7(), saveat = Δt, abstol=1e-12, reltol=1e-15)
res = Array(s)

plot(res',linewidth=4,xaxis="t",label=["pred θ" "pred ω"],layout=(2,1),legend=false)
plot!(angles0[:,start:end]',linewidth=2,xaxis="t",label=["θ" "ω"],layout=(2,1),legend=false)
savefig(output_figures*"plot_nn_30s_001.png")
plot(res[1,:],res[2,:], marker=:hex)
plot(angles0[1,:],angles0[2,:])
savefig(output_figures*"scatter_nn_30s_001.png")

prefix = hcat(angles0_raw[1:4],zeros(4))'
allResult = hcat(prefix,hcat(angles0[:,1:start-1],res))
plot(allResult',linewidth=2,xaxis="t",label=["θ [rad]" "ω [rad/s]"],layout=(2,1))
npzwrite("Data/new_predicted.npy", allResult)

#plot(angles0_raw[:,start:end]',linewidth=2,xaxis="t",label=["pred θ" "pred ω"],layout=(2,1),legend=false)
plot(angles0[:,start:end]',linewidth=2,xaxis="t",label=["θ" "ω"],layout=(2,1),legend=false)


θ = allResult[1,:]
θ̇ = allResult[2,:]
θ0 = angles0[1,:]
θ̇0 = angles0[2,:]
H = θ̇.^2 ./ 2 .+ g/L .* (1 .- cos.(θ))
H0 = θ̇0 .^2 ./ 2 .+ g/L .* (1 .- cos.(θ0))
plot(H)
plot!(H0)

#prob = ODEProblem(pendulum!,U0,tspan,M)
#sol = solve(prob, Tsit5(), saveat=Δt)
#data = Array(sol)

# results should match data
plot(res',linewidth=2,xaxis="t",label=["predicted θ [rad]" "predicted ω [rad/s]"],layout=(2,1))
plot!(data',linewidth=2,xaxis="t",label=["True θ [rad]" "True ω [rad/s]"],layout=(2,1))
savefig(output_figures*"comp_data_nn_30s_001.png")
# display what the neural network ann learned

θ_dot_range = -2.5:0.1:2.5
θ_range = -0.8:0.1:0.8

#torque_ann(x,y) =  ann([y, m, L, x], weights)[1]
#plot(θ_dot_range,torque_ann)

torque_ann(x,y) =  snet([x,y], weights)[1]
#torque_ann(x) =  snet([x], weights)[1]
#plot(θ_dot_range,torque_ann)
#torque_ann(x,y) =  ann([y, y^2, x, cos(x)], weights)[1]
torque(x,y) = poly_friction(y) + 3/(m*L^2)*2.0*cos(x)
torque_diff(x,y) = torque(x,y) - torque_ann(x,y)
F = zeros(length(θ_range),length(θ_dot_range))
i1 = 1

for i=θ_range
    j1 = 1
    for j=θ_dot_range
        F[i1,j1] = torque_ann(i,j)
        j1 = j1+1
    end
    i1 = i1+1
end
#surface(θ_range, θ_dot_range, torque)
surface(θ_range, θ_dot_range, torque_ann, xlabel="θ", ylabel="θ_dot")
savefig(output_figures*"fiction_nn.png")
surface(θ_range, θ_dot_range, torque_diff)
savefig(output_figures*"diff_torque_nn.png")
plot(θ_dot_range,F',legend=false)
savefig(output_figures*"fiction_nn_2d.png")
plot(θ_range, θ_dot_range, F', st=:surface)
#Makie.surface(θ_range, θ_dot_range, F)
plot(θ_dot_range, F[8,:])
plot(θ_range, F[:,8])

DX = Array(sol(sol.t, Val{1}))
plot(DX')
plot!(data')

# Create a Basis
@variables u[1:2]
# Lots of polynomials
polys = Operation[1]
for i ∈ 1:2
    push!(polys, u[1]^i)
    push!(polys, u[2]^i)
    for j ∈ i:2
        if i != j
            push!(polys, (u[1]^i)*(u[2]^j))
            push!(polys, u[2]^i*u[1]^i)
        end
    end
end

X = zeros(length(θ_range), length(θ_dot_range))
X = [[x, y] for x in θ_range, y in θ_dot_range][:]
X[:,:]
Y = [torque_ann(x[1], x[2]) for x in X]
X = hcat(X)
Y[:]
# And some other stuff
h = [cos.(u)...; sin.(u)...; polys...]
#h = [cos.(u)...; sin.(u)...; exp.(u)...; polys...]
basis = Basis(h, u)

# Create an optimizer for the SINDY problem
opt = SR3()
# Create the thresholds which should be used in the search process
λ =  exp10.(-6:0.1:2)
# Target function to choose the results from; x = L0 of coefficients and L2-Error of the model
#f_target(x, w) = iszero(x[1]) ? Inf : norm(w.*x, 2)
opt = STRRidge(0.01)

# Test on original data and without further knowledge
Ψ = SINDy(X, Y, basis, opt, maxiter = 1000)
println(Ψ)
print_equations(Ψ)
get_error(Ψ)
print(parameters(Ψ))

#F1(θ,θ_dot) = -0.55*cos(θ) - 0.685*cos(θ_dot) -0.1*θ_dot^2 - 0.028*θ_dot + 1.0
#F1(θ,θ_dot) = -L*cos(θ) - L*cos(θ_dot) -m*θ_dot^2 - m/2.3*θ_dot + 1.0
#F1(θ,θ_dot) = 5.335*cos(θ) + 0.59*cos(θ_dot) + 2.9*θ^2 + 0.15 *θ_dot^2 - 0.025*θ_dot - 5.8
#F1(θ,θ_dot) = 0.75 + 0.116 * cos(θ) -0.03 * sin(θ) - 0.67*cos(θ_dot) + 0.03 * sin(θ_dot) - 0.02 * θ^2 - 0.16 * θ_dot^2 + 0.07 * θ -0.04*θ_dot -0.007*θ_dot^2*θ + 0.02*θ*θ_dot
#F1(θ,θ_dot) = 0.38 + 0.0158 * θ - 0.03 * θ_dot
#F1(θ,θ_dot) = 0.54 - 0.178 * cos(θ_dot) - 0.05 * θ_dot^2 - 0.026*θ_dot
#F1(θ,θ_dot) = 0.39 - 0.01 * cos(θ_dot) + 0.09 * sin(θ) - 0.07 * θ_dot - 0.05 *θ_dot^2
#F1(θ,θ_dot) = θ_dot>0.5 ? 0.35 : θ_dot>-0.5 ? sin(θ_dot)-0.1 : 0.479
F1(θ,θ_dot) = 0.52 + 0.12*cos(θ) - 0.017 * cos(θ_dot) + 0.003*θ_dot^2 + 0.02 * θ^2 + 0.035 * θ *θ_dot
Plots.surface(θ_range, θ_dot_range, F1, xlabel="θ", ylabel="θ_dot")
#plot(θ_dot_range,F1)
savefig(output_figures*"SINDy-friction.png")
plot(θ_dot_range,F')
#Makie.surface(θ_range, θ_dot_range, F1)

function pendulum_f1!(du,u,p,t)
    du[1] = u[2]                    # θ'(t) = ω(t)
    du[2] = F1(u[1],u[2]) -(g/L)*sin(u[1]) # ω'(t) = -3g/(2l) sin θ(t) + 3/(ml^2)M(t)
end

start = 64
U0 = angles0[:,start]
tspan = (0.0,60.0)

prob_nn = ODEProblem(pendulum_f1!, U0, tspan)
#s = solve(prob_nn, Tsit5(), saveat = Δt)
s = solve(prob_nn, Vern7(), saveat = Δt, abstol=1e-12, reltol=1e-15)
res = Array(s)
print(sum(abs2, res[1,:] .- angles0[1,start:length(res[1,:])+start-1]))
plot(res',linewidth=4,xaxis="t",layout=(2,1), legend=false) #,label=["θ [rad]" "ω [rad/s]"],layout=(2,1))
plot!(angles0[:,start:start+length(res[1,:])]',linewidth=2,xaxis="t",layout=(2,1), legend=false)#,label=["θ" "ω"],layout=(2,1))
savefig(output_figures*"SINDy-Pendulum.png")

#allResult = hcat(angles0[:,1:start-1],res)
allResult = hcat(prefix,hcat(angles0[:,1:start-1],res))
plot(allResult',linewidth=2,xaxis="t",label=["θ [rad]" "ω [rad/s]"],layout=(2,1))
plot!(angles0',linewidth=2,xaxis="t",label=["θ" "ω"],layout=(2,1))

npzwrite("Data/new_predicted.npy", allResult)


nn = Chain(Dense(2,2,tanh), Dense(2, 1))
if isfile(output_models*"pendulum_nn.h5")
    weights = h5read(output_models*"pendulum_nn.h5", "weights")
    ps = Flux.params(nn)
    θ, re = Flux.destructure(nn);
    θ = weights
    nn=re(θ)
end
ps = params(nn)

function loss(x)
    y1 = nn(x)
    y2 = nn([x[1],-x[2]])
    l = sum(abs2,(y1-y2))
    println(l)
    return l
end

X = zeros(length(θ_range), length(θ_dot_range))
X =[[x, y] for x in θ_range, y in θ_dot_range][:]
y1 = nn(X[1])
opt=ADAM(0.0001)
for i in 100
    Flux.train!(loss,ps,X,opt)
end
nn(X[1])
F = zeros(length(θ_range),length(θ_dot_range))
i1 = 1
for i=θ_range
    j1 = 1
    for j=θ_dot_range
        F[i1,j1] = nn([i,j])[1]
        j1 = j1+1
    end
    i1 = i1+1
end
Plots.surface(θ_range, θ_dot_range, F, xlabel="θ", ylabel="θ_dot")
plot(θ_dot_range,F')
@save output_models*"friction_nn.bson" nn
θ, re = Flux.destructure(nn);
fid=h5open(output_models*"pendulum_nn_sym.h5","w")
fid["weights"] = θ
close(fid)

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
start = 64
U0 = angles0[:,start]
tspan = (0.0,10.0)
Δt = 1/60.0

prob = ODEProblem(pendulum!,u0,tspan,M)
sol = solve(prob, Tsit5(), saveat=Δt)
data = Array(sol)


# Use video data
angles = angles0[:,start:start+length(tspan[1]:Δt:tspan[2])-1]
#tspan = (0.0, length(angles[1,:])/60.0)
#dudt = Chain(x->[x[1],x[2],L, m],Dense(4,32,tanh),Dense(32,2)) #, y->[y[1],y[2],g/L])
#x->[x[1],x[2],-g/L],
#dudt = Chain(x->[x[1],x[2],-(g/L)*sin(x[1])],Dense(3,2,tanh),Dense(2,2))
dudt = Chain(x->[x[1],x[2],sin(x[1])],Dense(3,8,tanh),Dense(8,2))
function dudt1(x)
     [x[1], Chain(Dense(2,2,tanh),Dense(2,1),x->x[1])]
end
W=rand(2,2)
b=rand(2)
x= rand(2)
simpleNN(x) = W*x+b
u0 = angles[:,1]
simpleNN(u0)

#n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=Δt,reltol=1e-6,abstol=1e-7)
n_ode = NeuralODE(dudt, tspan, Vern7(), saveat = Δt, abstol=1e-6, reltol=1e-7)
if isfile(output_models*"pendulum_n_ode_weights.bson")
    @load output_models*"pendulum_n_ode_weights.bson" ps
    Flux.loadparams!(n_ode,ps)
end
ps = Flux.params(n_ode)


n_ode(u0)
function predict_n_ode()
  n_ode(u0)
end
d = predict_n_ode()
length(angles[1,:])
Array(d)
loss_n_ode() = sum(abs2, angles[1,:] .- Array(predict_n_ode(W,b))[1,1:length(angles[1,:])])

#ps = params(W,b)
#opt = ADAM(0.01)

#for epoch in 1:100
#    gs = gradient(() -> loss_n_ode(W,b), ps)
#    Flux.Optimise.update!(opt,ps, gs)
#    println(loss_n_ode())
#end

t = tspan[1]:Δt:tspan[2]

it = Iterators.repeated((), 1000)
opt = ADAM(0.01)
losses = []
cb = function () #callback function to observe training
  l = loss_n_ode()
  append!(losses,l)
  display(l)
  if length(losses)%10==0
      # plot current prediction against data
      cur_pred = predict_n_ode()
      #pl = scatter(angles[1,:],angles[2,:],label="data")
      #scatter!(pl,cur_pred[1,:], cur_pred[2,:],label="prediction")
      pl = plot(angles',label="data")
      plot!(pl,cur_pred',label="prediction")
      display(plot(pl))
  end
end

# Display the ODE with the initial parameter values.
cb()

Flux.train!(loss_n_ode, ps, it, opt, cb = cb)

tspan = (0.0, length(angles0[1,:])/60.0)
res = n_ode(u0)
d = hcat(res.u)
scatter(d[1,:], d[2,:])
plot(d[1,:])
plot!(angles0[1,start:end])

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
plot!(angles0[:,64:600]')


prob = ODEProblem(pendulum!,u0,tspan,M)
sol = solve(prob, Tsit5(), saveat=Δt)
data = Array(sol)
plot!(sol.t,data[1,:])




###############
##  Try a customized method using Newton's law
############


u0 = angles[:,1]
function pendulum_solver(θ, θ_dot, t)
    Δt = 0.001 #1/60.0
    results = [θ,θ_dot]
    for t in 0:Δt:t
        #append!(results, [θ;θ_dot])
        θ_double_dot = get_double_theta_dot(θ, θ_dot)
        θ += θ_dot * Δt
        θ_dot += θ_double_dot * Δt
        results = [results,[θ,θ_dot]]
    end
    results
end
results = Array([u0[1], u0[2]])
[results,[1,2]]
results = pendulum_solver(u0[1], u0[2], 20.0)
results
plot(results)
