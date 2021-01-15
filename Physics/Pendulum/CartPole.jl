cd(@__DIR__)
using Pkg;
Pkg.activate("..");
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
using ControlSystems
using Flux: throttle, @epochs
using ReverseDiff

output_figures="Figures/"
output_models="Models/"

############################
#   Sovlve the inverted pendulum problem
#       o
#       |
#       |
#   ---------
#   |       |
#   ---------
#     o    o
###########################

m = 1.0
M = 5.0
L = 2.0
g = -9.8
δ = 1.0
F = 1.0

function pendcart!(du, u, p, t)
    m,M,L,g,δ,F = p

    Sx = sin(u[3])
    Cx = cos(u[3])
    D = m*L*L*(M+m*(1-Cx^2))

    du[1] = u[2]
    du[2] = (1/D)*(-m^2*L^2*g*Cx*Sx + m*L^2*(m*L*u[4]^2*Sx - δ*u[2])) + m*L*L*(1/D)*F
    du[3] = u[4]
    du[4] = (1/D)*((m+M)*m*g*L*Sx - m*L*Cx*(m*L*u[4]^2*Sx - δ*u[2])) - m*L*Cx*(1/D)*F
end

function sim_Pend(func, u0, p, tspan, Δt)
    prob = ODEProblem(func, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=Δt, reltol=1.0e-7)
end

function anim_Pend(x, θ; step=10)
    W = 4
    H = 1
    L = 2
    anim = @animate for i ∈ 1:step:length(x)
        plot([(-10,0),(200,0)], xlims=(-10,200), ylims=(-5,5), legend=false)
        px = [x[i]-W/2, x[i]+W/2, x[i]+W/2, x[i]-W/2]
        py = [0+H/16, 0+H/16, H, H]
        plot!(Shape(px, py), fill=(0,:green), legend=false)
        scatter!([(x[i]-W/4,H/8), (x[i]+W/4,H/8)], marker = (8, 0.5, :red))
        px = x[i] + L*sin(θ[i])
        py = H - L*cos(θ[i])
        plot!([(x[i],H),(px,py)], marker = :hex, legend=false)
    end
    anim
end

function demo_invertPend()
    u0 = Float32[1, 0, pi, 0]
    tspan = (0.0, 30.0)
    p = Float32[m,M,L,g,δ,F]
    Δt = 0.01
    sol = sim_Pend(pendcart!, u0, p, tspan, Δt)
    data = Array(sol)
    #plot(sol.t, data[1,:])
    #plot(sol.t, data[3,:])
    anim = anim_Pend(data[1,:], data[3,:])
    gif(anim, output_figures*"anim_invertPend.gif", fps = 10)
end


demo_invertPend()


##
#  Analytics solution for the cartpole problem
##

b = 1 # pendulum up (b=1)

A = [0 1 0 0;
    0 -δ/M b*m*g/M 0;
    0 0 0 1;
    0 -b*δ/(M*L) -b*(m+M)*g/(M*L) 0]

B = [0; 1/M; 0; b*1/(M*L)]

A
λ = eigen(A)
λ.values

#ctrb(A, B)
rank(A)
Q = I(4)
R = 0.00001
# calculate the Liniear Quadratic Regulator (LQR) to control the pole
K = lqr(A,B,Q,R)

#pe = [-.3; -.4; -.5; -.6]
#pe = [-1.; -1.1; -1.2; -1.3]
pe = [-2.; -2.1; -2.2; -2.3]
#pe = [-3.; -3.1; -3.2; -3.3]
K1 = place(A, B, pe)
eigen(A-B*K1)

u0 = Float32[1, 0, pi-0.1, 0]
target = [10, 0, pi, 0]
tspan = (0.0, 10.0)
p = Float32[m,M,L,g,δ]
Δt = 0.05

u0-target
K*(u0-target)[1:4]
target_Y = []
data_X = []
count = 1
function pendcartLQR!(du, u, p, t)
    m,M,L,g,δ= p
    F = (-K * (u-target)[1:4])[1]
    append!(target_Y,F)
    append!(data_X, u-target)


    Sx = sin(u[3])
    Cx = cos(u[3])
    D = m*L*L*(M+m*(1-Cx^2))

    du[1] = u[2]
    du[2] = (1/D)*(-m^2*L^2*g*Cx*Sx + m*L^2*(m*L*u[4]^2*Sx - δ*u[2])) + m*L*L*(1/D)*F
    du[3] = u[4]
    du[4] = (1/D)*((m+M)*m*g*L*Sx - m*L*Cx*(m*L*u[4]^2*Sx - δ*u[2])) - m*L*Cx*(1/D)*F
    #du[5] = F
end


sol = sim_Pend(pendcartLQR!,u0,p,tspan,Δt)
plot(sol)

ref_data = Array(sol)
#plot(sol.t, data[1,:])
#plot(sol.t, data[3,:])
anim = anim_Pend(ref_data[1,:], ref_data[3,:]; step=2)
gif(anim, output_figures*"anim_invertPend_LQR.gif", fps = 10)

target_Y1 = hcat(target_Y...)
data_X1 = Float64.(reshape(data_X, (4,:)))
target_Y

plot(target_Y1')
length(target_Y)
plot(data_X1')
length(data_X)

NN = Chain(Dense(4,32,σ), Dense(32,32,σ),Dense(32, 1))
ps = params(NN)
dataloader = Flux.Data.DataLoader((data_X1, target_Y1), batchsize=256, shuffle=true)
#using IterTools: ncycle
loss(x, y) = Flux.Losses.mse(NN(x), y)
opt = ADAM(0.01)
evalcb() = @show(loss(data_X1, target_Y1))
@epochs 1000 Flux.train!(loss, ps, dataloader, opt, cb = evalcb)
dataloader.data
ps
p_ann1,re = Flux.destructure(NN)
p_ann1
re
for (x,y) in dataloader
    print(size(x))
    print(size(y))
end
data_X[:,1]
loss(data_X1, target_Y1)
data_X1[:,100]
target_Y1[100]
ŷ = NN(data_X1)
NN(data_X1)
plot(ŷ[1,:])
plot!(target_Y)

## Use SciML UDE to find the optimal control parameters
##

# train using 10-second data with Δt=0.1
target = [10, 0, pi, 0]
u0 = Float32[1, 0, pi-0.5, 0]
tspan = (0.0, 10.0)
Δt = 0.05

ann = FastChain(FastDense(4,32,σ), FastDense(32,32,σ), FastDense(32, 1))
#ann = FastChain(FastDense(4,32,σ),FastDense(32, 1))

if isfile(output_models*"invertPend.h5")
    p_ann = h5read(output_models*"invertPend.h5", "weights")
else
    p_ann = initial_params(ann)
end
p_ann = initial_params(ann)
p_ann = p_ann1

function pendcart_ann!(du, u, p, t)
    Sx = sin(u[3])
    Cx = cos(u[3])
    D = m*L*L*(M+m*(1-Cx^2))
    #state = vcat(u,[Sx,Cx,D])
    #du[5] = u[5] + ΔF
    #F = u[5]
    F = ann((u-target)[1:4],p)[1]

    du[1] = u[2]
    du[2] = (1/D)*(-m^2*L^2*g*Cx*Sx + m*L^2*(m*L*u[4]^2*Sx - δ*u[2])) + m*L*L*(1/D)*F
    du[3] = u[4]
    du[4] = (1/D)*((m+M)*m*g*L*Sx - m*L*Cx*(m*L*u[4]^2*Sx - δ*u[2])) - m*L*Cx*(1/D)*F
    #du[5] = ann((target-u)[1:4],p)[1]
end

prob = ODEProblem(pendcart_ann!, u0, tspan, p_ann)

function predict(θ)
    #sol = solve(prob, Tsit5(), saveat=Δt, reltol=1.0e-7)
    #Array(sol)
    Array(solve(prob, Vern7(); u0=u0, p=θ, saveat = Δt,
                         abstol=1e-5, reltol=1e-7,
                         sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))

end

# No regularisation right now
function loss(θ)
    pred = predict(θ)
    #sum(abs2, data[1] .- pred[1,:]) + sum(abs2, data[3] .- pred[3,:]), pred
    #sum(abs2, target[1] .- pred[1,:]) +
    -pred[1,end] + sum(abs2, target[3] .- pred[3,:]), pred
    #sum(abs2, ref_data[:,1:length(pred[1,:])] .- pred) * 0.01, pred #[:,Int(round(length(pred[1,:])/2)):end]), pred
end

const losses = []
callback(θ,l,pred) = begin
    push!(losses, l)
    if length(losses)%5==0
        println(losses[end])
        pl = plot(ref_data')
        plot!(pl, pred')
        display(plot(pl))
    end
    false
end

res1 = DiffEqFlux.sciml_train(loss, p_ann, ADAM(0.01), cb=callback, maxiters = 100)
p_ann = res1.minimizer
res2 = DiffEqFlux.sciml_train(loss, p_ann, BFGS(initial_stepnorm=0.01), cb=callback, maxiters = 10)
p_ann = res2.minimizer

loss(p_ann)

pred = predict(p_ann)

plot(pred[3,:])
anim = anim_Pend(pred[1,:], pred[3,:]; step=2)
gif(anim, output_figures*"anim_invertPend_ann.gif", fps = 10)

weights = p_ann
fid=h5open(output_models*"invertPend.h5","w")
fid["weights"] = weights
close(fid)


##
# reinforce learning
##

using ReinforcementLearning
run(E`JuliaRL_BasicDQN_CartPole`)

re.agent.trajectory.t1.traces.reward.buffer

data = re.agent.trajectory.t2.t1.x

plot(data[1,:])

using Trebuchet

t = TrebuchetState()
simulate(t)

target = 80 # or nothing
s = visualise(t, target)

if @isdefined(Blink)
  body!(Blink.Window(), s)
end
