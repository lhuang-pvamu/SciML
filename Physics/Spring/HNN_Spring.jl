cd(@__DIR__)
using Pkg;
Pkg.activate("..");
Pkg.instantiate();
using DiffEqFlux, Flux, OrdinaryDiffEq, Statistics, Plots, ReverseDiff
gr()
output_figures="Figures/"
output_models="Models/"

## One spring system with gravity
##

t = range(0.0f0, 30.0f0, length = 1024)
m = 2.0
k = 2.0
L = 1.0
g = 9.8

function Hamiltonian_fn(u, parm)
    q, p = u
    m, k, L = parm
    1/(2*m) * p^2 + k/2 * (q-L)^2 + m*g*q
end

u0 = Float32[0.5, 0.0]
tspan = Float32[0.0,10.0]
Δt = 0.01
parm = Float32[m,k,L]
Hamiltonian_fn(u0, parm)

function ODE_Ham_fn(u, p, t)
    dHdq, dHdp  = ReverseDiff.gradient((u,p) -> Hamiltonian_fn(u,p), (u,p))[1]
    # dHdp = dqdt ; -dHdq = dpdt
    [dHdp, -dHdq]
end

ODE_Ham_fn(u0, parm, 0)

prob = ODEProblem(ODE_Ham_fn, u0, tspan, parm)
sol = solve(prob, Tsit5(), saveat=Δt, reltol=1.0e-9)
data = Array(sol)
#plot(sol)
plot(data[1,:], data[2,:], axis="q", yaxis="p", label="phase", marker = :hex)

target = [ODE_Ham_fn(data[:,i],parm,0) for i in 1:length(data[1,:])]
target = hcat(target...)

dataloader = Flux.Data.DataLoader(data, target; batchsize=256, shuffle=true)

hnn = HamiltonianNN(
    Chain(Dense(2, 32, tanh), Dense(32, 1))
)


p = hnn.p
p
opt = ADAM(0.01)
hnn(u0,p)

loss(x, y, p) = mean((hnn(x, p) .- y) .^ 2)

callback() = println("Loss Neural Hamiltonian DE = $(loss(data, target, p))")

epochs = 1000
for epoch in 1:epochs
    for (x, y) in dataloader
        gs = ReverseDiff.gradient(p -> loss(x, y, p), p)
        #print(gs)
        Flux.Optimise.update!(opt, p, gs)
    end
    if epoch % 100 == 1
        callback()
    end
end

model = NeuralHamiltonianDE(
    hnn, (0.0f0, 30.0f0),
    Tsit5(), save_everystep = false,
    save_start = true, saveat = t
)

pred = Array(model(data[:, 1]))
plot(data[1, :], data[2, :], lw=4, label="Original")
plot!(pred[1, :], pred[2, :], lw=4, label="Predicted")
Plots.xlabel!("Position (q)")
Plots.ylabel!("Momentum (p)")

anim = @animate for i ∈ 1:10:length(pred[1,:])
    plot([(0,0),(0, pred[1,i])], marker = :hex, xlims=(-2,2), ylims=(-20,5),
    title="$(round((i-1)*Δt,digits=1)) second", legend = false)
end
gif(anim, "anim_friction.gif", fps = 1024/(10*30))


## 3 Springs + 2 mass System
# |~~~~~~O~~~~~O~~~~~~~|
##
m1 = 2.0
m2 = 1.0
k1 = 2.0
k2 = 1.0
k3 = 3.0
L1 = 1.0
L2 = 1.0
L3 = 1.0
L = 3.0

function Hamiltonian_3Springs_fn(u, parm)
    q1, q2, p1, p2 = u
    m1, k1, m2, k2, k3, L = parm
    1/m1 * p1^2 +1/m2 * p2^2 + k1/2 * (q1)^2 + k2/2 * (q2-q1)^2  + k3/2 * (L-q2)^2
end

u0 = Float32[1.5, 2.5, 0.0, 0.0]
tspan = Float32[0.0,30.0]
Δt = 0.01
parm = Float32[m1,k1,m2,k2,k3,L]
Hamiltonian_3Springs_fn(u0, parm)

function ODE_Ham_3S_fn(u, p, t)
    dHdq1, dHdq2, dHdp1, dHdp2  = ReverseDiff.gradient((u,p) -> Hamiltonian_3Springs_fn(u,p), (u,p))[1]
    # dHdp = dqdt ; -dHdq = dpdt
    [dHdp1, dHdp2, -dHdq1, -dHdq2]
end
ReverseDiff.gradient((u,p) -> Hamiltonian_3Springs_fn(u,p), (u0,parm))

ODE_Ham_3S_fn(u0, parm, 0)

prob = ODEProblem(ODE_Ham_3S_fn, u0, tspan, parm)
sol = solve(prob, Tsit5(), saveat=Δt, reltol=1.0e-9)
data = Array(sol)
#plot(sol)
plot(data[1,:], data[3,:], axis="q", yaxis="p", label="phase")
plot!(data[2,:], data[4,:], axis="q", yaxis="p", label="phase")
savefig(output_figures*"3Springs_plot.png")

anim = @animate for i ∈ 1:10:length(data[1,:])
    plot([(0,0),(data[1,i],0)], marker = :hex, xlims=(0,3), ylims=(-1,1),legend = false)
    plot!([(data[1,i],0),(data[2,i],0)], marker = :hex, xlims=(0,3), ylims=(-1,1),legend = false)
    plot!([(data[2,i],0),(3,0)], marker = :hex, xlims=(0,3), ylims=(-1,1),
    title="$(round((i-1)*Δt,digits=1)) second", legend = false)
end
gif(anim, "anim_friction.gif", fps = 10)


target = [ODE_Ham_3S_fn(data[:,i],parm,0) for i in 1:length(data[1,:])]
target = hcat(target...)

dataloader = Flux.Data.DataLoader(data, target; batchsize=256, shuffle=true)

hnn_3S = HamiltonianNN(
    Chain(Dense(4, 32, tanh), Dense(32, 1))
)


p = hnn_3S.p
opt = ADAM(0.01)
hnn_3S(u0,p)

loss(x, y, p) = mean((hnn_3S(x, p) .- y) .^ 2)

callback() = println("Loss Neural Hamiltonian DE = $(loss(data, target, p))")

epochs = 1000
for epoch in 1:epochs
    for (x, y) in dataloader
        gs = ReverseDiff.gradient(p -> loss(x, y, p), p)
        #print(gs)
        Flux.Optimise.update!(opt, p, gs)
    end
    if epoch % 100 == 1
        callback()
    end
end

model = NeuralHamiltonianDE(
    hnn_3S, (0.0f0, 30.0f0),
    Tsit5(), save_everystep = false,
    save_start = true, saveat = t
)

pred = Array(model(data[:, 1]))
plot(data[1, :], data[3, :], lw=2, label="Original 1")
plot!(data[2,:], data[4,:], lw=2, label="Original 2")
plot!(pred[1, :], pred[3, :], lw=2, label="Predicted 1")
plot!(pred[2, :], pred[4, :], lw=2, label="Predicted 2")
Plots.xlabel!("Position (q)")
Plots.ylabel!("Momentum (p)")


##  4 spings 1 mass
#     |
#   ~~o~~
#     |
##

m = 2.0
k1 = 1.0
k2 = 1.0
k3 = 1.0
k4 = 1.0
L1 = 1.0
L2 = 1.0
L3 = 1.0
L4 = 1.0

function Hamiltonian_4Springs_fn(u, parm, t)
    qx, qy, px, py = u
    #m, k1, k2, k3, k4, L1, L2, L3, L4 = parm
    q1 = sqrt((L1+qx)^2 + qy^2)
    q2 = sqrt((L2-qx)^2 + qy^2)
    q3 = sqrt((L3-qy)^2 + qx^2)
    q4 = sqrt((L4+qy)^2 + qx^2)
    (1/(2*m) * (px^2 + py^2) + k1/2 * (q1-L1)^2 + k2/2 * (q2-L2)^2
    + k3/2 * (q3 - L3)^2 + k4/2 * (q4 - L4)^2 - 0.003*t[1]*(px^2 + py^2))
end

u0 = Float32[0.5, 0.5, 0.0, 0.0]
tspan = Float32[0.0,60.0]
Δt = 0.01
parm = Float32[m1,k1,m2,k2,k3,L]
Hamiltonian_4Springs_fn(u0, parm, [1])

function ODE_Ham_4S_fn(u, p, t)
    dHdqx, dHdqy, dHdpx, dHdpy  = ReverseDiff.gradient((u,p,t) -> Hamiltonian_4Springs_fn(u,p,t), (u,p,[t]))[1]
    # dHdp = dqdt ; -dHdq = dpdt
    [dHdpx, dHdpy, -dHdqx, -dHdqy]
end
ReverseDiff.gradient((u,p) -> Hamiltonian_4Springs_fn(u,p), (u0,parm))

ODE_Ham_4S_fn(u0, parm, 0)

prob = ODEProblem(ODE_Ham_4S_fn, u0, tspan, parm)
sol = solve(prob, Tsit5(), saveat=Δt, reltol=1.0e-9)
data = Array(sol)
#plot(sol)
plot(data[1,:], data[3,:], axis="q", yaxis="p", label="phase")
plot!(data[2,:], data[4,:], axis="q", yaxis="p", label="phase")
savefig(output_figures*"4Springs_plot.png")


anim = @animate for i ∈ 1:10:length(data[1,:])
    plot([(1,0),(data[1,i],data[2,i])], marker = :hex, xlims=(-1,1), ylims=(-1,1),legend = false)
    plot!([(0,1),(data[1,i],data[2,i])], marker = :hex, xlims=(-1,1), ylims=(-1,1),legend = false)
    plot!([(0,-1),(data[1,i],data[2,i])], marker = :hex, xlims=(-1,1), ylims=(-1,1),legend = false)
    plot!([(-1,0),(data[1,i],data[2,i])], marker = :hex, xlims=(-1,1), ylims=(-1,1),
    title="$(round((i-1)*Δt,digits=1)) second", legend = false)
end
gif(anim, "anim_4springs.gif", fps = 10)
