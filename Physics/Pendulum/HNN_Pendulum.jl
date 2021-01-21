cd(@__DIR__)
using Pkg;
Pkg.activate("..");
Pkg.instantiate();
using DiffEqFlux, Flux, OrdinaryDiffEq, Statistics, Plots, ReverseDiff

t = range(0.0f0, 1.0f0, length = 1024)
π_32 = Float32(π)
q_t = reshape(sin.(2π_32 * t), 1, :)
p_t = reshape(cos.(2π_32 * t), 1, :)
dqdt = 2π_32 .* p_t
dpdt = -2π_32 .* q_t

data = cat(q_t, p_t, dims = 1)
target = cat(dqdt, dpdt, dims = 1)
dataloader = Flux.Data.DataLoader(data, target; batchsize=256, shuffle=true)

hnn = HamiltonianNN(
    Chain(Dense(2, 64, relu), Dense(64, 1))
)


p = hnn.p
p
opt = ADAM(0.01)

loss(x, y, p) = mean((hnn(x, p) .- y) .^ 2)

callback() = println("Loss Neural Hamiltonian DE = $(loss(data, target, p))")

epochs = 1000
for epoch in 1:epochs
    for (x, y) in dataloader
        gs = ReverseDiff.gradient(p -> loss(x, y, p), p)
        Flux.Optimise.update!(opt, p, gs)
    end
    if epoch % 100 == 1
        callback()
    end
end
callback()

hnn.re(hnn.p)([0.0, 1.0])
hnn([0.0, 0.0])

model = NeuralHamiltonianDE(
    hnn, (0.0f0, 1.0f0),
    Tsit5(), save_everystep = false,
    save_start = true, saveat = t
)

pred = Array(model(data[:, 1]))
plot(data[1, :], data[2, :], lw=4, label="Original")
plot!(pred[1, :], pred[2, :], lw=4, label="Predicted")
Plots.xlabel!("Position (q)")
Plots.ylabel!("Momentum (p)")

## HNN for Pendulum

m=1.0
L=1.0
g = 9.8
Ham_Pendulum(θ, p) = p^2/(2*m*L^2) - m*g*L*cos(θ)

Ham_Pendulum(1.0, 0.0)

# θ̇ = p/mL^2 , ṗ = -mgLsin(θ)
analytics_Ham_Pendulum(θ,p) = [p/(m*L^2), -m*g*L*sin(θ)]

analytics_Ham_Pendulum(1.0, 0.0)

function ODE_Ham_analytics_fn(u, p, t)
    analytics_Ham_Pendulum(u[1],u[2])
end

function ODE_Ham_fn(u, p, t)
    dHdq, dHdp  = ReverseDiff.gradient(u -> Ham_Pendulum(u[1],u[2]), u)
    # dHdp = dqdt ; -dHdq = dpdt
    [dHdp, -dHdq]
end

ODE_Ham_fn([1.0,0.0], 0.0, 1.0)

u0 = Float32[1.0, 0.0]
tspan = Float32[0.0,10.0]
Δt = 0.01
prob = ODEProblem(ODE_Ham_analytics_fn, u0, tspan)
sol = solve(prob, Tsit5(), saveat=Δt, reltol=1.0e-9)
data = Array(sol)
#plot(sol)
plot(data[1,:], data[2,:], axis="q", yaxis="p", label="phase", marker = :hex)

prob = ODEProblem(ODE_Ham_fn, u0, tspan)
sol = solve(prob, Tsit5(), saveat=Δt, reltol=1.0e-9)
data = Array(sol)
#plot(sol)
plot(data[1,:], data[2,:], axis="q", yaxis="p", label="phase", marker = :hex)

NN = FastChain(FastDense(3,32,tanh),FastDense(32,1))
pnn = initial_params(NN)

NN([1.0,0.0,0.0],pnn)

Flux.gradient(x -> sum(NN(x, p)), [1.0,0.0,1.0])
