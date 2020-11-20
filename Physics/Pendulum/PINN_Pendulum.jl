cd(@__DIR__)
using Pkg;
Pkg.activate("..");
Pkg.instantiate();
using LinearAlgebra
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using OrdinaryDiffEq

output_figures="Figures/"
output_models="Models/"

#################
# Pendulum motion prediction using PINN
#################

@parameters t θ
@variables u(..)
@derivatives Dtt''~t
@derivatives Dt'~t

L=1.0
# ODE
eq = Dtt(u(t,θ)) ~ -9.8/L * sin(u(t,θ))

# Initial and boundary conditions
bcs = [u(0.,θ) ~ pi/3,
       Dt(u(0.,θ)) ~ 0.0]

# Space and time domains
domains = [t ∈ IntervalDomain(0.0,10.0)]

# Discretization
dt = 0.1

# Neural network
chain = FastChain(FastDense(1,32,Flux.σ),FastDense(32,1))
ps = initial_params(chain)

discretization = PhysicsInformedNN(dt,
                                   chain,
                                   init_params = ps,
                                   strategy= GridTraining())
                                   #strategy=StochasticTraining(include_frac=0.5))
pde_system = PDESystem(eq,bcs,domains,[t],[u])
prob = discretize(pde_system,discretization)

opt = Optim.BFGS()
#opt = Optim.ADAM(0.01)
res = GalacticOptim.solve(prob,opt; cb = cb, maxiters=2000)
phi = discretization.phi
ps = res.minimizer

ts = [domain.domain.lower:dt/10:domain.domain.upper for domain in domains][1]
u_predict  = [first(phi(t,res.minimizer)) for t in ts]
t_plot = collect(ts)
plot(t_plot,u_predict,label="PINN", title = "PINN Predict")

g = 9.8
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
U0 = Float32[pi/3, 0.0]
tspan = (0.0,10.0)
Δt = 0.01
p = Float32[0.1,L]

prob = ODEProblem(pendulum_ode, U0, tspan, p)
sol = solve(prob, Vern7(), saveat=Δt)
data = Array(sol)

plot!(sol.t,data[1,:],label = "ODE")
savefig(output_figures*"PINN_ODE_Pendulum.png")
plot(data[1,:],data[2,:])

###############
##  Only use the first order ODE for PINN
#  It is harder to train since there is no the first order Dt(t)=0.0 and no second order gradients calculated
###############

# Run a solve on scalars
linear = (u, p, t) -> cos(t)

tspan = (0.0f0, 3.0f0)
u0 = [pi/3.0, 0.0f0]
#prob = ODEProblem(linear, u0, tspan)
prob = ODEProblem(pendulum_ode, u0, tspan, p)
chain = Flux.Chain(Dense(1, 64, σ), Dense(64, 64, σ), Dense(64, 2))
opt = Flux.ADAM(0.001, (0.9, 0.95))
@time sol = solve(prob, NeuralPDE.NNODE(chain, opt), dt=1 / 50f0, verbose=true,
            abstol=1e-10, maxiters=10000)
data = Array(sol)
plot(sol.t,data[1,:])
savefig(output_figures*"PINN_1st_ODE_Pendulum.png")
