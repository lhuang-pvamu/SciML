cd(@__DIR__)
using Pkg;
Pkg.activate("..");
Pkg.instantiate();
using LinearAlgebra
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using OrdinaryDiffEq
using ForwardDiff
using Adapt
using Plots
gr()

output_figures="Figures/"
output_models="Models/"
include("../../Utils/SymmetryNet.jl")
#################
# Pendulum motion prediction using PINN
#################

@parameters t θ
@variables u(..)
@derivatives Dtt''~t
@derivatives Dt'~t

L=1.0
# ODE

function friction(θ_dot)
    -0.1*θ_dot
end

#eq = Dtt(u(t,θ)) ~ friction(Dt(u(t,θ))) - 9.8/L * sin(u(t,θ))
eq = Dtt(u(t,θ)) ~ - 9.8/L * sin(u(t,θ))

# Initial and boundary conditions
bcs = [u(0.,θ) ~ pi/3,
       Dt(u(0.,θ)) ~ 0.0]

# Space and time domains
domains = [t ∈ IntervalDomain(-5.0,5.0)]

# Discretization
dt = 0.1

# Neural network
chain = FastChain(FastDense(1,32,Flux.σ),FastDense(32,1))
ps = initial_params(chain)

snet = SymmetryNet(chain, sym=0)
ps = snet.p
initial_params(snet.model)

discretization = PhysicsInformedNN(dt,
                                   chain,
                                   ps,
                                   strategy= GridTraining())
                                   #strategy=StochasticTraining(include_frac=0.5))
pde_system = PDESystem(eq,bcs,domains,[t],[u])
prob = discretize(pde_system,discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end
opt = Optim.BFGS()
#opt = ADAM(0.01)
res = GalacticOptim.solve(prob,opt; cb = cb, maxiters=2000)
phi = discretization.phi
ps = res.minimizer

ts = [domain.domain.lower:dt/10:domain.domain.upper for domain in domains][1]
u_predict  = [first(phi(t,res.minimizer)) for t in ts]
t_plot = collect(ts)
plot!(t_plot,u_predict,label="PINN", title = "PINN Predict")

g = 9.8
function pendulum_ode(u, p, t)
    θ = u[1]
    θ_dot = u[2]
    mu = p[1]
    L = p[2]
    #poly_friction(θ_dot)
    #(0.06*(θ_dot^2) -0.03*θ_dot)
    [θ_dot, friction(θ_dot) - (g/L)*sin(θ)]
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
linear = (u, p, t) -> cos(2pi*t)

tspan = (0.0f0, 3.0)
u0 = sin(tspan[1])/(2pi) #[pi/3.0, 0.0f0]
prob = ODEProblem(linear, u0, tspan)
#prob = ODEProblem(pendulum_ode, u0, tspan, p)
chain = FastChain(FastDense(1, 8, σ),  FastDense(8, 1))
initθ = initial_params(chain)
snet=SymmetryNet(chain,sym=1)
opt = Flux.ADAM(0.1, (0.9, 0.95))
#opt = Optim.BFGS()
@time sol = solve(prob, NeuralPDE.NNODE(chain, opt), dt=1 / 20f0, verbose=true,
            abstol=1e-10, maxiters=1000)
data = Array(sol)
plot(sol.t,data)
savefig(output_figures*"PINN_1st_ODE_Pendulum.png")

snet(-pi/2)
snet(pi/2)
sol.alg
ps=initial_params(chain)
sol.alg.chain(3,sol.alg.initθ)

sol.alg.initθ

Δt=0.01
prob = ODEProblem(linear, u0, tspan, p)
sol1 = solve(prob, Vern7(), saveat=Δt)
data = Array(sol1)
plot!(sol1.t,data)

x=collect(sol.t)
y = [ chain(t,sol.alg.initθ)[1] for t in x]

plot(sol.t,y)

node = NeuralPDE.NNODE(chain, opt)

node.chain(1.0, node.initθ)

y = [ node.chain(t,node.initθ)[1] for t in x]
plot(sol.t, y)

y1 = sin.(x)
plot(sol.t, y1)

## Test

alg = NeuralPDE.NNODE(chain, opt)
chain  = alg.chain
opt    = alg.opt
autodiff = alg.autodiff
u0 = prob.u0
tspan = prob.tspan
f = prob.f
p = prob.p
initθ = alg.initθ
t0 = tspan[1]
dt = 1/20f0
ts = tspan[1]:dt:tspan[2]
phi = (t,θ) -> u0 + (t-tspan[1])*first(chain(adapt(typeof(initθ),[t]),θ))
#phi = (t,θ) -> u0 + first(chain(adapt(typeof(initθ),[t]),θ))
phi(3,initθ)
dfdx = (t,θ) -> ForwardDiff.derivative(t->phi(t,θ),t)
#dfdx = (t,θ) -> (phi(t+sqrt(eps(t)),θ) - phi(t,θ))/sqrt(eps(t))
dfdx(3.0,initθ)
function inner_loss(t,θ)
    sum(abs2,dfdx(t,θ) - prob.f(phi(t,θ),p,t))
end
loss(θ) = sum(abs2,[inner_loss(t,θ) for t in ts]) # sum(abs2,inner_loss(t,θ) for t in ts) but Zygote generators are broken
abstol=1e-10
cb = function (p, l)
    println("Current loss is: $l")
    l < abstol
end
res = DiffEqFlux.sciml_train(loss, initθ, opt; cb = cb, maxiters=1000, alg.kwargs...)

loss(res.minimizer)

u = [first(phi(t,res.minimizer)) for t in ts]

plot(ts, u)

phi(0.25,res.minimizer)

adapt(typeof(initθ),[1])

typeof(initθ)
prob.f
prob.p

prob.f(phi(1,initθ),prob.p,1)
phi(1,initθ)

prob.f(-1.0, prob.p, 0.5)

cos(pi)
phi(0.5,res.minimizer)

prob.f(1, prob.p, 0.5)
