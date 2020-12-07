# NeuralPDE example 2 -- 1D wave equation initial value problem

cd(@__DIR__)
using Pkg;
Pkg.activate("..");
# Pkg.activate("~/.julia/environments/v1.5/Project.toml");
Pkg.instantiate();

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Plots

@parameters x, t, θ
@variables u(..)
@derivatives Dxx''~x
@derivatives Dtt''~t
@derivatives Dt'~t

#2D PDE
C=1
eq  = Dtt(u(x,t,θ)) ~ C^2*Dxx(u(x,t,θ))

# Initial and boundary conditions
bcs = [u(0,t,θ) ~ 0.,# for all t > 0
       u(1,t,θ) ~ 0.,# for all t > 0
       u(x,0,θ) ~ x*(1. - x), #for all 0 < x < 1
       Dt(u(x,0,θ)) ~ 0. ] #for all  0 < x < 1]

# Space and time domains
domains = [x ∈ IntervalDomain(0.0,1.0),
           t ∈ IntervalDomain(0.0,1.0)]
# Discretization
dx = 0.1

# Neural network
chain = FastChain(FastDense(2,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

discretization = PhysicsInformedNN(dx,
                                   chain,
                                   strategy= GridTraining())

pde_system = PDESystem(eq,bcs,domains,[x,t],[u])
prob = discretize(pde_system,discretization)

# optimizer
opt = Optim.BFGS()
#res = GalacticOptim.solve(prob,opt; cb = cb, maxiters=1200)
result = GalacticOptim.solve(prob,opt;  maxiters=1200)
phi = discretization.phi
# We can plot the predicted solution of the PDE and compare it with the analytical solution in order to plot the relative error.

xs,ts = [domain.domain.lower:dx:domain.domain.upper for domain in domains]
analytic_sol_func(x,t) =  sum([(8/(k^3*pi^3)) * sin(k*pi*x)*cos(C*k*pi*t) for k in 1:2:50000])

u_predict = reshape([first(phi([x,t],result.minimizer)) for x in xs for t in ts],(length(xs),length(ts)))
u_real = reshape([analytic_sol_func(x,t) for x in xs for t in ts], (length(xs),length(ts)))

diff_u = abs.(u_predict .- u_real)
p1 = plot(xs, ts, u_real, linetype=:contourf,title = "analytic");
p2 =plot(xs, ts, u_predict, linetype=:contourf,title = "predict");
p3 = plot(xs, ts, diff_u,linetype=:contourf,title = "error");
plot(p1,p2,p3)
