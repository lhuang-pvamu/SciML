# Poisson 2D equation  Example 1 from Physics-Informed Neural Networks

cd(@__DIR__)
using Pkg;
Pkg.activate("..");
# Pkg.activate("~/.julia/environments/v1.5/Project.toml");
Pkg.instantiate();

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Plots

# special macros
@parameters x y θ
@variables u(..)
@derivatives Dxx''~x
@derivatives Dyy''~y

# 2D PDE
eq  = Dxx(u(x,y,θ)) + Dyy(u(x,y,θ)) ~ -sin(pi*x)*sin(pi*y)

# Boundary conditions
bcs = [u(0,y,θ) ~ 0.f0, u(1,y,θ) ~ -sin(pi*1)*sin(pi*y),
       u(x,0,θ) ~ 0.f0, u(x,1,θ) ~ -sin(pi*x)*sin(pi*1)]
# Space domains
domains = [x ∈ IntervalDomain(0.0,1.0),
           y ∈ IntervalDomain(0.0,1.0)]

# Here, we define the neural network, where
# the input of NN equals the number of dimensions and
# output equals the number of equations in the system.

# Neural network
dim = 2 # number of dimensions
chain = FastChain(FastDense(dim,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

# Here, we build PhysicsInformedNN algorithm where
# dx is the step of discretization and
# strategy stores information for choosing a training strategy.

# Discretization
dx = 0.05
discretization = PhysicsInformedNN(dx,
                                   chain,
                                   strategy = GridTraining())

# As described in the API docs, we now need to define the PDESystem and
# create PINNs problem using the discretize method.

pde_system = PDESystem(eq,bcs,domains,[x,y],[u])
prob = discretize(pde_system,discretization)

# Here, we define the callback function and the optimizer.
# And now we can solve the PDE using PINNs
# (with the number of epochs maxiters=1000).

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

#result = GalacticOptim.solve(prob, BFGS(); progress = false, cb = cb, maxiters=1000)
result = GalacticOptim.solve(prob, BFGS(); maxiters=1000)
phi = discretization.phi

# We can plot the predicted solution of the PDE and compare it with
# the analytical solution in order to plot the relative error.

xs,ys = [domain.domain.lower:dx/10:domain.domain.upper for domain in domains]
analytic_sol_func(x,y) = (sin(pi*x)*sin(pi*y))/(2pi^2)

u_predict = reshape([first(phi([x,y],result.minimizer)) for x in xs for y in ys],(length(xs),length(ys)))
u_real = reshape([analytic_sol_func(x,y) for x in xs for y in ys], (length(xs),length(ys)))
diff_u = abs.(u_predict .- u_real)

p1 = plot(xs, ys, u_real, linetype=:contourf,title = "analytic");
p2 = plot(xs, ys, u_predict, linetype=:contourf,title = "predict");
p3 = plot(xs, ys, diff_u,linetype=:contourf,title = "error");
plot(p1,p2,p3)
