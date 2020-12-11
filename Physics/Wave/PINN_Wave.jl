cd(@__DIR__)
using Pkg;
Pkg.activate("..");
Pkg.instantiate();
using LinearAlgebra
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using OrdinaryDiffEq
using Plots
gr()

output_figures="Figures/"
output_models="Models/"

@parameters x, t
@variables u(..)
@derivatives Dxx''~x
@derivatives Dtt''~t
@derivatives Dt'~t
@derivatives Dx'~x
#2D PDE
# Discretization
dx = 0.05
dt = 0.05
len = Int(1.0/dx)+1
c=Array(ones(len))
c[Int(round(len/2))]=2.0
plot(c)

function ricker(t, fpeak)
    sigmaInv = pi * fpeak * sqrt(2)
    #cut = 1.e-6
    t0 = 6.0 / sigmaInv
    Δt = t .- t0
    expt = (pi * fpeak .* Δt) .^ 2
    wv = (1.0 .- 2.0 .* expt) .* exp.(-expt)
end

function vel(x)
    index = Int(round((x-0.0)/dx))+1
    c[index]
end

@register vel(x)
vel(1)

r = ricker(0.0:dx:2.0, 2)
plot(r)

function seisrc(x,t,fpeak)
    index = Int(round((t-0.0)/dt))+1
    #print(index)
    if x==0.5
        r[index]
    else
        0.0
    end
end

seisrc(0.5,0.5,5)

# Space and time domains
domains = [x ∈ IntervalDomain(0.0,1.0),
           t ∈ IntervalDomain(0.0,1.0)]

xs,ts = [domain.domain.lower:dx:domain.domain.upper for domain in domains]
ts
source = reshape([seisrc(x,t,2.0) for x in xs for t in ts], (length(ts),length(xs)))
heatmap(source)
plot(source[:,11])

@register seisrc(x,t,fpeak)

eq  = 1.0/vel(x)^2 *  Dtt(u(x,t)) ~ Dxx(u(x,t)) + seisrc(x,t,5)
#eq  = Dtt(u(x,t)) ~ Dxx(u(x,t)) + seisrc(x,t,5)

# Initial and boundary conditions
#bcs = [u(0,t) ~ 0.,# for all t > 0
#       u(1,t) ~ 0.,# for all t > 0
#       u(x,0) ~ x*(1. - x), #for all 0 < x < 1
#       Dt(u(x,0)) ~ 0.] #for all  0 < x < 1]

bcs = [u(x,0) ~ 0.,
        #u(0,t) ~ 0.,
        #u(1,t) ~ 0.,
        Dt(u(x,0)) ~ 0.,
        Dt(u(0,t)) ~ vel(0)*Dx(u(0,t)),
        Dt(u(1,t)) ~ vel(1)*Dx(u(1,t))]

# Neural network
chain = FastChain(FastDense(2,32,Flux.σ),FastDense(32,32,Flux.σ),FastDense(32,1))

discretization = PhysicsInformedNN(dx, chain,
                                   strategy= GridTraining())

pde_system = PDESystem(eq,bcs,domains,[x,t],[u])
prob = discretize(pde_system,discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end
# optimizer
opt = Optim.BFGS()
#opt = Optim.Newton()
@time res = GalacticOptim.solve(prob,opt; cb = cb, maxiters=200)
phi = discretization.phi


analytic_sol_func(x,t) =  sum([(8/(k^3*pi^3)) * sin(k*pi*x)*cos(vm(x)*k*pi*t) for k in 1:2:50000])

u_predict = reshape([first(phi([x,t],res.minimizer)) for x in xs for t in ts],(length(ts),length(xs)))
u_real = reshape([analytic_sol_func(x,t) for x in xs for t in ts], (length(ts),length(xs)))
u_real
diff_u = abs.(u_predict .- u_real)
p1 = plot(xs, ts, u_real, linetype=:contourf,title = "analytic");
p2 =plot(xs, ts, u_predict, linetype=:contourf,title = "predict");
p3 = plot(xs, ts, diff_u,linetype=:contourf,title = "error");
plot(p1,p2,p3)
plot(p2)
savefig(output_figures*"wave_plot.png")

maximum(u_predict)
minimum(u_predict)

heatmap(u_predict)

plot(u_predict[:,:])
