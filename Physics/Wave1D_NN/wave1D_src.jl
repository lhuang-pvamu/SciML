# NeuralPDE -- 1D wave equation with source term

cd(@__DIR__)
using Pkg;
Pkg.activate("..");
# Pkg.activate("~/.julia/environments/v1.5/Project.toml");
Pkg.instantiate();

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Plots

println("NeuralPDE -- 1D wave equation with source term")

# Problem definition parameters
xbgn = 0. ;  xend = 0.6 # 6000.
psrc = 0.1;  xsrc = (xend-xbgn)*psrc + xbgn
tlast = 3.
fpeak = 10.
C= 0.2 # 2000.  # 1.

nGrid = 11
dn = 1. / (nGrid-1)  # Discretization fraction in both x and t domains
dx = dn * (xend - xbgn)
dt = round(dn * tlast; sigdigits=3 )

# Build a Ricker wavelet with a given peak frequency (Hz)
function ricker(t, fpeak)
    sigmaInv = pi * fpeak * sqrt(2)
    cut = 1.e-6
    t0 = 6.0 / sigmaInv
    Δt = t .- t0
    expt = (pi * fpeak .* Δt) .^ 2
    wv = (1.0 .- 2.0 .* expt) .* exp.(-expt)
    abs(wv) > cut ? wv : 0.
end

# Define a wavelet at a single point in x-space
function seisrc(x::Number, t::Number)
    # abs(x-xsrc) > dx ? 0. : ricker(t, fpeak)
    println("seisrc: x= ",x,"  t= ",t)
    if abs(x-xsrc) > dx; return 0.; end
    ricker(t, fpeak)
end
function seisrc(x::Operation, t::Operation)
    println("seisrcOp: x= ",x,"  t= ",t)
    return 0.
end

#2D PDE
@parameters x, t, θ
@variables u(..)
@derivatives Dxx''~x
@derivatives Dtt''~t
@derivatives Dt'~t
@register seisrc(x,t)

eqn = Dtt(u(x,t,θ)) ~  C^2*Dxx(u(x,t,θ)) + seisrc(x,t)

# Space and time domains
xdom = x ∈ IntervalDomain(xbgn,xend)
tdom = t ∈ IntervalDomain(0.0,tlast)
domains = [xdom,tdom]
#println(">>>>> domains:...")
#dump(domains,maxdepth=3)

# Initial and boundary conditions
bcs = [u(xbgn,t,θ) ~ 0., # for all t > 0
       u(xend,t,θ) ~ 0., # for all t > 0
       u(x,0,θ) ~  (x - xbgn)*(xend - x), # for all 0 < x < 1
       Dt(u(x,0,θ)) ~ 0. ] # for all x in domain
#println(">>>>> bcs:...")
#dump(bcs,maxdepth=3)

# Build a discretized NN struct
xs = xdom.domain.lower:dx:xdom.domain.upper
ts = tdom.domain.lower:dt:tdom.domain.upper
println("xs= ",xs," len ",length(xs),"  ts= ",ts," len ",length(ts))

# Neural network
chain = FastChain(FastDense(2,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

discreteNet = PhysicsInformedNN(dn, chain, strategy= GridTraining())
#println(">>>>> discreteNet:...")
#dump(discreteNet, maxdepth=3)

pde_system = PDESystem(eqn, bcs, domains, [x,t], [u] )
#println(">>>>> pde_system:...")
#dump(pde_system, maxdepth=3)

prob = discretize(pde_system, discreteNet)

# optimizer
opt = Optim.BFGS()

println(">>>>> Solving...")
#res = GalacticOptim.solve(prob,opt; cb = cb, maxiters=1200)
@time result = GalacticOptim.solve(prob, opt;  maxiters=1200)
phi = discreteNet.phi  # trial solution

# We can plot the predicted solution of the PDE and
# compare it with the analytical solution in order to plot the relative error.
println(">>>>> Plotting...")

#xs,ts = [domain.domain.lower:dx:domain.domain.upper for domain in domains]

u_predict = reshape([first(phi([x,t],result.minimizer)) for x in xs for t in ts],(length(xs),length(ts)))

wvPlt = plot(xs, ts, u_predict, linetype=:contourf, title = "predict");
plot(wvPlt)
