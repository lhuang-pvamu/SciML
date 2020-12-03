# NeuralPDE -- 1D wave equation with source term

cd(@__DIR__)
using Pkg;
Pkg.activate("..");
# Pkg.activate("~/.julia/environments/v1.5/Project.toml");
Pkg.instantiate();

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Plots

println("NeuralPDE -- 1D wave equation with source term")

@parameters x, t, θ
#@parameters xbgn, xend, tlast
@variables u(..)
@derivatives Dxx''~x
@derivatives Dtt''~t
@derivatives Dt'~t

# Build a Ricker wavelet with a given peaNeuralPDE -- 1D wave equation with source termk frequency (Hz)
function ricker(t, fpeak)
    sigmaInv = pi * fpeak * sqrt(2)
    #cut = 1.e-6
    t0 = 6.0 / sigmaInv
    Δt = t .- t0
    expt = (pi * fpeak .* Δt) .^ 2
    wv = (1.0 .- 2.0 .* expt) .* exp.(-expt)
end

# Place a source wavelet sample at a position in space
function point_source( wvlt_samp, xsrc, xbgn,dx,xend)
    nx = Int( 1 + round( (xend-xbgn)/dx ) )
    dx = (xend - xbgn)/(nx-1)
    f = zeros(nx, 1)
    # Find the first spatial sample point beyond the source location
    ixs = 46
    frac = 0.
    #=
    xpos = xsrc - xbgn
    xndx = 1 + trunc( xpos / dx + .5 )
    println("xpos: ",xpos," dx: ", dx, " xndx: ",xndx)
    ixs = Int(xndx)
    #ixs = max(1, ceil(Int,xndx)) + 1
    # Distribute the unit amplitude proportionally
    # between the two neighboring sample positions
    frac = (ixs * dx - xpos) / dx
    #print(frac, ", ", ixs, ", ", xpos)
    =#
    f[ixs, 1] = (1.0 - frac) * wvlt_samp
    f[ixs-1, 1] = frac * wvlt_samp
    #println("point_source: f=",f)
    f
end

function seisrc(x,t)
#=
    wvlt_samp = ricker(t, fpeak)
    s = point_source(wvlt_samp, x, xbgn,dx,xend)
    s
=#
    s = 0.
end
#@register seisrc(x,t)

#2D PDE
C= 2. # 1. # 2000.  # 1.
eqn = Dtt(u(x,t,θ)) ~ C^2*Dxx(u(x,t,θ)) # + seisrc(x,t)

# Space and time domains
xbgn = 0. ; xend = 6. # 6000.
tlast = 3.
fpeak = 10.
xdom = x ∈ IntervalDomain(xbgn,xend)
tdom = t ∈ IntervalDomain(0.0,tlast)
domains = [xdom,tdom]
#println(">>>>> domains:...")
#dump(domains,maxdepth=3)

# Initial and boundary conditions
bcs = [u(xbgn,t,θ) ~ 0., # for all t > 0
       u(xend,t,θ) ~ 0., # for all t > 0
       u(x,0,θ) ~ (x - xbgn)*(xend - x), #  x*(1. - x), # for all 0 < x < 1
       Dt(u(x,0,θ)) ~ 0. ] # for all x in domain
#println(">>>>> bcs:...")
#dump(bcs,maxdepth=3)

# Build a discretized NN struct
nGrid = 16
dn = 1. / (nGrid-1)  # Discretization fraction in both x and t domains
dx = dn * (xdom.domain.upper - xdom.domain.lower)
xs = xdom.domain.lower:dx:xdom.domain.upper
dt = round(dn * (tdom.domain.upper - tdom.domain.lower); sigdigits=3 )
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
