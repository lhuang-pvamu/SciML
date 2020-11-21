# NeuralPDE -- 1D wave equation with source term

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

macro seisrc(x,t)
    #wvlt_samp = ricker($t, fpeak)
    #println("Src: fpeak=",fpeak," xbgn,dx,xend=",xbgn,dx,xend)
    #seisrc = point_source(wvlt_samp, $x, xbgn,dx,xend)
    return :( println("point_source( ricker($t, fpeak), $x, xbgn,dx,xend)") )
end

# Build a Ricker wavelet with a given peak frequency (Hz)
function ricker(t, fpeak)
    sigmaInv = pi * fpeak * sqrt(2)
    #cut = 1.e-6
    t0 = 6.0 / sigmaInv
    Δt = t .- t0
    expt = (pi * fpeak .* Δt) .^ 2
    (1.0 .- 2.0 .* expt) .* exp.(-expt)
end

# Place a source wavelet sample at a position in space
function point_source( wvlt_samp, xsrc, xbgn,dx,xend)
    nx = Int( 1 + round( (xend-xbgn)/dx ) )
    dx = (xend - xbgn)/(nx-1)
    f = zeros(nx, 1)
    # Find the first spatial sample point beyond the source location
    xpos = xsrc - xbgn
    xndx = xpos / dx
    ixs = max(1, ceil(Int,xndx)) + 1
    # Distribute the unit amplitude proportionally
    # between the two neighboring sample positions
    frac = (ixs * dx - xpos) / dx
    #print(frac, ", ", ixs, ", ", xpos)
    f[ixs, 1] = (1.0 - frac) * wvlt_samp
    f[ixs-1, 1] = frac * wvlt_samp
    println("point_source: f=",f)
    f
end

# macro test
fpeak = 10.
xbgn = 0.
xend = 6000.
xsrc = 900.
dx = 20.
tlast = 3.
fpeak = 10.

mex = @macroexpand( @seisrc( xsrc, .2 ))
arr = eval(mex)
arr1 = eval(arr)
arr1
arr2 = eval( @seisrc( xsrc, .2 ) )
arr5
arr2
mex

println( arr3[0] )
point_source( ricker(t, fpeak), x, xbgn,dx,xend)
#2D PDE
C=2000.
eq  = Dtt(u(x,t,θ)) ~ C^2*Dxx(u(x,t,θ)) + @seisrc(x,t)

# Initial and boundary conditions
bcs = [u(0,t,θ) ~ 0., # for all t > 0
       u(1,t,θ) ~ 0., # for all t > 0
       u(x,0,θ) ~ 0., # was: x*(1. - x), #for all 0 < x < 1
       Dt(u(x,0,θ)) ~ 0. ] #for all  0 < x < 1]

# Space and time domains
xbgn = 0.
xend = 6000.
dx = 20.
tlast = 3.
fpeak = 10.
domains = [x ∈ IntervalDomain(xbgn,xend),
           t ∈ IntervalDomain(0.0,tlast)]

# Discretization
dx = 20.

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

u_predict = reshape([first(phi([x,t],result.minimizer)) for x in xs for t in ts],(length(xs),length(ts)))

wvPlt =plot(xs, ts, u_predict, linetype=:contourf,title = "predict");
plot(wvPlt)
