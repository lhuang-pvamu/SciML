# NeuralPDE -- 1D wave equation with source term

cd(@__DIR__)
using Pkg;
Pkg.activate("..");
# Pkg.activate("~/.julia/environments/v1.5/Project.toml");
Pkg.instantiate();

using GalacticOptim, Optim
using NeuralPDE, Flux, ModelingToolkit, DiffEqFlux
using Plots

println("NeuralPDE -- 1D wave equation with source term")

# Problem definition parameters
xbgn = 0. ;  xend = 0.6 # 6000.
psrc = 0.1;  xsrc = (xend-xbgn)*psrc + xbgn
tlast = 3.
fpeak = 2.
C= 0.2 # 2000.  # 1
SSQ = 1. / C^2   # Slowness squared

nGrid = 49
netSz = 64
dn = 1. / (nGrid-1)  # Discretization fraction in both x and t domains
dx = dn * (xend - xbgn)
dt = round(dn * tlast; sigdigits=3 )
println("For nGrid= "*string(nGrid)*",  dx= "*string(dx)*"  dt= "*string(dt))

xs = xbgn:dx:xend
ixsrc = Int(round( (xsrc-xbgn)/dx )) + 1
ts = 0:dt:tlast

# Build a Ricker wavelet with a given peak frequency (Hz)
function ricker(t, fpeak)
    sigmaInv = pi * fpeak * sqrt(2)
    cut = 1.e-6
    t0 = 6.0 / sigmaInv
    Δt = t .- t0
    expt = (pi * fpeak .* Δt) .^ 2
    wv = (1.0 .- 2.0 .* expt) .* exp.(-expt)
    for (i,w) in enumerate(wv); wv[i] = (abs(w) > cut ? w : 0.); end
    wv
end

ricksrc = ricker(ts, fpeak)
plot(ricksrc)

# Define a wavelet at a single grid point in x-space
function seisrc(x, t)
    if abs(x-xsrc) >= dx; return 0.; end
    ixt = Int(round((t-0.0)/dt))+1
    w = ricksrc[ixt]
    #println("seisrc: x= "*string(x)*"  t= "*string(t)*",  w= "*string(w))
    #w
end

# plot the source field
source = reshape([seisrc(x,t) for x in xs for t in ts], (length(ts),length(xs)))
heatmap(xs,ts,source)

#2D PDE
@parameters x, t
@variables u(..)
@derivatives Dxx''~x
@derivatives Dtt''~t
@derivatives Dt'~t

@register seisrc(x,t)
#=
# Initial point source
function ixpoint(x)
    ( abs(x-xsrc) < dx ? 1.0 : 0.0 )
end
@register ixpoint(x)
=#

# Initial ramp function
function ramps(x)
    ret = (x - xbgn)*(xend - x)
end
@register ramps(x)

# Exponential spike
function espike(x, x0)
    ret = exp( -100. .* abs(x-x0) )
end
@register espike(x, x0)

evel = zeros( length(xs) )
for (i,x) in enumerate(xs); evel[i] = espike(x,xsrc); end
plot(evel)

eqn = SSQ * Dtt(u(x,t)) ~ Dxx(u(x,t))   # + seisrc(x,t)

# Space and time domains
xdom = x ∈ IntervalDomain(xbgn,xend)
tdom = t ∈ IntervalDomain(0.0,tlast)
domains = [xdom,tdom]
#println(">>>>> domains:...")
#dump(domains,maxdepth=3)

# Initial and boundary conditions
bcs = [u(xbgn,t) ~ 0., # for all t > 0
       u(xend,t) ~ 0., # for all t > 0
       #u(x,0) ~ (x - xbgn)*(xend - x), # for all 0 < x < 1
       u(x,0) ~ 0.,
       Dt(u(x,0)) ~ espike(x,xsrc) ] # for all x in domain
#println(">>>>> bcs:...")
#dump(bcs,maxdepth=3)

# Build a discretized NN struct
chain = FastChain(FastDense(2,netSz,Flux.σ),FastDense(netSz,netSz,Flux.σ),FastDense(netSz,1))

discreteNet = PhysicsInformedNN(dn, chain, strategy= GridTraining())
#println(">>>>> discreteNet:...")
#dump(discreteNet, maxdepth=3)

pde_system = PDESystem(eqn, bcs, domains, [x,t], [u] )
#println(">>>>> pde_system:...")
#dump(pde_system, maxdepth=3)

# Callback for each iteration
losses = []
capture = function (p,loss)
    global losses
    losses = push!(losses,loss)
    iter = length(losses)
    if iter%20 == 0 || iter < 20
        println("At iter "*string(iter)*",  loss= "*string(loss))
    end
    return false
end

# Set up the optimization problem
prob = discretize(pde_system, discreteNet)

opt = Optim.BFGS()

println(">>>>> Solving...")
@time result = GalacticOptim.solve(prob, opt;  cb=capture, maxiters=2400)
phi = discreteNet.phi  # trial solution

println(">>>>> Plotting result...")
lossPlt = plot(losses, yscale=:log10, title="Seis1D_NN: Loss history")
plot(lossPlt)

u_predict = reshape([first(phi([x,t],result.minimizer)) for x in xs for t in ts],(length(xs),length(ts)))
phi = discreteNet.phi  # trial solution

wvPlt = plot(xs, ts, u_predict, linetype=:contourf, title="Seis1D_NN: wavefield u(x,t)")
plot(wvPlt)
