cd(@__DIR__)
#using Pkg;
#Pkg.activate(".");
#Pkg.instantiate();

using DifferentialEquations
using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra, DiffEqSensitivity, Optim
using DiffEqFlux, Flux
using Plots
gr()

config = Dict()
config["dx"] = 0.005
config["x"] = 0.0:config["dx"]:1.0

function set_config!(config, c)
    config["dx"] = 0.005
    config["x"] = 0.0:config["dx"]:1.0
    alpha = 0.6
    dt = alpha * config["dx"] / maximum(c)
    config["dt"] = dt
    config["t"] = 0:config["dt"]:3.0
    # Source position(s)
    config["x_s"] = 0.1
    # Receiver position(s)
    config["x_r"] = [0.2, 0.4, 0.6]
    config["recr_m"] = zeros(length(config["x_r"]), length(config["x"]))
    for i = 1:length(config["x_r"])
        jr = argmin(abs.(config["x"] .- config["x_r"][i]))
        config["recr_m"][i,jr] = 1.0
        #traces[i, :] = UT[jr, :]
        #push!(traces,U[:, jr])
    end
    config
end

function ricker(t, nu0)
    sigmaInv = pi * nu0 * sqrt(2)
    cut = 1.e-6
    t0 = 6.0 / sigmaInv
    #println(t0)
    tdel = t .- t0
    expt = (pi * nu0 .* tdel) .^ 2
    (1.0 .- 2.0 .* expt) .* exp.(-expt)
end


function point_source(value, position, config)
    nx = length(config["x"])
    dx = config["dx"]
    xbgn = config["x"][1]
    xend = config["x"][nx]
    f = zeros(nx, 1)
    # Find the first spatial sample point beyond the source location
    xpos = position - xbgn
    ixs = convert(Int, max(1, ceil(xpos / dx))) + 1
    # Distribute the unit amplitude proportionally
    # between the two neighboring sample positions
    frac = (ixs * dx - xpos) / dx
    #print(frac, ", ", ixs, ", ", xpos)
    f[ixs, 1] = (1.0 - frac) * value
    f[ixs-1, 1] = frac * value
    f
end

#f = point_source(ricker(0.15, 10), 0.1, config)
#plot(f)

function velocity_model()
    xrange = config["x"]
    c = -100.0 * (xrange .- 0.5) .* exp.(-((xrange .- 0.5) .^ 2) / 1.e-4)
    c[abs.(c).<1e-7] .= 0
    c0 = ones(length(xrange))
    c = c0 .+ c
    c, c0
end



function set_matrics(c)
    M = Diagonal(1 ./ (c .^ 2))
    NS = length(config["x"])
    M[1, 1] = 0
    M[NS, NS] = 0
    #display(M)
    A = zeros(NS, NS)
    A[1, 1] = 0.5 / c[1]
    A[1, 2] = 0.5 / c[2]
    A[NS, NS-1] = 0.5 / c[NS-1]
    A[NS, NS] = 0.5 / c[NS]
    dx = config["dx"]
    Kxx =
        Diagonal(2 * ones(NS)) +
        diagm(1 => ones(NS - 1) * -1, -1 => ones(NS - 1) * -1)
    Kxx[1, :] .= 0
    Kxx[NS, :] .= 0
    Kxx = Kxx / (dx * dx)
    Kx = zeros(NS, NS)
    Kx[1, 1] = 1.0 / dx
    Kx[1, 2] = -1.0 / dx
    Kx[NS, NS-1] = -1.0 / dx
    Kx[NS, NS] = 1.0 / dx

    js = argmin(abs.(config["x"] .- config["x_s"]))
    S = zero(c)
    S[js] = 1.0 / dx
    dt = config["dt"]
    R = (1 / (dt^2)) .* M .+ (1 / dt) .* A
    RI = inv(R)
    #display(R)
    #heatmap(R)
    M, A, Kxx, Kx, S, R, RI
end



function forward(NT, NS, M, A, Kxx, Kx, dt, w, S, RI)
    U = zeros(NT, NS)
    Z = 2 / (dt^2) .* M + (1 / dt) .* A - Kxx - Kx
    for i = 2:NT-1
        Y1 = 1 / (dt^2) .* (M * U[i-1, :])
        Y2 = Z * U[i, :]
        U[i+1, :] = RI * (w[i] * S - Y1 + Y2)
    end
    U
end

#M, A, Kxx, Kx, S, R, RI = set_matrics(c)

#U = forward(NT, NS, M, A, Kxx, Kx, config["dt"], w, S, RI)

#heatmap(U)

function record_data(U)
    #traces = []
    UT = transpose(U)
    config["recr_m"]*UT
    #transpose(convert(Array,VectorOfArray(traces)))
    #traces
end

#traces = record_data(U)
#plot(traces[1, :])
#plot!(traces[2, :])
#plot!(traces[3, :])

function Forward_Driver()
    c, c0 = velocity_model()
    set_config!(config, c)
    NS = length(config["x"])
    plot(c)
    plot!(c0)
    NT = length(config["t"])
    w = ricker(config["t"], 10.0)
    plot(config["t"], w)
    M, A, Kxx, Kx, S, R, RI = set_matrics(c)
    Ux = forward(NT, NS, M, A, Kxx, Kx, config["dt"], w, S, RI)
    tr = record_data(Ux)
    heatmap(Ux)
    plot(tr[1, :])
    Ux, tr
end

U, tr = Forward_Driver()
heatmap(U)
plot(transpose(tr))

function F(c0)
    M, A, Kxx, Kx, S, R, RI = set_matrics(c0)
    NT = length(config["t"])
    NS = length(config["x"])
    Ux = forward(NT, NS, M, A, Kxx, Kx, config["dt"], w, S, RI)
    t = record_data(Ux)
    sum((traces - t) .^ 2)
end


#diff = F(c0)

function F_inverse()
    res = optimize(
        F,
        c0,
        LBFGS(),
        Optim.Options(show_trace = true, iterations = 5, store_trace=true, f_tol=1e-8),
    )

    summary(res)
    Optim.minimum(res)
    plot(res.minimizer)
    plot!(c)
    Ux, tr = Forward_Driver(res.minimizer)
    plot(tr[1, :])
    plot!(traces[1, :])
    heatmap(Ux)

    Optim.converged(res)

    display(res.trace)

    @show res

    t = Optim.trace(res)

    display(t)

    losses = []
    trace_time = []
    for i in 1:length(res.trace)
        append!(losses, parse(Float64, split(string(res.trace[i]))[2]))
        append!(trace_time, parse(Float64, split(string(res.trace[i]))[end]))
    end

    print(trace_time)
    plot(losses)
    plot(trace_time, log10.(losses/losses[end]))
    return res
end

#res = F_inverse()

#plot(res.minimizer)
#plot!(c0)
#plot!(c)

#Ux, tr = Forward_Driver(res.minimizer)
#plot(tr[1, :])
#heatmap(Ux)
