cd(@__DIR__)
using Pkg; Pkg.activate(".."); Pkg.instantiate()

# Single experiment, move to ensemble further on
# Some good parameter values are stored as comments right now
# because this is really good practice

using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra, DiffEqSensitivity, Optim
using DiffEqFlux, Flux
using Plots
gr()
#plotly()

function corona!(du,u,p,t)
    S,E,I,R,N,D,C = u
    F, β0,α,κ,μ,σ,γ,d,λ = p
    dS = -β0*S*F/N - β(t,β0,D,N,κ,α)*S*I/N -μ*S # susceptible
    dE = β0*S*F/N + β(t,β0,D,N,κ,α)*S*I/N -(σ+μ)*E # exposed
    dI = σ*E - (γ+μ)*I # infected
    dR = γ*I - μ*R # removed (recovered + dead)
    dN = -μ*N # total population
    dD = d*γ*I - λ*D # severe, critical cases, and deaths
    dC = σ*E # +cumulative cases

    du[1] = dS; du[2] = dE; du[3] = dI; du[4] = dR
    du[5] = dN; du[6] = dD; du[7] = dC
end
β(t,β0,D,N,κ,α) = β0*(1-α)*(1-D/N)^κ
S0 = 3.0e8
u0 = [0.9*S0, 0.0, 1000.0, 0.0, S0, 0.0, 0.0]
p_ = [10.0, 0.5944, 0.4239, 1117.3, 0.02, 1/3, 1/5,0.2, 1/11.2]
R0 = p_[2]/p_[7]*p_[6]/(p_[6]+p_[5])
tspan = (0.0, 300.0)
prob = ODEProblem(corona!, u0, tspan, p_)
solution = solve(prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat = 1)

plot(solution)

scatter(solution, vars=[2,3,4], label=["True Exposed" "True Infected" "True Recovered"])

plot(solution, vars=[7])
