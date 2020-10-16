## Generate some data by solving a differential equation
########################################################

using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq

using LinearAlgebra
using Plots
gr()

# Create a test problem
function lorenz(u,p,t)
    x, y, z = u
    ẋ = 10.0*(y - x)
    ẏ = x*(28.0-z) - y
    ż = x*y - (8/3)*z
    return [ẋ, ẏ, ż]
end

u0 = [-8.0; 7.0; 27.0]
p = [10.0; -10.0; 28.0; -1.0; -1.0; 1.0; -8/3]
tspan = (0.0,100.0)
dt = 0.001
problem = ODEProblem(lorenz,u0,tspan)
solution = solve(problem, Tsit5(), saveat = dt, atol = 1e-7, rtol = 1e-8)

X = Array(solution)
DX = similar(X)
for (i, xi) in enumerate(eachcol(X))
    DX[:,i] = lorenz(xi, [], 0.0)
end

## Now automatically discover the system that generated the data
################################################################

@variables x y z
u = Operation[x; y; z]
polys = Operation[]
for i ∈ 0:4
    for j ∈ 0:i
        for k ∈ 0:j
            push!(polys, u[1]^i*u[2]^j*u[3]^k)
            push!(polys, u[2]^i*u[3]^j*u[1]^k)
            push!(polys, u[3]^i*u[1]^j*u[2]^k)
        end
    end
end

basis = Basis(polys, u)

opt = STRRidge(0.1)
Ψ = SINDy(X, DX, basis, opt, maxiter = 100, normalize = true)
print_equations(Ψ)
get_error(Ψ)
