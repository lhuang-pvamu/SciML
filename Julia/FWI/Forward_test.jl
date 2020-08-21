cd(@__DIR__)
include("Forward_ODE_1d.jl")

U, Traces = forward_ODE_test()
SciML_Wave_1D_Training(U, Traces)
SciML_Wave_1D_Test()
