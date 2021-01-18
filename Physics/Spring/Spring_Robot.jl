cd(@__DIR__)
using Pkg;
Pkg.activate("..");
Pkg.instantiate();

using DifferentialEquations, Flux, Optim, DiffEqFlux, DiffEqSensitivity
using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra
using Plots
gr()
using Zygote
using HDF5
#using ControlSystems
#using Flux: throttle, @epochs
#using ReverseDiff
#using ForwardDiff
include("Robot_Config.jl")

output_figures="Figures/"
output_models="Models/"

##
# Build a robot based on springs
##
g= -4.8 * 0.001
m=1.0
damping = 0

## index


bolts, springs, connections = setup_robotA()
bolts, springs = setup_robotB()
bolts
springs
connections

sum(bolts,dims=2)
c = compute_center(bolts)

distance16 = get_distance(bolts[:,1], bolts[:,6])
distance34 = get_distance(bolts[:,3], bolts[:,4])


visualize_robot(bolts,springs)


u = bolts
v = zeros(size(u))
u0 = Float32.(vcat(u,v))
goal = Float32[1.0,0.05]
size(bolts)
ann = FastChain(FastDense(size(bolts)[2]*4+2,32,tanh),FastDense(32, size(springs)[2],tanh))
#ann = FastChain(FastDense(4,32,σ),FastDense(32, 1))

if isfile(output_models*"spring_robot_model.h5")
    p_ann = h5read(output_models*"spring_robot_model.h5", "weights")
else
    p_ann = initial_params(ann)
end
p_ann = initial_params(ann)
pvec = vcat(p_ann, ones(length(springs[1,:])))
count=1
damping=0.1 #exp(-Δt*15)
function springs_eq!(du, u, p, t)
    positions = u[1:2,:]
    #velocity = u[3:4,:]
    #(k1,k2,k3,k4,m,damping,x0,y0) = p
    #du[1:2,:] = velocity
    dampingForce = damping * u[3:4,:]
    springForce = []
    sinwaves = []
    for i in 1:length(u[1,:])
        append!(springForce, [[]])
        if positions[2,i] > 0.0
            append!(springForce[i], [[0.0, g*m]])
        end
    end
    #sinwaves = [sin(t*Δt*10 + 2pi/10 * i) for i in 1:10]
    #actuation = ones(length(springs[1,:]))*sin(10*t*2pi)*0.1
    c = compute_center(u[1:2,:])
    append!(sinwaves, (u[1:2,:].-c))
    append!(sinwaves, u[3:4,:]*1.0)
    append!(sinwaves, (goal-c)*0.5)
    #input = Float32.(vcat(sinwaves,vec(u[1:2,:].-center),vec(u[3:4,:])))
    actuation = ann(hcat(sinwaves),p[1:2878])
    stiffness = p[2879:end]*3e3
    #print(vec(u))
    for i in 1:length(springs[1,:])
        direction = positions[:,Int(springs[SOURCE,i])] - positions[:,Int(springs[TARGET,i])]
        distance = sqrt(sum(direction.^2))
        #if springs[ACTUATION, i]==0.0
        #    target_distance = springs[RESTLEN,i]
        #else
        target_distance = distance * (1.0 + actuation[i]*springs[ACTUATION, i]*0.3)
        #end
        #print()
        vec = direction / distance
        #magnitude = springs[STIFFNESS,i] * (target_distance - springs[RESTLEN, i])
        #@show stiffness
        magnitude = stiffness[i] * (target_distance - springs[RESTLEN, i])
        #@show magnitude
        append!(springForce[Int(springs[SOURCE,i])], [-vec * (magnitude)]) #+actuation[i]*0.1)])
        append!(springForce[Int(springs[TARGET,i])], [vec * (magnitude)]) #+actuation[i]*0.1)])
    end
    #mask = [] #ones(2,length(bolts[1,:]))
    #@show springForce
    #print(springForce)
    #velocity=[]
    #acceleration = []
    du[1:2,:]=u[3:4,:]
    for i in 1:length(u[1,:])
        du[3:4,i] = (sum(springForce[i]) *0.8) #- dampingForce[:,i])/m
        #append!(acceleration,[sum(springForce[i]) * 0.5 - dampingForce[:,i]/m])
        if u[2,i] <= 0.0 && u[4,i] < 0.0
            #append!(velocity,[[0,0]])
            du[:,i] = [0,-u[4,i]*0.1,0,0]
        #else
            #append!(velocity,[u[3:4,i]])
            #springForce[:,i] = dampingForce[:,i]
            #append!(mask,[i,0.0,0.0])
            #du[1,i] = 0.0
            #du[2,i] = 0.0 #abs(du[2,i]*0.2)
            #du[3,i] = 0.0
            #du[4,i] = dampingForce[2,i]/m #abs(du[4,i]*0.01)
        end
    end
    #velocity = velocity .* mask
    #acceleration = (springForce-dampingForce)/m
    #print(length(velocity))
    #vcat(hcat(velocity...), hcat(acceleration...))
end

#u0[2,2] = 0.15
#u0[2,3] = 0.10
u0
du = zeros(size(u0))
u1 = springs_eq!(du, u0, pvec, 1)
du

tspan = (0.0,3.0)
Δt = 0.01
prob = ODEProblem(springs_eq!, u0, tspan, pvec)
sol = solve(prob, Tsit5(), saveat=Δt, reltol=1e-3)
data = Array(sol)
t =tspan[1]:Δt:tspan[2]
count

function predict(θ)
    Array(solve(prob, Tsit5(); u0=u0, p=θ, saveat = Δt,reltol=1e-3,
                         #abstol=1e-5, reltol=1e-7))
                         #sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
                         sensealg=ReverseDiffAdjoint()))
end

# No regularisation right now
function loss(θ)
    pred = predict(θ)
    #sum(abs2, data[1] .- pred[1,:]) + sum(abs2, data[3] .- pred[3,:]), pred
    #sum(abs2, target[1] .- pred[1,:]) +
    #-sum(pred[1,:,end]) + (sum(abs2, get_distance.(pred[1:2,1,:],pred[1:2,6,:]) .- distance16)
    #+ sum(abs2, get_distance.(pred[1:2,3,:],pred[1:2,4,:]) .- distance34))*0.1 , pred
    #-sum(pred[1,:,end]), pred
    sum(abs2, compute_center(pred[1:2,:,end])-goal) +
    #sum(abs2,pred[2,:,:].- u0[2,:])*0.1, pred
    sum(abs2, refdata[:,1,:] - pred[1:2,1,1:length(refdata[1,1,:])]) +
    sum(abs2, refdata[:,2,:] - pred[1:2,4,1:length(refdata[1,1,:])]), pred
    #+ sum(abs2,pred[2,:,:].-u0[2,:])*0.1 +
    #sum(abs2,((pred[1,3,:]-pred[1,1,:]).-(u0[1,3]-u0[1,1])))*0.1, pred
    #sum(abs2, ref_data[:,1:length(pred[1,:])] .- pred) * 0.01, pred #[:,Int(round(length(pred[1,:])/2)):end]), pred
end

const losses = []
callback(θ,l,pred) = begin
    push!(losses, l)
    if length(losses)%5==0
        println(losses[end])
        pl = plot(pred[1:2,2,:]')
        #plot!(pl, pred')
        display(plot(pl))
    end
    false
end

res1 = DiffEqFlux.sciml_train(loss, pvec, ADAM(0.01), cb=callback, maxiters = 100)
pvec = res1.minimizer
res2 = DiffEqFlux.sciml_train(loss, pvec, BFGS(initial_stepnorm=0.01), cb=callback, maxiters = 100)
pvec = res2.minimizer
Zygote.gradient(p -> loss(p), p_ann)

pvec[3199:end]
loss(pvec)

data = predict(pvec)

plot3d(t, data[1,2,:],data[2,2,:])

plot(data[1:2,2,:]')

anim = anim_robot(data, springs; step=10)
gif(anim, output_figures*"anim_spring_robot.gif", fps = 10)

weights = p_ann
fid=h5open(output_models*"spring_robot_model.h5","w")
fid["weights"] = weights
close(fid)

refdata = zeros(2,2,301)
refdata[:,:,1] = [0.125, 0.0,
                0.225, 0.0]
refdata
for i in 2:100
    refdata[:,1,i]=[refdata[1,1,1],0.0]
    refdata[:,2,i]=[refdata[1,2,1]+(i-1)*0.0005,0.05]
end
for i in 101:200
    refdata[:,1,i]=[refdata[1,1,100]+(i-100)*0.0005,0.05]
    refdata[:,2,i]=[refdata[1,2,100],0.0]
end
for i in 201:301
    refdata[:,1,i]=[refdata[1,1,200],0.0]
    refdata[:,2,i]=[refdata[1,2,200]+(i-200)*0.0005,0.05]
end
refdata[:,1,301]=[refdata[1,1,300],0.0]
for i in 11:20
    refdata[1,i]=refdata[1,10]+(i-10)*0.01
    refdata[2,i]=refdata[2,10]
end
for i in 21:30
    refdata[1,i]=refdata[1,20]
    refdata[2,i]=refdata[2,20]+(i-20)*0.01
end
for i in 31:40
    refdata[1,i]=refdata[1,30]+(i-30)*0.01
    refdata[2,i]=refdata[2,30]
end
for i in 41:51
    refdata[1,i]=refdata[1,40]
    refdata[2,i]=refdata[2,30]+(i-40)*0.01
end
refdata[:,1,1:11]
plot(refdata[:,1,:]')
plot(refdata[:,2,:]')
##
# Euler integrater
##

function forward(func, u0, timespan, Δt, p)
    state = Vector{typeof(u0)}()
    push!(state, u0)
    u=u0
    for t in timespan[1]:Δt:timespan[2]
        du = zeros(size(u0))
        func(du, u, p, t)
        u = u+Δt*du
        push!(state,u)
    end
    state
end

sol = forward(springs_eq!, u0, tspan, 0.001, p_ann)

data = hcat(sol)
data = reshape(data, (4,6,:))

plot(data[1:2,2,1:5000]')
anim = anim_robot(data, springs; step=100)
gif(anim, output_figures*"anim_spring_robot.gif", fps = 10)
