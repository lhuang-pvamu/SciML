cd(@__DIR__)
using Pkg;
Pkg.activate("..");
Pkg.instantiate();

using DifferentialEquations, Flux, Optim, DiffEqFlux, DiffEqSensitivity
using OrdinaryDiffEq
using Plots
gr()
#using Zygote
#using Zygote: gradient, @ignore
using HDF5
using ReverseDiff
#using ForwardDiff
include("Robot_Config.jl")

output_figures="Figures/"
output_models="Models/"

gravity = -4.8f0
friction = 2.5f0
damping = 15
Δt = 0.001f0
goal = Float32[1.0,0.11]

bolts
compute_center(bolts)

ann = FastChain(FastDense(size(bolts)[2]*6+12,32,tanh),FastDense(32, size(springs)[2],tanh))

if isfile(output_models*"spring_robot_model_C.h5")
    p_ann = h5read(output_models*"spring_robot_model_C.h5", "weights")
else
    p_ann = initial_params(ann)
end
p_ann = initial_params(ann)

function apply_spring_force(u, v, p, t)
    sinwaves = [sin(t*Δt*10 + 2pi/10 * i) for i in 1:10] .|> Float32
    #actuation = ones(length(springs[1,:]))*sin(10*t*2pi)*0.1
    c = compute_center(u)
    append!(sinwaves, (u.-c))
    append!(sinwaves, u)
    append!(sinwaves, v*0.1)
    append!(sinwaves, (goal-c)*0.5)
    #@show sinwaves
    #input = Float32.(vcat(sinwaves,vec(u[1:2,:].-center),vec(u[3:4,:])))
    actuation = ann(hcat(sinwaves),p).*springs[ACTUATION, :]
    inc = []
    for i in 1:length(u[1,:])
        append!(inc, [[]])
    end
    for i in 1:length(springs[1,:])
        a = Int(springs[SOURCE,i])
        b = Int(springs[TARGET,i])
        #print(a,b)
        direction = u[:,a] - u[:,b]
        distance = sqrt(sum(direction.^2))
        #print(typeof(distance))
        target_distance = springs[RESTLEN,i] * (1.0f0 + actuation[i])  #convert(Float64,actuation[i])# (1.0 + springs[ACTUATION, i]*convert(Float64,actuation[i])) #sin(t+2pi/10*i))
        #print(typeof(target_distance))
        Δd = distance - target_distance
        #println(Δd)
        impulse = Δt * Δd * (springs[STIFFNESS, i] / distance) * direction
        #println(distance - target_distance)
        append!(inc[a], [-impulse])
        append!(inc[b], [impulse])
    end
    hcat([sum(inc[i]) for i in 1:length(u[1,:])]...)
end

function advance(state,du,t)
    #len = length(state)
    new_state = []
    for i in 1:length(bolts[1,:])
        s = Float32(exp(-Δt * damping))
        old_v = s*state[3:4,i] + Δt*gravity*[0.0f0,1.0f0] + du[:,i]
        old_x = state[1:2,i]
        new_v = old_v
        #new_x = old_x + Δt * new_v
        toi = 0.0f0
        if old_x[2] < 0 && new_v[2] < 0
            toi = -(old_x[2] / old_v[2])
            new_v = [0.0f0,abs(new_v[2])*0.1f0]
        end
        new_x = old_x + toi*old_v + (Δt-toi) * new_v

        append!(new_state, [vcat(new_x, new_v)])
    end
    hcat(new_state...)
end



function goforward(θ)
    u0 = bolts
    v0 = Float32.(vcat(bolts,zeros(size(bolts))))
    total_step =  2
    u = u0
    v = zeros(size(bolts))
    state = [] #Vector{typeof(u0)}()
    #s = Vector{Int}()
    append!(state,[v0])
    newstate = v0

    for t in 1:total_step
        du = apply_spring_force(u, v, θ, t)
        newstate = advance(newstate, du, t)
        #@show newstate
        u = newstate[1:2,:]
        v = newstate[3:4,:]
        #print(u)
        #@ignore push!(s, u0)
        append!(state,[newstate])
        #append!(state, [du])
    end
    hcat(state...)
    #u
end


state = goforward(p_ann)
data = reshape(state, (4,size(bolts)[2],:))

# No regularisation right now
function forward_loss(θ)
    data = goforward(θ)
    #sum(abs2, data)
    pred = reshape(data, (4,size(bolts)[2],:))
    #sum(abs2, data[1] .- pred[1,:]) + sum(abs2, data[3] .- pred[3,:]), pred
    #sum(abs2, target[1] .- pred[1,:]) +
    #-sum(pred[1,:,end]) + (sum(abs2, get_distance.(pred[1:2,1,:],pred[1:2,6,:]) .- distance16)
    #+ sum(abs2, get_distance.(pred[1:2,3,:],pred[1:2,4,:]) .- distance34))*0.1 , pred
    -sum(pred[1,:,end])
    #-(pred[1,1,end]+pred[1,3,end])+
    #-(pred[1,1,end]+pred[1,3,end]+pred[1,13,end]+pred[1,15,end]) +
    #sum(abs2, compute_center(pred[1:2,:,end])-goal) +#, pred
    sum(abs2,pred[2,:,:].- u0[2,:])*0.01 #, pred
    #sum(abs2, refdata[:,1,:] - pred[1:2,1,1:length(refdata[1,1,:])]) +
    #sum(abs2, refdata[:,2,:] - pred[1:2,3,1:length(refdata[1,1,:])]), pred
    #+ sum(abs2,pred[2,:,:].-u0[2,:])*0.1 +
    #sum(abs2,((pred[1,3,:]-pred[1,1,:]).-(u0[1,3]-u0[1,1])))*0.1, pred
    #sum(abs2, ref_data[:,1:length(pred[1,:])] .- pred) * 0.01, pred #[:,Int(round(length(pred[1,:])/2)):end]), pred
end

#res1 = DiffEqFlux.sciml_train(forward_loss, p_ann, ADAM(0.01), cb=callback, maxiters = 50;
#        sensealg=ReverseDiffAdjoint())
#p_ann = res1.minimizer

Zygote.gradient(p -> forward_loss(p), p_ann)
forward_loss(p_ann)
gs = ReverseDiff.gradient(p -> forward_loss(p), p_ann)
#gs = ForwardDiff.gradient(p -> forward_loss(p), p_ann)
opt = ADAM(0.001)
epochs = 100
Flux.Optimise.update!(opt, p_ann, gs)
callback() = println("Loss = $(forward_loss(p_ann))")

for epoch in 1:epochs
    gs = ReverseDiff.gradient(p -> forward_loss(p), p_ann)
    Flux.Optimise.update!(opt, p_ann, gs)
    if epoch % 2 == 1
        callback()
    end
end

forward_loss(p_ann)

#data1 = Iterators.repeated((), 200)
#Flux.train!(forward_loss, p_ann, data1, ADAM(0.05), cb = callback)

state = goforward(p_ann)
data = reshape(state, (4,size(bolts)[2],:))

plot(data[1:2,2,:]')
anim = anim_robot(data, springs; step=100)
gif(anim, output_figures*"anim_spring_robot.gif", fps = 10)

weights = p_ann
fid=h5open(output_models*"spring_robot_model_A.h5","w")
fid["weights"] = weights
close(fid)

##

visualize_robot(bolts,springs)


u0 = bolts
v0 = Float32.(vcat(bolts,zeros(size(bolts))))
st = []
append!(st,[v0])
st[1][3:4, 1]
du = apply_spring_force(u0, zeros(size(bolts)), p_ann, 1)
newstate = advance(st[1], du, 1)
u=u0
direction = u[:,1] - u[:,2]
distance = sqrt(sum(direction.^2))
eltype(distance)
springs[RESTLEN,1] * (1.0 + springs[ACTUATION, 1]*0.1) * 1.0
sinwaves = [sin(1*Δt*10 + 2pi/10 * i) for i in 1:10]
