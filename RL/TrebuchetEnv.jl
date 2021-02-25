#export TrebucheEnv

using Trebuchet
using Random
using IntervalSets

struct TrebuchetEnvParams{T}
    min_angle::T
    max_angle::T
    min_wind::T
    max_wind::T
    min_distance::T
    max_distance::T
    weight::T
    max_steps::Int
end

function TrebuchetEnvParams(;
    T = Float64,
    min_angle = 0.0,
    max_angle = round(pi/2,digits=2),
    min_wind = -5.0,
    max_wind = 5.0,
    min_distance = 20.0,
    max_distance = 120.0,
    weight = 300,
    max_steps = 50,
    )
    TrebuchetEnvParams{T}(
    min_angle,
    max_angle,
    min_wind,
    max_wind,
    min_distance,
    max_distance,
    weight,
    max_steps,
    )
end

mutable struct TrebuchetEnv{A,T,ACT,R<:AbstractRNG} <: AbstractEnv
    params::TrebuchetEnvParams{T}
    action_space::A
    observation_space::Space{Vector{ClosedInterval{T}}}
    state::Vector{T}
    goal::T
    wind::T
    action::ACT
    done::Bool
    t::Int
    rng::R
end

function TrebuchetEnv(;
    T=Float64,
    continuous = false,
    rng = Random.GLOBAL_RNG,
    kwargs...,
    )
    params = TrebuchetEnvParams(; T = T, kwargs...)
    action_space = continuous ? ClosedInterval{T}(-1.0,1.0) : Base.OneTo(4)
    env = TrebuchetEnv(
        params,
        action_space,
        Space([params.min_angle..params.max_angle, params.min_wind..params.max_wind,
                params.min_distance.. params.max_distance]),
        zeros(T, 3),
        80.0,
        -1.0,
        rand(action_space),
        false,
        0,
        rng,
    )
    reset!(env)
    env
end


Random.seed!(env::TrebuchetEnv, seed) = Random.seed!(env.rng, seed)

RLBase.action_space(env::TrebuchetEnv) = env.action_space
RLBase.state_space(env::TrebuchetEnv) = env.observation_space

RLBase.state(env::TrebuchetEnv, ::Observation{Int}, p) = Int(
    round(round((env.state[3] - env.params.min_distance)*10) * (((env.params.max_wind-env.params.min_wind)*10+1) * ((env.params.max_angle-env.params.min_angle)*100+1)) +
    round((env.state[2] - env.params.min_wind)*10) * ((env.params.max_angle-env.params.min_angle)*100+1) +
    round(env.state[1] * 100+1))
    )

RLBase.state_space(env::TrebuchetEnv, ::Observation{Int}, p) =
    Base.OneTo(Int(((env.params.max_angle-env.params.min_angle)*100+1) *
    ((env.params.max_wind-env.params.min_wind)*10+1) *
    (env.params.max_distance-env.params.min_distance+1)*10 ))

function RLBase.reward(env::TrebuchetEnv{A,T}) where {A,T}
    if env.done
        zero(T)
    else
        ang, wind, distance = env.state
        t, d = Trebuchet.shoot(wind, ang, env.params.weight)
        - (d - env.goal)^2
    end
end

RLBase.is_terminated(env::TrebuchetEnv) = env.done
RLBase.state(env::TrebuchetEnv) = env.state

function RLBase.legal_action_space(env::TrebuchetEnv, p)
    A = []
    if round(env.state[1] - 0.1,digits=2) in env.params.min_angle:0.01:env.params.max_angle
        append!(A, 1)
    end
    if round(env.state[1] - 0.01,digits=2) in env.params.min_angle:0.01:env.params.max_angle
        append!(A, 2)
    end
    #append!(A,3)
    if round(env.state[1] + 0.01,digits=2) in env.params.min_angle:0.01:env.params.max_angle
        append!(A, 3)
    end
    if round(env.state[1] + 0.1,digits=2) in env.params.min_angle:0.01:env.params.max_angle
        append!(A, 4)
    end
    A
end



function RLBase.reset!(env::TrebuchetEnv{A,T}) where {A,T}
    env.wind = round(rand(env.rng, T) * 10.0 - 5.0, digits=1)
    env.goal = 80 #round(rand(env.rng, T) * 100.0 + 20, digits=1)
    env.state[1] = round(pi/2 * rand(env.rng, T), digits=2)
    env.state[2] = env.wind
    env.state[3] = env.goal
    env.done = false
    env.t = 0
    nothing
end

function (env::TrebuchetEnv{<:ClosedInterval})(a::AbstractFloat)
    #@show a
    #@assert a in env.action_space
    #a = clamp(a, -1, 1)
    #low = 1
    #high = 4
    #a =  Int(round(low + (a + 1) * 0.5 * (high - low)))
    env.action = a
    _step!(env, a)
end

function (env::TrebuchetEnv{<:Base.OneTo{Int}})(a::Int)
    @assert a in env.action_space
    high = 1.0
    low = -1.0
    float_a =  (high-low) * (a-1)/(4-1) + low
    env.action = a
    _step!(env, env.action) # decrease, keep, or increase the angle by one unit
end

function _step!(env::TrebuchetEnv, action::Int)
    env.t += 1
    angle, wind, distance = env.state
    if action == 1
        angle -= 0.1
    elseif action == 2
        angle -= 0.01
    elseif action == 3
        angle += 0.01
    else
        angle += 0.1
    end
    #angle += action * 0.1
    angle = round(angle, digits=2)
    @show angle
    if angle >= env.params.min_angle && angle <= env.params.max_angle #in env.params.min_angle:0.01:env.params.max_angle
        t, d = Trebuchet.shoot(wind, angle, env.params.weight)
        env.done =
            abs(d-env.goal) < 1.0 || env.t >= env.params.max_steps
        env.state[1]=angle
    else
        print("else")
        env.done = env.t >= env.params.max_steps
    end
    nothing
end

function _step!(env::TrebuchetEnv, action::AbstractFloat)
    #print("Float...")
    env.t += 1
    angle, wind, distance = env.state
    angle += action * 0.1
    angle = clamp(angle,env.params.min_angle, env.params.max_angle)
    #angle = round(angle, digits=2)
    #@show angle
    if angle >= env.params.min_angle && angle <= env.params.max_angle #in env.params.min_angle:0.01:env.params.max_angle
        t, d = Trebuchet.shoot(wind, angle, env.params.weight)
        env.done =
            abs(d-env.goal) < 1.0 || env.t >= env.params.max_steps
        env.state[1]=angle
    else
        #print("else")
        env.done = env.t >= env.params.max_steps
    end
    nothing
end

function set_goal!(env, goal)
    env.goal = goal
    env.state[3] = goal
end

function set_wind!(env, wind)
    env.wind = wind
    env.state[2] = wind
end

function render(env::TrebuchetEnv)
    ang, wind = env.state
    t, d = Trebuchet.shoot(wind, ang, env.params.weight)
    visualise(t, env.goal)
end


## Test
##

#env = TrebuchetEnv()
