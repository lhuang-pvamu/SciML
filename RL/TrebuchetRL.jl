cd(@__DIR__)
using Pkg;
Pkg.activate(".");
Pkg.instantiate();
Pkg.precompile();

using ReinforcementLearning
using Plots
using Zygote
using Flux
using Random
using IntervalSets
using StatsPlots
using BSON: @save, @load
using StableRNGs
using Dates
using Logging
using TensorBoardLogger: TBLogger

include("TrebuchetEnv.jl")

output_figures="Figures/"
output_models="Models/"
output_data="Data/"

function analysis()
	env = TrebuchetEnv()
	display(violin(
		[
			[
				begin
					reset!(env)
					env(a)
					reward(env)
				end
				for _ in 1:100
			]
			for a in action_space(env)
		],
		leg=false
	))
end

analysis()

# agent = Agent(
# 	policy = VBasedPolicy(;
# 			learner=MonteCarloLearner(;
# 				approximator = TabularVApproximator(;
# 					n_state=16117580,
# 					init=-1e6,
# 					opt=InvDecay(1.0)
# 				),
# 				γ = 1.0,
# 				kind = FIRST_VISIT,
# 				sampling = NO_SAMPLING
# 			),
# 			mapping = select_action
# 	),
# 	trajectory=VectorSARTTrajectory(;state=Int)
# )

function train()
	env = TrebuchetEnv()
	E = DefaultStateStyleEnv{Observation{Int}()}(env)
	explorer = EpsilonGreedyExplorer(0.1)
	agent = Agent(
		policy = QBasedPolicy(;
				learner=TDLearner(;
					approximator = TabularQApproximator(;
						n_state=16117580,
						n_action=4,
						init=-1e6,
						opt=InvDecay(1.0)
					),
					γ = 1.0,
					method=:SARSA,
					n = 0,
				),
				explorer = explorer
		),
		trajectory=VectorSARTTrajectory(;state=Int)
	)
	if isfile(output_models*"agent_Q.bson")
		@load output_models*"agent_Q.bson" table
		agent.policy.learner.approximator.table[:,:] = table[:,:]
	end
	reset!(E)
	## train the agent
	run(agent, E, StopAfterEpisode(1_000))
	@show maximum(table), minimum(table), count(table.>-1e6)
	## test the agent
	@show run(agent, E, StopAfterEpisode(10), StepsPerEpisode())
	table = agent.policy.learner.approximator.table
	@save output_models*"agent_Q.bson" table
end


train()
	## Try some random cases

function test()
	explorer = EpsilonGreedyExplorer(0.0)
	agent = Agent(
		policy = QBasedPolicy(;
				learner=TDLearner(;
					approximator = TabularQApproximator(;
						n_state=16117580,
						n_action=4,
						init=-1e6,
						opt=InvDecay(1.0)
					),
					γ = 1.0,
					method=:SARSA,
					n = 0,
				),
				explorer = explorer
		),
		trajectory=VectorSARTTrajectory(;state=Int)
	)
	if isfile(output_models*"agent_Q.bson")
		@load output_models*"agent_Q.bson" table
		agent.policy.learner.approximator.table[:,:] = table[:,:]
	end
	env = TrebuchetEnv()
	E = DefaultStateStyleEnv{Observation{Int}()}(env)
	reset!(E)
	@show state(env)
	#set_goal!(env, 80)
	rewards = [reward(E)]
	angles = [env.state[1]]
	while !is_terminated(E)
		action = agent(E)
		E(action)
		r = reward(E)
		@show action, r
		append!(rewards, r)
		append!(angles, env.state[1])
	end

	display(plot([rewards,angles],label=["rewards" "θ"], layout=(2,1)))
	savefig(output_figures*"exp_Q8.png")
	render(env)
end

test()


inner_env = TrebuchetEnv(continuous=true)
#inner_env = TrebuchetEnv()
state(inner_env)
A = action_space(inner_env)
inner_env(0.8)
state(inner_env)


t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_DDPG_Trebuchet_$(t)")


lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)
seed = 123
rng = StableRNG(seed)
init = glorot_uniform(rng)

ns = length(state(env))
low = 1
high = 4
action_mapping = x -> Int(round((high-low) * (x+1)/2 + low))
action_mapping(-0.33)

env = ActionTransformedEnv(
	inner_env;
	action_mapping =  x -> Int(round((high-low) * (x+1)/2 + low)),
)

env(1)

create_actor() = Chain(
	Dense(ns, 30, tanh; initW = init),
	Dense(30, 30, tanh; initW = init),
	Dense(30, 1, tanh; initW = init),
)

create_critic() = Chain(
	Dense(ns + 1, 30, relu; initW = init),
	Dense(30, 30, relu; initW = init),
	Dense(30, 1; initW = init),
)

agent = Agent(
	policy = DDPGPolicy(
		behavior_actor = NeuralNetworkApproximator(
			model = create_actor(),
			optimizer = ADAM(),
		),
		behavior_critic = NeuralNetworkApproximator(
			model = create_critic(),
			optimizer = ADAM(),
		),
		target_actor = NeuralNetworkApproximator(
			model = create_actor(),
			optimizer = ADAM(),
		),
		target_critic = NeuralNetworkApproximator(
			model = create_critic(),
			optimizer = ADAM(),
		),
		γ = 0.99f0,
		ρ = 0.995f0,
		batch_size = 64,
		start_steps = 1000,
		start_policy = RandomPolicy(-1.0..1.0; rng = rng),
		update_after = 1000,
		update_every = 1,
		act_limit = 1.0,
		act_noise = 0.1,
		rng = rng,
	),
	trajectory = CircularArraySARTTrajectory(
		capacity = 10000,
		state = Vector{Float32} => (ns,),
		action = Float32 => (),
	),
)

stop_condition = StopAfterStep(100_000)
total_reward_per_episode = TotalRewardPerEpisode()
time_per_step = TimePerStep()
hook = ComposedHook(
	total_reward_per_episode,
	time_per_step,
	DoEveryNStep() do t, agent, env
		with_logger(lg) do
			@info(
				"training",
				actor_loss = agent.policy.actor_loss,
				critic_loss = agent.policy.critic_loss
			)
		end
	end,
	DoEveryNEpisode() do t, agent, env
		with_logger(lg) do
			@info "training" reward = total_reward_per_episode.rewards[end] log_step_increment =
				0
		end
	end,
)

agent(inner_env)
env(2)
run(agent, inner_env, stop_condition, hook)
inner_env(1.0)

state(env)
agent(env)
env(-0.5)
action_space(env)
env.env.state[1]

env1 = TrebuchetEnv(continuous=true)
reset!(env1)
@show state(env1)
#set_goal!(env, 80)
rewards = [reward(env1)]
angles = [env1.state[1]]
while !is_terminated(env1)
	action = agent(env1)
	env1(action)
	r = reward(env1)
	@show action, r
	append!(rewards, r)
	#append!(angles, env.state[1])
end


##




function train_DDPG()
	seed = 123
	env = TrebuchetEnv()
	rng = StableRNG(seed)
	init = glorot_uniform(rng)

	explorer = EpsilonGreedyExplorer(0.1)
	agent = Agent(
		policy = QBasedPolicy(;
				learner=TDLearner(;
					approximator = TabularQApproximator(;
						n_state=16117580,
						n_action=4,
						init=-1e6,
						opt=InvDecay(1.0)
					),
					γ = 1.0,
					method=:SARSA,
					n = 0,
				),
				explorer = explorer
		),
		trajectory=VectorSARTTrajectory(;state=Int)
	)
	if isfile(output_models*"agent_Q.bson")
		@load output_models*"agent_Q.bson" table
		agent.policy.learner.approximator.table[:,:] = table[:,:]
	end
	reset!(E)
	## train the agent
	run(agent, E, StopAfterEpisode(1_000))
	@show maximum(table), minimum(table), count(table.>-1e6)
	## test the agent
	@show run(agent, E, StopAfterEpisode(10), StepsPerEpisode())
	table = agent.policy.learner.approximator.table
	@save output_models*"agent_Q.bson" table
end
