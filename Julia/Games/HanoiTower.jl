
num_disks = 8
all_disks = 1:num_disks
first_stack = collect(all_disks)
starting_stacks = [first_stack, [], []]
display(starting_stacks)

moves = []
function hanoi(n, stacks, source::Int, helper::Int, target::Int)
	if n>0
		hanoi(n-1, stacks, source, target, helper)
		if !isempty(stacks[source])
			disk = popfirst!(stacks[source])
			pushfirst!(stacks[target], disk)
			push!(moves, (source,target))
		end
		hanoi(n-1, stacks, helper, source, target)
	end
end

new_stacks = deepcopy(starting_stacks)
hanoi(num_disks, new_stacks, 1, 2, 3)
display(new_stacks)
display(moves)

function islegal(stacks)
	order_correct = all(issorted, stacks)

	#check if we use the same disk set that we started with

	disks_in_state = sort([disk for stack in stacks for disk in stack])
	disks_complete = disks_in_state == all_disks

	order_correct && disks_complete
end

function iscomplete(stacks)
	last(stacks) == all_disks
end

function move(stacks, source::Int, target::Int)
	#check if the from stack if not empty
	if isempty(stacks[source])
		error("Error: attempted to move disk from empty stack")
	end

	new_stacks = deepcopy(stacks)

	disk = popfirst!(new_stacks[source]) #take disk
	pushfirst!(new_stacks[target], disk) #put on new stack

	return new_stacks
end

function eval(stacks)
	if isempty(stacks[3])
		0
	else
		sum(stacks[3])
	end
end

function solve(stacks)::Array{Tuple{Int, Int}}
	sol = []
	moves = [(1,2),(1,3),(2,3),(3,2),(2,1),(3,1)]
	print("Start")
	checkpoint = deepcopy(stacks)
	checkpoint_sol = sol
	points = eval(checkpoint)
	num_steps = 0
	#what to do?
	while !iscomplete(stacks)
        new_mv = true
		idx = rand(1:6)
		if isempty(stacks[moves[idx][1]])
			continue
		end
		new_stacks = move(stacks,moves[idx][1],moves[idx][2])
		if islegal(new_stacks)
			stacks = new_stacks
			push!(sol,moves[idx])
			#display(stacks)
			new_points = eval(stacks)
			if new_points > points
				points = new_points
				checkpoint = deepcopy(stacks)
				checkpoint_sol = deepcopy(sol)
			end
            new_mv = true
		end
		num_steps = num_steps + 1
        if num_steps>50000
            print("too many moves: ", points)
            stacks = deepcopy(checkpoint)
			sol = deepcopy(checkpoint_sol)
			display(checkpoint)
			num_steps=0
        end
	end
	return sol
end

function run_solution(solver::Function, start = starting_stacks)
	moves = solver(deepcopy(start)) #apply the solver

	all_states = Array{Any,1}(undef, length(moves) + 1)
	all_states[1] = starting_stacks

	for (i, m) in enumerate(moves)
		try
			all_states[i + 1] = move(all_states[i], m[1], m[2])
		catch
			all_states[i + 1] = missing
		end
	end

	return all_states
end

moves = solve(deepcopy(starting_stacks))

run_solution(solve)

function check_solution(solver::Function, start = starting_stacks)
	try
		#run the solution
		all_states = run_solution(solver, start)

		#check if each state is legal
		all_legal = all(islegal, all_states)

		#check if the final state is is the completed puzzle
		complete = (iscomplete ∘ last)(all_states)

		all_legal && complete
	catch
		#return false if we encountered an error
		return false
	end
end

check_solution(solve)
