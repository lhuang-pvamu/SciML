#cd(@__DIR__)
#using Pkg;
#Pkg.activate("..");
#Pkg.instantiate();

SOURCE = 1
TARGET = 2
RESTLEN = 3
STIFFNESS = 4
ACTUATION = 5

function get_distance(a,b)
    sqrt(sum((a.-b).^2))
end

function compute_center(bolts)
    c = sum(bolts,dims=2)
    (1.0f0/length(bolts[1,:])) .* c
end

function add_spring(springs, bolts, a, b, stiffness=10.0, actuation=0.1)
    index = findall(x->x[1:2]==[a,b] || x[1:2]==[b,a],springs)
    if length(index) == 0
        restLen = sqrt(sum((bolts[a]-bolts[b]).^2))
        append!(springs, [[a,b,restLen,stiffness,actuation]])
    end
end

function add_bolt(bolts,i,j)
    len = 0.05
    i = i*len+0.1
    j = j*len
    index = findall(x->x==[i,j],bolts)
    if length(index)==0
        append!(bolts,[[i,j]])
    end
    findall(x->x==[i,j],bolts)[1]
end

function add_mesh_triangle(bolts,springs,i,j,stiffness=3f4,actuation=0.1)
    a = add_bolt(bolts,i+0.5,j)
    b = add_bolt(bolts,i,j+1)
    c = add_bolt(bolts,i+1,j+1)
    for i in [a,b,c]
        for j in [a,b,c]
            if i !=j
                add_spring(springs,bolts,i,j,stiffness,actuation)
            end
        end
    end
end

function add_mesh_triangle_right(bolts,springs,i,j,stiffness=3f4,actuation=0.1)
    a = add_bolt(bolts,i,j)
    b = add_bolt(bolts,i,j+1)
    c = add_bolt(bolts,i+1,j+1)
    for i in [a,b,c]
        for j in [a,b,c]
            if i !=j
                add_spring(springs,bolts,i,j,stiffness,actuation)
            end
        end
    end
end

function add_mesh_square(bolts, springs, i, j, stiffness=3f4, actuation=0.1)

    a = add_bolt(bolts,i,j)
    b = add_bolt(bolts,i,j+1)
    c = add_bolt(bolts,i+1,j)
    d = add_bolt(bolts,i+1,j+1)

    add_spring(springs,bolts,a,b,stiffness,actuation)
    add_spring(springs,bolts,c,d,stiffness,actuation)

    for i in [a,b,c,d]
        for j in [a,b,c,d]
            if i !=j
                add_spring(springs,bolts,i,j,stiffness,0.0)
            end
        end
    end
end


bolts = []
springs = []
index = add_bolt(bolts,1,2)
bolts

add_mesh_square(bolts,springs,0,0)
bolts
springs

index = findall(x->x[1:2]==[1,2] || x[1:2]==[2,1],springs)
index

function setup_robotA()
    bolts=[]
    springs = []
    connections = []
    append!(bolts, [[0.2, 0.0]])
    append!(bolts, [[0.3, 0.1]])
    append!(bolts, [[0.4, 0.0]])
    append!(bolts, [[0.2, 0.2]])
    append!(bolts, [[0.3, 0.2]])
    append!(bolts, [[0.4, 0.2]])
    for i in 1:length(bolts)
        append!(connections, [[]])
    end
    #print(length(connections))

    stiffness=1.4e4
    actuation=0.1
    function link(i,j,stiffness=1.4e4)
        restLen = sqrt(sum((bolts[i]-bolts[j]).^2))
        append!(springs, [[i,j,restLen,stiffness,actuation]])
        id = length(springs)
        append!(connections[i],id)
        append!(connections[j],id)
    end
    link(1, 2)
    link(2, 3)
    link(4, 5)
    link(5, 6)
    link(1, 4)
    link(3, 6)
    link(1, 5)
    link(2, 5)
    link(3, 5)
    link(4, 2)
    link(6, 2)
    bolts = hcat(bolts...) .|> Float32 # 2D array
    springs = hcat(springs...) .|> Float32 # 2D array
    connections = hcat(connections)  # an array of array due to the variance of length
    bolts, springs, connections
end


function setup_robotB()
    bolts = []
    springs = []
    add_mesh_triangle(bolts,springs,0,0,3f4,0f0)
    add_mesh_triangle(bolts,springs,2,0,3f4,0f0)
    add_mesh_square(bolts,springs,0,1,3f4,0.1f0)
    add_mesh_square(bolts,springs,0,2)
    add_mesh_square(bolts,springs,1,2)
    add_mesh_square(bolts,springs,2,1,3f4,0.1f0)
    add_mesh_square(bolts,springs,2,2)
    return hcat((bolts)...).|> Float32,hcat(springs...) .|> Float32
end

function setup_robotC()
    bolts = []
    springs = []
    add_mesh_square(bolts,springs,0,0,3f4,0.25f0)
    add_mesh_square(bolts,springs,0,1,3f4,0.25f0)
    add_mesh_square(bolts,springs,0,2)
    add_mesh_square(bolts,springs,1,2)
    add_mesh_square(bolts,springs,2,2)
    add_mesh_square(bolts,springs,3,0,3f4,0.25f0)
    add_mesh_square(bolts,springs,3,1,3f4,0.25f0)
    add_mesh_square(bolts,springs,3,2)
    add_mesh_square(bolts,springs,3,3)
    add_mesh_square(bolts,springs,3,4)
    add_mesh_triangle_right(bolts,springs,4,3)
    return hcat((bolts)...).|> Float32,hcat(springs...) .|> Float32
end

function visualize_robot(bolts,springs)
    pl = plot([(0,0),(1,0)], xlims=(0,1), ylims=(0,1), legend=false)
    for i in 1:length(springs[1,:])
        cord =hcat(bolts[:,Int(springs[1,i])],bolts[:,Int(springs[2,i])])
        plot!(pl, cord[1,:],cord[2,:], marker = :hex, legend=false)
    end
    display(pl)
end


function anim_robot(data,springs;step=10)
    anim = @animate for i ∈ 1:step:length(data[1,1,:])
        plot([(0,0),(1,0)], xlims=(0,1), ylims=(0,1), legend=false)
        for j in 1:length(springs[1,:])
            cord =hcat(data[:,Int(springs[1,j]),i],data[:,Int(springs[2,j]),i])
            plot!(cord[1,:],cord[2,:], marker = :hex, legend=false)
        end
    end
    anim
end

bolts, springs = setup_robotA()
visualize_robot(bolts,springs)
compute_center(bolts)
