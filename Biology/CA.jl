cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()
#Pkg.precompile()

using Flux
using Plots
gr()
using Images
using Zygote
#using ImageView

CHANNEL_N = 16
GRID_X = 40
GRID_Y = 40
BATCH_SIZE = 1

A = zeros(GRID_X,GRID_Y,CHANNEL_N,BATCH_SIZE)
A[30:40,30:40,:,1] .= 1

heatmap(A[:,:,1,1])
filter = ones(3,3,1,16)
filter[2,2,1,:] .= 0
B = conv(A, filter, pad=(1,1), groups=16)

layer = Conv((3,3),16=>16)
layer(A)

for i=1:35
    B = conv(A, filter, pad=(1,1), groups=16)
    A[B.>3] .= 0
    A[B.<2] .= 0
    A[B.==3] .= 1
    display(heatmap(A[:,:,4,1]))
    sleep(0.5)
end

d = Dense(16, 128)
c = permutedims(A,[3,2,1,4])
d(c)
heatmap(A[:,:,1,1])
heatmap(B[:,:,1,1])

model = Chain(
    Conv((3,3), 1=>1)
    )

model(A)


sobel_x = [
    -1 0 1
    -2 0 2
    -1 0 1
]
sobel_y = transpose(sobel_x)
sobel_x = reshape(sobel_x,3,3,1,1)
sobel_y = reshape(sobel_y,3,3,1,1)
sx = Float64.(repeat(sobel_x, 1,1,1,16))
filter
conv(A, sx, pad=(1,1),groups=16)

grad_x = conv(A, sobel_x, pad=(1,1))
grad_y = conv(A, sobel_y, pad=(1,1))
heatmap(grad_x[:,:,1,1])
heatmap(grad_y[:,:,1,1])
vcat(grad_x, grad_y, A)

function perceive(state)
    sobel_x = [
        -1.0 0.0 1.0
        -2.0 0.0 2.0
        -1.0 0.0 1.0
    ]
    sobel_y = transpose(sobel_x)
    sobel_x = repeat(reshape(sobel_x,3,3,1,1),1,1,1,CHANNEL_N)
    sobel_y = repeat(reshape(sobel_y,3,3,1,1),1,1,1,CHANNEL_N)
    #state = permutedims(state, [3,2,1,4])
    grad_x = conv(state, sobel_x, pad=(1,1),groups=CHANNEL_N)
    grad_y = conv(state, sobel_y, pad=(1,1),groups=CHANNEL_N)
    cat(grad_x, grad_y, state, dims=3)
end

B = perceive(A)

size(B)
function build_model(channel_n)
    return Chain(
        Dense(channel_n*3, 128, relu),
        Dense(128, channel_n) #, σ)
    )
end

model = build_model(16)
ps = Flux.params(model)
opt = ADAM(0.00001)

C = permutedims(B,[3,2,1,4])
model(C)

function update(state)
    return model(state)
end

c = rand(64,64)
mask = zeros(64,64)
mask[c.<0.5].=1.0
mask
rand_mask = rand(64,64).<0.5

function stochastic_update(state, ds)
    rand_mask = rand(GRID_X,GRID_Y).<0.5
    #buf = Zygote.Buffer(rand_mask)
    #buf = rand_mask
    #buf[rand(64,64).<0.5].=0.0
    #rand_mask[rand(64,64).<0.5].=0.0
    #rand_mask = copy(buf)
    ds = ds .* rand_mask
    return state+ds
end

function alive_mask(state)
    neighors = MaxPool((3,3), pad=(1,1), stride=(1,1))(state[:,:,4:4,:])
    alive = neighors.>0.1
    #buf = Zygote.Buffer(alive)
    #buf = alive
    #buf[buf.<=0.1] .= 0.0
    #buf[buf.>0.1] .= 1.0
    #alive = copy(buf)
    return state .* alive
end

neighors = MaxPool((3,3), pad=(1,1), stride=(1,1))(A[:,:,4:4,:])
alive = neighors.>0.1
heatmap(alive[:,:,1,1])
A .* alive
B = alive_mask(A)
heatmap(B[:,:,5,1])

heatmap(A[:,:,4,1])

function forward(state; steps=1)
    for i in 1:steps
        state1 = perceive(state)
        state2 = permutedims(state1,[3,2,1,4])
        ds = model(state2) / 10.0
        ds = permutedims(ds,[3,2,1,4])
        #state = state + ds
        state = stochastic_update(state, ds)
        state = alive_mask(state)
    end
    return state
end

A = zeros(GRID_X,GRID_Y,CHANNEL_N,BATCH_SIZE)
A[Int(GRID_X/2),Int(GRID_Y/2),4:end,1] .= 1
heatmap(A[:,:,4,1])
state = forward(A, steps=100)
heatmap(state[:,:,1,1])

function loss(state, target)
    sum(abs2, state[:,:,1:3,1] .- target)
end


img = load("./Data/emoji.png")
#img = imresize(img, (64,64))
imgc = channelview(img)

mat = permutedims(convert(Array{Float64}, imgc), (2,3,1))
maximum(mat)
target = mat[1:40, 1:40, 1:3]

loss(state, target[:,:,1:3])

for epoch in 1:1000
    gs = Flux.gradient(ps) do
        state = forward(A, steps=70)
        l = loss(state, target)
        println(l)
        l
    end
    Flux.Optimise.update!(opt, ps, gs)
    #display(heatmap(state[:,:,1,1]))
end

state = forward(A, steps=70)
heatmap(state[:,:,1,1])

#imshow(target)
heatmap(target[:,:,1])
