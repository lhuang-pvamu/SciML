"""
Create a symmetric network with odd/even symmetry embedded in the network


"""
#using DiffEqFlux, Flux

struct SymmetryNet{M, R, P, S<:AbstractArray, T<:AbstractVector}
    model::M
    re::R
    p::P
    W::S
    b::T
end

# sym= 0: even symmetry; 1: odd symmetry
# model is a regular Chain() in Flux
function SymmetryNet(model;sym=0)
    len = length(model.layers)
    p = Flux.params(model)
    s = size(p[len*2])[1]
    W = Flux.glorot_uniform(1,s)
    b = zeros(s)
    layers = model.layers
    re = nothing
    if sym == 0
        model1 = Chain(x->hcat(x,-x), layers..., x->(1/2 * sum(W * x))[1])
        #p = Flux.params(model1,W,b)
        _p, re = Flux.destructure(model1)
        _p = vcat(_p,W...)
    else
        model1 = Chain(x->hcat(x,-x), layers..., x->(1/2 * W * (x[:,1] - x[:,2]))[1])
        #p = Flux.params(model1,W)
        _p, re = Flux.destructure(model1)
        _p = vcat(_p,W...)
    end
    display(model1)
    return SymmetryNet(model1,re,_p,W,b)
end

function symSign(x)
    si=ones(size(x)[1]-1)
    si=vcat(si,-1)
    hcat(x,si.*x)
end

function SymmetryNet(model::FastChain; p = initial_params(model), sym=0)
    len = length(model.layers)
    p = initial_params(model)
    s = model.layers[len].initial_params.out
    W = Flux.glorot_uniform(s)
    b = zeros(s)
    re = nothing

    if sym == 0
        model1 = FastChain((x,p)->hcat(x,-x), model.layers..., (x,p)->(1/2 * (x[:,1] + x[:,2])))
        #p = Flux.params(model1,W,b)
        #_p  = vcat(initial_params(model1),W..)
        _p  = initial_params(model1)
    else
        model1 = FastChain((x,p)->hcat(x,-x), model.layers..., (x,p)->(1/2 * (x[:,1] - x[:,2])))
        #p = Flux.params(model1,W)
        #_p  = vcat(initial_params(model1),W...)
        _p  = initial_params(model1)
    end
    return SymmetryNet(model1,re,_p,W,b)
end

#initial_params(c::SymmetryNet) = vcat(initial_params.(c.model.layers)...)

Flux.trainable(snet::SymmetryNet) = (snet.p)

(snet::SymmetryNet)(x, p=snet.p) =
    if typeof(snet.model) <: FastChain
        snet.model(x, p)
    else
        snet.re(p)(x)
    end

iter = 0
cb = function () #callback function to observe training
  global iter+=1
  if iter%100==0
      display(loss())
  end
end

function test_symmetry(sym=0)
    opt=Flux.ADAM(0.01)
    m = Chain(
        Dense(1, 8, σ),
        Dense(8, 4)
        )
    data = Iterators.repeated((), 5000)
    snet = SymmetryNet(m,sym=sym)
    target(x) = sym==0 ? cos(x) : sin(x)
    loss() =  sum(abs2, snet(x) - target(x) for x in collect(-10:0.1:10))
    cb = function () #callback function to observe training
      global iter+=1
      if iter%100==0
          display(loss())
      end
    end
    Flux.train!(loss, params(snet.p), data, opt; cb=cb)
    x=collect(-10:0.1:10)
    plot(x,target.(x))
    plot!(x,snet.(x))
end

function test_FastChain_symmetry(sym=0)
    m = FastChain(
        FastDense(1, 8, σ),
        FastDense(8, 4)
        )
    snet = SymmetryNet(m,sym=sym)
    target(x) = sym==0 ? cos(x) : sin(x)
    loss(θ) =  sum(abs2, snet(x,θ) - target(x) for x in collect(-10:0.1:10))
    iter=0
    callback(θ,l) = begin
        global iter+=1
        if iter%100==0
            println(l)
        end
        false
    end

    res = DiffEqFlux.sciml_train(loss, snet.p, ADAM(0.01), cb=callback, maxiters = 5000)
    x=collect(-10:0.1:10)
    plot(x,target.(x))
    y = [ snet(x,res.minimizer) for x in collect(-10:0.1:10)]
    plot!(x,y)
end

"""
Testing
test_symmetry()
test_symmetry(1)
test_FastChain_symmetry()
test_FastChain_symmetry(1)


test_symmetry()
"""
w =  rand(1,4)
m = FastChain(
    FastDense(2, 8, σ),
    FastDense(8, 4),
    )

initial_params(m)
snet = SymmetryNet(m,sym=1)
snet([1,2])
snet([-1,-2])
snet.model
snet.model.layers[2]
x=hcat([1,2],-[1,2])

w = rand(8,2)
r1=w*x

w1 = rand(4,8)
r2 = w1*r1
r3 = r2[:,1] + r2[:,2]
w2 = rand(4)
w2.*r3

x=[1,2]
[1,-1].*x

w=ones(size(x))
w[end]=-1
w

symSign(x)
