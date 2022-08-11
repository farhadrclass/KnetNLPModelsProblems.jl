
using CUDA, IterTools, Knet, MLDatasets, NLPModels
using KnetNLPModels
using Statistics: mean
using Knet
struct Conv
    w::Any
    b::Any
    f::Any
end
(c::Conv)(x) = c.f.(pool(conv4(c.w, x) .+ c.b))
Conv(w1, w2, cx, cy, f = relu) = Conv(param(w1, w2, cx, cy), param0(1, 1, cy, 1), f)

struct Dense
    w::Any
    b::Any
    f::Any
    p::Any
end
(d::Dense)(x) = d.f.(d.w * mat(dropout(x, d.p)) .+ d.b) # todo change * in future 
Dense(i::Int, o::Int, f = sigm; pdrop = 0.0) = Dense(param(o, i), param0(o), f, pdrop)

struct Chainnll <: KnetNLPModels.Chain
    layers::Any
    Chainnll(layers...) = new(layers)
end
(c::Chainnll)(x) = (for l in c.layers
    x = l(x)
end;
x)
(c::Chainnll)(x, y) = Knet.nll(c(x), y)  # nécessaire
(c::Chainnll)(data::Tuple{T1,T2}) where {T1,T2} = c(first(data, 2)...)
(c::Chainnll)(d::Knet.Data) = Knet.nll(c; data = d, average = true)
