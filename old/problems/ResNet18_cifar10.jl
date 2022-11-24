# Install packages before first run: using Pkg; pkg"add Knet IterTools MLDatasets"
using Knet, IterTools, MLDatasets

# Define convolutional layer:
struct Conv
    w::Any
    b::Any
end
Conv(w1, w2, nx, ny) = Conv(param(w1, w2, nx, ny), param0(1, 1, ny, 1))
(c::Conv)(x) = relu.(pool(conv4(c.w, x) .+ c.b))

# Define dense layer:
struct Dense
    w::Any
    b::Any
    f::Any
end
Dense(i, o; f = identity) = Dense(param(o, i), param0(o), f)
(d::Dense)(x) = d.f.(d.w * mat(x) .+ d.b)

# Define a chain of layers and a loss function:
struct Chain
    layers::Any
end
(c::Chain)(x) = (for l in c.layers
    x = l(x)
end;
x)
(c::Chain)(x, y) = nll(c(x), y)

# Load MNIST data:
xtrn, ytrn = MNIST.traindata(Float32);
ytrn[ytrn.==0] .= 10;
xtst, ytst = MNIST.testdata(Float32);
ytst[ytst.==0] .= 10;
dtrn = minibatch(xtrn, ytrn, 100; xsize = (28, 28, 1, :))
dtst = minibatch(xtst, ytst, 100; xsize = (28, 28, 1, :))

# Define and train LeNet (~10 secs on a GPU or ~3 mins on a CPU to reach ~99% accuracy)
LeNet = Chain((
    Conv(5, 5, 1, 20),
    Conv(5, 5, 20, 50),
    Dense(800, 500, f = relu),
    Dense(500, 10),
))
progress!(adam(LeNet, ncycle(dtrn, 3)))
accuracy(LeNet, data = dtst)
