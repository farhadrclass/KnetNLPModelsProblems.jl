include("..//src//utils.jl")
include("..//src//NN_CIFAR10.jl")
include("/Users/nathanallaire/Desktop/GERAD/JSOSolvers.jl/src/R2.jl")

T = Float32
Knet.atype() = Array{T}
(xtrn, ytrn), (xtst, ytst) = loaddata(2, T)

# size of minibatch 
m = 512

knetModel, myModel = cifar10_prob(xtrn, ytrn, xtst, ytst, minibatchSize = m)

plotSamples(myModel, xtrn, ytrn, CIFAR10; samples = 5)


trained_model = train_knetNLPmodel!(
    myModel,
    R2,
    xtrn,
    ytrn;
    mbatch = m,
    mepoch = 50,
    maxTime = 100,
    all_data = false,
    verbose = true,
    epoch_verbose = true,
)

res = trained_model[2]
epochs = res[:, 1]
acc = res[:, 2]

fig = plot(
    epochs,
    title = "Best accuracy vs Epoch on Float32",
    acc,
    label = "best accuracy",
    legend = :bottomright,
    xlabel = "epoch",
    ylabel = "accuracy",
)
