include("..//src//utils.jl")
include("/Users/nathanallaire/Desktop/GERAD/JSOSolvers.jl/src/R2.jl")
include("..//src//Lenet_mnist.jl")
include("..//src//FC_mnist.jl")


T = Float16
Knet.atype() = Array{T}
(xtrn, ytrn), (xtst, ytst) = loaddata(1, T)

# size of minibatch 
m = 100

knetModel, myModel = lenet_prob(xtrn, ytrn, xtst, ytst, minibatchSize = m)

trained_model = train_knetNLPmodel!(myModel, R2, xtrn, ytrn; mbatch = m, mepoch = 3, maxTime = 100, all_data = false, verbose = true, epoch_verbose = true)


res = trained_model[2]
epochs = res[:, 1]
acc = res[:, 2]

fig = plot(epochs, title="Best accuracy vs Epoch on Float32", acc, label="best accuracy", legend=:bottomright, xlabel = "epoch", ylabel = "accuracy")


plotSamples(myModel, xtrn, ytrn, MNIST; samples=10)