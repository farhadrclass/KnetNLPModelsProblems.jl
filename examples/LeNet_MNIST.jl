include("..//src//utils.jl")
include("/Users/Farhad/Documents/GitHub/JSOSolvers.jl/src/R2.jl")
include("..//src//Lenet_mnist.jl")
include("..//src//FC_mnist.jl")

using Knet: Knet, conv4, pool, mat, nll, accuracy, progress, sgd, param, param0, dropout, relu, minibatch, Data

T = Float32
Knet.atype() = Array{T}
(xtrn, ytrn), (xtst, ytst) = loaddata(1, T)

# size of minibatch 
m = 100

knetModel, myModel = lenet_prob(xtrn, ytrn, xtst, ytst, minibatchSize = m)

trained_model = train_knetNLPmodel!(myModel, R2, xtrn, ytrn; mbatch = m, mepoch = 3, maxTime = 100, all_data = false, verbose = true, epoch_verbose = true)
res = trained_model[2]
epochs = res[:, 1]
acc = res[:, 2]


# # Train Knet
# dtrn = minibatch(xtrn, ytrn, 100; xsize = (28,28,1,:))
# dtst = minibatch(xtst, ytst, 100; xsize = (28,28,1,:));
# knet_lenet = train_knet!("myKnet_lenet.jld2",knetModel,dtrn,dtst)

# plot([acc, trained_model[1,:], trained_model[2,:]],ylim=(0.0,0.1),
# labels=["NLP_r2" "trnCNN" "tstCNN"],xlabel="Epochs",ylabel="Loss")
# sgdopt(x,xtrn, ytrn;ep=1) = (Knet.seed!(1); m=knetModel(xtrn, ytrn); sgd!(m,repeat(dtrn,ep),lr=10^(x[1]-1)); m(dtrn))


fig = plot(epochs, title="Best accuracy vs Epoch on Float32", acc, label="best accuracy", legend=:bottomright, xlabel = "epoch", ylabel = "accuracy")

plotSamples(myModel, xtrn, ytrn, MNIST; samples=10)