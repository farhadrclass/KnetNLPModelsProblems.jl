include("..//src//utils.jl")
include("/Users/nathanallaire/Desktop/GERAD/JSOSolvers.jl/src/R2.jl")
include("..//src//Lenet_mnist.jl")
include("..//src//FC_mnist.jl")

T = Float16

(xtrn, ytrn), (xtst, ytst) = loaddata(1, T)
dtrn = minibatch(xtrn, ytrn, 100; xtype = T, ytype = T, xsize = (size(xtrn, 1), size(xtrn, 2), 1, :))
dtst = minibatch(xtst, ytst, 100; xtype = T, ytype = T, xsize = (size(xtst, 1), size(xtst, 2), 1, :))


# size of minibatch 
m = 100

knetModel, myModel = lenet_prob(xtrn, ytrn, xtst, ytst, minibatchSize = m)

trained_model = train_knetNLPmodel!(myModel, R2, xtrn, ytrn; mbatch = m, mepoch = 1, maxTime = 100, all_data = true, verbose = false, epoch_verbose = true)

#training loop to go over all the dataset
# for i = 0:(length(ytrn)/m)-1
#     reset_minibatch_train!(myModel)
#     println("Minibatch =============================", Int64(i+1))
#     statsCNN = R2(myModel)
#     w = statsCNN.solution
#     myModel.w = w
#     println("Minibatch ", Int64(i+1), " Accuracy: ", KnetNLPModels.accuracy(myModel))
# end


res = trained_model[2]
epochs = res[:, 1]
acc = res[:, 2]

fig = plot(epochs, title="Best accuracy vs Epoch on Float16", acc, label="best accuracy", legend=:bottomright, xlabel = "epoch", ylabel = "accuracy")
include("..//src//utils.jl")
# include("/Users/nathanallaire/Desktop/GERAD/JSOSolvers.jl/src/R2.jl")
include("..//src//R2.jl")
include("..//src//Lenet_mnist.jl")
include("..//src//FC_mnist.jl")

using Knet:
    Knet,
    conv4,
    pool,
    mat,
    nll,
    accuracy,
    progress,
    sgd,
    param,
    param0,
    dropout,
    relu,
    minibatch,
    Data

T = Float32
Knet.atype() = Array{T}
(xtrn, ytrn), (xtst, ytst) = loaddata(1, T)

# size of minibatch 
m = 512
max_epochs = 50

knetModel, myModel = lenet_prob(xtrn, ytrn, xtst, ytst, minibatchSize = 10)

trained_model = train_knetNLPmodel!(
    myModel,
    R2,
    xtrn,
    ytrn;
    mbatch = 10,
    mepoch = max_epochs,
    maxTime = 100,
    all_data = false,
    verbose = true,
    epoch_verbose = true,
    gamma = 0.9
)
res = trained_model[2]
epochs = res[:, 1]
acc = res[:, 2]
train_acc = res[:, 3]



# # Train Knet
trained_model_knet = train_knet(knetModel, xtrn, ytrn, xtst, ytst;mbatch=m, mepoch = 5) #TODO some reason when mepoch=max_epochs, will give error , maybe Int(max_epochs)
res_knet = trained_model_knet[2]
epochs_knet = res_knet[:, 1]
acc_knet = res_knet[:, 2]
train_acc_knet = res_knet[:, 3]


fig = plot(
    epochs,
    # title = " test accuracy vs Epoch",
    markershape = :star4,
    acc,
    label = "test accuracy R2",
    legend = :bottomright,
    xlabel = "epoch",
    ylabel = "accuracy",
)
plot!(fig, epochs, markershape = :star1, acc_knet, label = "test accuracy SGD")

# plotSamples(myModel, xtrn, ytrn, MNIST; samples=10)

fig = plot(
    epochs,
    title = "train accuracy vs Epoch on Float32",
    markershape = :star4,
    train_acc,
    label = "train accuracy R2",
    legend = :bottomright,
    xlabel = "epoch",
    ylabel = "accuracy",
)
plot!(fig, epochs, markershape = :star1, train_acc_knet, label = "train accuracy SGD")


# plotSamples(myModel, xtrn, ytrn, MNIST; samples=10)


#Plot all

fig = plot(
    epochs,
    # title = " All accuracy vs Epoch on Float32",
    markershape = :star1,
    acc,
    label = "test accuracy R2",
    legend = :bottomright,
    xlabel = "epoch",
    ylabel = "accuracy",
)


plot!(
    fig,
    epochs,
    markershape = :star1,
    train_acc,
    label = "train accuracy R2",
    legend = :bottomright,
    linestyle = :dash,
)
plot!(fig, epochs, markershape = :star4, acc_knet, label = "test accuracy SGD")

plot!(
    fig,
    epochs,
    markershape = :star4,
    train_acc_knet,
    label = "train accuracy SGD",
    linestyle = :dot,
)