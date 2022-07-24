include("..//src//utils.jl")

include("..//src//Lenet_mnist.jl")
include("..//src//FC_mnist.jl")
include("/Users/nathanallaire/Desktop/GERAD/JSOSolvers.jl/src/R2.jl")

using StochasticRounding
T = Float32sr

(xtrn, ytrn), (xtst, ytst) = loaddata(1, T)

dtrn = minibatch(xtrn, ytrn, 100; xtype = Float32sr, ytype = Float32sr, xsize = (size(xtrn, 1), size(xtrn, 2), 1, :))
dtst = minibatch(xtst, ytst, 100; xtype = Float32sr, ytype = Float32sr, xsize = (size(xtst, 1), size(xtst, 2), 1, :))

# size of minibatch 
m = 100

knetModel, myModel = lenet_prob(xtrn, ytrn, xtst, ytst, minibatchSize = m)


# statsCNN = R2(myModel)
# print(statsCNN)
# w = statsCNN.solution
# myModel.w = w

trained_model = train_knetNLPmodel!(myModel, R2, xtrn, ytrn; mbatch = m, mepoch = 20, maxTime = 100, all_data = false)



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

fig = plot(epochs, title="Best accuracy vs Epoch", acc, label="best accuracy", legend=:bottomright, xlabel = "epoch", ylabel = "accuracy")