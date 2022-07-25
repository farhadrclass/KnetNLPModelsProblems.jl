include("..//src//utils.jl")

include("..//src//Lenet_mnist.jl")
include("..//src//FC_mnist.jl")


T = Float16

(xtrn, ytrn), (xtst, ytst) = loaddata(1)
dtrn = minibatch(xtrn, ytrn, 100; xtype = T, ytype = T, xsize = (size(xtrn, 1), size(xtrn, 2), 1, :))
dtst = minibatch(xtst, ytst, 100; xtype = T, ytype = T, xsize = (size(xtst, 1), size(xtst, 2), 1, :))


# size of minibatch 
m = 100

knetModel, myModel = lenet_prob(xtrn, ytrn, xtst, ytst, minibatchSize = m)

trained_model = train_knetNLPmodel!(myModel, R2, xtrn, ytrn; mbatch = m, mepoch = 15, maxTime = 100, all_data = true, verbose = false)



# statsCNN = R2(LeNetNLPModel)

# #training loop to go over all the dataset
# for i = 0:(length(ytrn)/m)-1
#     reset_minibatch_train!(LeNetNLPModel)
#     println("Minibatch =============================", i)
#     statsCNN = R2(LeNetNLPModel)
#     w = statsCNN.solution
#     LeNetNLPModel.w = w
#     println("Minibatch Accracy: ", KnetNLPModels.accuracy(LeNetNLPModel))
# end


res = trained_model[2]
epochs = res[:, 1]
acc = res[:, 2]

fig = plot(epochs, title="Best accuracy vs Epoch on Float16", acc, label="best accuracy", legend=:bottomright, xlabel = "epoch", ylabel = "accuracy")
