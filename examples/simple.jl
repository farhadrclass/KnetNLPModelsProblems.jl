include("..//src//utils.jl")

include("..//src//Lenet_mnist.jl")
include("..//src//FC_mnist.jl")



(xtrn, ytrn), (xtst, ytst) = loaddata(1)
dtrn = minibatch(xtrn, ytrn, 100; xsize = (size(xtrn, 1), size(xtrn, 2), 1, :))
dtst = minibatch(xtst, ytst, 100; xsize = (size(xtst, 1), size(xtst, 2), 1, :))

# size of minibatch 
m = 100


knetModel, myModel = lenet_prob(xtrn, ytrn, xtst, ytst, minibatchSize = m)




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


