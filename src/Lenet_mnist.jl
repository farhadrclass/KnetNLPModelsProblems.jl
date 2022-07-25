"""
Lenet for  MNIST data
"""

include("utils.jl")

"""
Lenet model
Data need to be MNIST (#todo test with CIFAR10))
Returns Knet model and KnetNLPModel model
"""
function lenet_prob(xtrn, ytrn, xtst, ytst; minibatchSize = 100)

    LeNet = Chainnll(
        Conv(5, 5, 1, 20), # Conv(filter_length, filter_width, number of input channels from previous layer :n filters, 3 for RGB, 1 for Grey..., number of filters)
        Conv(5, 5, 20, 50),
        Dense(800, 500),
        Dense(500, 10, identity),
    )

    LeNetNLPModel = KnetNLPModel(
        LeNet;
        data_train = (xtrn, ytrn),
        data_test = (xtst, ytst),
        size_minibatch = minibatchSize,
    )


    return LeNet, LeNetNLPModel

end
