"""
Fully connceted MNIST data
"""


include("utils.jl")

"""
Fully conncected with one hidden layers
Data need to be MNIST (#todo test with CIFAR10))
Returns Knet model and KnetNLPModel model
"""
function FC_small_mnist(xtrn, ytrn, xtst, ytst; minibatchSize = 100)

    DenseNet = Chainnll(Dense(784, 20), Dense(20, 10))

    DenseNetNLPModel = KnetNLPModel(
        DenseNet;
        size_minibatch = minibatchSize,
        data_train = (xtrn, ytrn),
        data_test = (xtst, ytst),
    ) # define the KnetNLPModel

    return DenseNet, DenseNetNLPModel

end



"""
Fully conncected with 2s hidden layers
Data need to be MNIST (#todo test with CIFAR10))
Returns Knet model and KnetNLPModel model
"""
function FC_2h(xtrn, ytrn, xtst, ytst; minibatchSize = 100)

    DenseNet = Chainnll(Dense(784, 500), Dense(500, 200), Dense(200, 100), Dense(100, 10))

    DenseNetNLPModel = KnetNLPModel(
        DenseNet;
        size_minibatch = minibatchSize,
        data_train = (xtrn, ytrn),
        data_test = (xtst, ytst),
    ) # define the KnetNLPModel

    return DenseNet, DenseNetNLPModel

end