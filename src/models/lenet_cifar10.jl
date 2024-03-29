"""
lenet for CIFAR10 data
"""



"""
A NN model, close to LeNet
Data need to be CIFAr10
Returns Knet model and KnetNLPModel model
"""
function cifar10_prob(xtrn, ytrn, xtst, ytst; minibatchSize = 100)

    LeNet = Chainnll(
        Conv(5, 5, 3, 6), # Conv(filter_length, filter_width, number of input channels from previous layer :n filters, 3 for RGB, 1 for Grey..., number of filters)
        Conv(5, 5, 6, 16),
        Dense(400, 120),
        Dense(120, 84),
        Dense(84, 10, identity),
    )

    LeNetNLPModel = KnetNLPModel(
        LeNet;
        data_train = (xtrn, ytrn),
        data_test = (xtst, ytst),
        size_minibatch = minibatchSize,
    )


    return LeNet, LeNetNLPModel


end
