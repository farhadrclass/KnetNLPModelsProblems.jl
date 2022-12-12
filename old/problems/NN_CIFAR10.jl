include("..//src//utils.jl")
include("..//src//NN_CIFAR10.jl")
include("/Users/nathanallaire/Desktop/GERAD/JSOSolvers.jl/src/R2.jl")

T = Float32

(xtrn, ytrn), (xtst, ytst) = loaddata(2, T)

dtrn = minibatch(xtrn, ytrn, 100; xsize = (size(xtrn, 1), size(xtrn, 2), 1, :))
dtst = minibatch(xtst, ytst, 100; xsize = (size(xtst, 1), size(xtst, 2), 1, :))

m = 100

knetModel, myModel = nn_prob(xtrn, ytrn, xtst, ytst, minibatchSize = m)

trained_model = train_knetNLPmodel!(
    myModel,
    R2,
    xtrn,
    ytrn;
    mbatch = m,
    mepoch = 1,
    maxTime = 100,
    all_data = false,
    verbose = false,
)
