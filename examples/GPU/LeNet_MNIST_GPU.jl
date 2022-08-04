include("..//src//utils.jl")
include("/Users/nathanallaire/Desktop/GERAD/KnetNLPModelsProblems/GPU/R2.jl")
include("..//src//Lenet_mnist.jl")
include("..//src//FC_mnist.jl")



function train_gpu(;T = Float32,
    minibatch_size = 100, 
    max_epochs = 5, 
    solver = R2, 
    all_data_arg = false, 
    verbose_arg = true, 
    epoch_verbose_arg = true
    )

    # @eval Knet.atype() = Array{T}
    if epoch_verbose_arg
        @info("The type is ", T)
    end
    
    (xtrn, ytrn), (xtst, ytst) = loaddata(1, T)
    knetModel, myModel = lenet_prob(xtrn, ytrn, xtst, ytst, minibatchSize = minibatch_size)

    trained_model = train_knetNLPmodel!(
        myModel,
        solver,
        xtrn,
        ytrn;
        mbatch = minibatch_size,
        mepoch = max_epochs,
        maxTime = 100,
        all_data = all_data_arg,
        verbose = verbose_arg,
        epoch_verbose = epoch_verbose_arg
        )
    res = trained_model[2]
    epochs, acc, train_acc = res[:, 1], res[:, 2], res[:, 3]
    return epochs, acc, train_acc
end

train_gpu()