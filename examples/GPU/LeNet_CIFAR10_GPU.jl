include("../../src/utils.jl")
include("R2.jl")
include("../../src/NN_CIFAR10.jl")

function train_gpu_c10(;T = Float32,
    minibatch_size = 100, 
    max_epochs = 5, 
    max_iter = -1,
    solver = R2, 
    all_data_arg = false, 
    verbose_arg = true, 
    epoch_verbose_arg = true
    )

    if CUDA.functional() 
        Knet.array_type[] = CUDA.CuArray{T}
    else 
        Knet.array_type[] = Array{T}
    end
    
    if epoch_verbose_arg
        @info("The type is ", T)
    end
    
    (xtrn, ytrn), (xtst, ytst) = loaddata(2, T)
    knetModel, myModel = cifar10_prob(xtrn, ytrn, xtst, ytst, minibatchSize = minibatch_size)

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
        max_iter = max_iter,
        epoch_verbose = epoch_verbose_arg
        )
    res = trained_model[2]
    epochs, acc, train_acc = res[:, 1], res[:, 2], res[:, 3]
    return epochs, acc, train_acc
end