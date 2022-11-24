include("../../src/utils.jl")
include("../../src/FC_mnist.jl")


function train_gpu_fc_knet(;
    T = Float32,
    minibatch_size = 100,
    max_epochs = 5,
    max_iter = -1,
    epoch_verbose_arg = true,
)

    if CUDA.functional()
        Knet.array_type[] = CUDA.CuArray{T}
    else
        Knet.array_type[] = Array{T}
    end

    if epoch_verbose_arg
        @info("The type is ", T)
    end

    (xtrn, ytrn), (xtst, ytst) = loaddata(1, T)
    knetModel, myModel = FC_2h(xtrn, ytrn, xtst, ytst; minibatchSize = minibatch_size)

    trained_model = train_knet(
        knetModel,
        xtrn,
        ytrn,
        xtst,
        ytst;
        opt = sgd,
        mbatch = minibatch_size,
        lr = 0.1,
        mepoch = max_epochs,
        iters = max_iter,
    )

    res = trained_model[2]
    epochs, acc, train_acc = res[:, 1], res[:, 2], res[:, 3]
    return epochs, acc, train_acc
end
