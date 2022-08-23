include("../../src/utils.jl")
include("../../src/R2.jl")
include("../../src/FC_mnist.jl")
include("../../src/small_FC_mnist.jl")

function train_fc_small(;T = Float32,
    minibatch_size = 100, 
    max_epochs = 5, 
    max_iter = -1,
    solver = R2, 
    all_data_arg = false, 
    verbose_arg = true, 
    epoch_verbose_arg = true,
    gamma = 0.9
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
    knetModel, myModel = FC_small_mnist(xtrn, ytrn, xtst, ytst; minibatchSize = minibatch_size)
    

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
        epoch_verbose = epoch_verbose_arg,
        gamma = gamma
        )
    res = trained_model[2]
    epochs, acc, train_acc = res[:, 1], res[:, 2], res[:, 3]
    return epochs, acc, train_acc
end


fig = plot(
    g0[1], 
    g0[2], 
    label = "max_iter = 1 - test",
    color = "darkred",
    markershape = :circle,
    lw = 2,
    thickness_scaling = 1.3,
    legend = :topleft,
    size=(1000, 500),
    title = "Accuracy of MNIST dataset - γ = 0.9",
    xlabel = "epoch",
    ylabel = "accuracy on test dataset"
)

plot!(
    fig,
    g0[1], 
    g0[3], 
    label = "max_iter = 1 - train",
    color = "tomato",
    markershape = :diamond,
    lw = 2,
    thickness_scaling = 1.3,
    legend = :bottomright,
)

plot!(
    fig,
    g0[1],
    g0_9[2],
    label = "max_iter = 10 - test",
    lw = 2,
    color = "blue4",
    markershape = :star4,
)

plot!(
    fig,
    g0[1],
    g0_9[3],
    label = "max_iter = 10 - train",
    color = "dodgerblue",
    markershape = :xcross,
)