include("../../src/utils.jl")
include("R2.jl")
include("../../src/FC_mnist.jl")
include("../../src/small_FC_mnist.jl")

function train_fc_small(;T = Float32,
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
        epoch_verbose = epoch_verbose_arg
        )
    res = trained_model[2]
    epochs, acc, train_acc = res[:, 1], res[:, 2], res[:, 3]
    return epochs, acc, train_acc
end


fig = plot(
    f16_rn[1], 
    f16_sr[2], 
    label = "float16 - stochastic rounding",
    color = "darkred",
    markershape = :circle,
    lw = 2,
    thickness_scaling = 1.3,
    legend = :topleft,
    size=(1000, 500),
    title = "Accuracy of MNIST dataset - CPU",
    xlabel = "epoch",
    ylabel = "accuracy on test dataset"
)

plot!(
    fig,
    f16_rn[1], 
    f16_rn[2], 
    label = "float16 - round to nearest",
    color = "tomato",
    markershape = :diamond,
    lw = 2,
    thickness_scaling = 1.3,
    legend = :bottomright,
)

plot!(
    fig,
    f16_rn[1],
    f32_sr[2],
    label = "float32 - stochastic rounding",
    lw = 2,
    color = "blue4",
    markershape = :star4,
)
plot!(
    fig,
    f16_rn[1],
    f32_rn[2],
    label = "float32 - round to nearest",
    color = "dodgerblue",
    markershape = :xcross,
)