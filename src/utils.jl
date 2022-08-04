# using Revise        # importantly, this must come before
using JSOSolvers
using LinearAlgebra
using Random
using Printf
using NLPModels
using SolverCore
using Plots
using Knet, Images, MLDatasets
using StochasticRounding
using KnetNLPModels
using Knet:
    Knet,
    conv4,
    pool,
    mat,
    nll,
    accuracy,
    progress,
    sgd,
    param,
    param0,
    dropout,
    relu,
    minibatch,
    Data

include("struct_utils.jl")
function loaddata(data_flag, T)
    if (data_flag == 1)
        @info("Loading MNIST...")
        xtrn, ytrn = MNIST.traindata(T)
        ytrn[ytrn.==0] .= 10
        xtst, ytst = MNIST.testdata(T)
        ytst[ytst.==0] .= 10
        @info("Loaded MNIST")
        # else if # MNIST FASHION
    else # CIFAR data
        @info("Loading CIFAR 10...")
        xtrn, ytrn = CIFAR10.traindata(T)
        xtst, ytst = CIFAR10.testdata(T)
        xtrn = convert(Knet.array_type[], xtrn)
        xtst = convert(Knet.array_type[], xtst)
        #= Subtract mean of each feature
        where each channel is considered as
        a single feature following the CNN
        convention=#
        mn = mean(xtrn, dims = (1, 2, 4))
        xtrn = xtrn .- mn
        xtst = xtst .- mn
        @info("Loaded CIFAR 10")
    end
    return (xtrn, ytrn), (xtst, ytst)
end

"""
    All_accuracy(nlp::AbstractKnetNLPModel)
Compute the accuracy of the network `nlp.chain` given the data in `nlp.tests`.
uses the whole test data sets"""
All_accuracy(nlp::AbstractKnetNLPModel) = Knet.accuracy(nlp.chain; data = nlp.data_test)

#runs over only one random one
function epoch!(
    modelNLP,
    solver,
    xtrn,
    ytrn,
    iter;
    verbose = true,
    epoch_verbose = true,
    mbatch = 64,
)
    reset_minibatch_train!(modelNLP)
    stats = solver(modelNLP; atol = 0.09, rtol =0.09,verbose = verbose)
    new_w = stats.solution
    set_vars!(modelNLP, new_w)
    if epoch_verbose
        @info("Epoch # ", iter)
        # @info("Minibatch accuracy: ", KnetNLPModels.accuracy(modelNLP)) # this takes too long 
    end
    return KnetNLPModels.accuracy(modelNLP)
end

#run over all minibatched 
function epoch_all!(
    modelNLP,
    solver,
    xtrn,
    ytrn,
    epoch;
    verbose = true,
    epoch_verbose = true,
    mbatch = 64,
)
    #training loop to go over all the dataset
    @info("Epoch # ", epoch)
    for i = 0:(length(ytrn)/m)-1
        reset_minibatch_train!(modelNLP)
        stats = solver(modelNLP; atol = 0.05, rtol =0.09, verbose = verbose)
        new_w = stats.solution
        set_vars!(modelNLP, new_w)
        if epoch_verbose
            @info("Minibatch = ", i)
            # @info("Minibatch accuracy: ", KnetNLPModels.accuracy(modelNLP)) # this takes too long
        end
    end
    return KnetNLPModels.accuracy(modelNLP)
end

function train_knetNLPmodel!(
    modelNLP,
    solver,
    xtrn,
    ytrn;
    mbatch = 64,
    mepoch = 10,
    maxTime = 100,
    all_data = false,
    verbose = true,
    epoch_verbose = true,
)

    acc_arr = []
    train_acc_arr = []
    iter_arr = []
    best_acc = 0
    for j = 1:mepoch
        if all_data
            acc =
                epoch_all!(modelNLP, solver, xtrn, ytrn, j; verbose, epoch_verbose, mbatch)
        else
            acc = epoch!(modelNLP, solver, xtrn, ytrn, j; verbose, epoch_verbose, mbatch)
        end

        if acc > best_acc
            #TODO write to file, KnetNLPModel, w
            best_acc = acc
        end
        # train accracy
        data_buff = create_minibatch(
            modelNLP.current_minibatch_training[1],
            modelNLP.current_minibatch_training[2],
            mbatch,
        ) # we need to create iterator it has one item
        train_acc = Knet.accuracy(modelNLP.chain; data = data_buff)
        append!(train_acc_arr, train_acc) #TODO fix this to save the acc


        append!(acc_arr, acc) #TODO fix this to save the acc
        append!(iter_arr, j)

        if j % 2 == 0
            @info("epoch #", j, "  acc= ", train_acc)
        end
        # add!(acc_arr, (j, acc))
        #TODO wirte to file 
        # if acc > best_acc
        #     #TODO write to file, KnetNLPModel, w
        #     best_acc = acc
        # end
    end

    c = hcat(iter_arr, acc_arr, train_acc_arr)
    #after each epoch if the accuracy better, stop 
    # all_data =true, go over them all, false only select minibatch
    return best_acc, c
end


"""This function will plots some samples and predicted vs true tags
"""
function plotSamples(myModel, xtrn, ytrn, data_set; samples = 5)
    rp = randperm(10000)
    x = [xtrn[:, :, :, rp[i]] for i = 1:samples]
    A = cat(x..., dims = 4)
    buff = myModel.chain(A)
    pred_y = findmax.(eachcol(buff))
    imgs = [data_set.convert2image(xtrn[:, :, :, rp[i]]) for i = 1:samples]

    p = plot(layout = (1, samples)) # Step 1
    i = 1
    for item in imgs
        plot!(
            p[i],
            item,
            ticks = false,
            title = string("T:", ytrn[rp[i]], "\nP:", pred_y[i][2]),
        )
        i = i + 1
    end
    display(p)
end


###############
#Knet

# function Knet_helper(knetModel)
#     loss(x, y) = knetModel(x, y)
#     return loss
# end

create_minibatch_iter(x_data, y_data, minibatch_size) = Knet.minibatch(
    x_data,
    y_data,
    minibatch_size;
    xsize = (size(x_data, 1), size(x_data, 2), 1, :),
)

function train_knet(
    knetModel,
    xtrn,
    ytrn,
    xtst,
    ytst;
    opt = sgd,
    mbatch = 100,
    lr = 0.1,
    mepoch = 5,
    iters = 1800,
)
    dtrn = minibatch(xtrn, ytrn, mbatch; xsize = (28, 28, 1, :)) #TODO the dimention fix this, Nathan fixed that for CIFAR-10
    test_minibatch_iterator = create_minibatch_iter(xtst, ytst, mbatch) # this is only use so our accracy can be compared with KnetNLPModel, since thier accracy use this
    acc_arr = []
    iter_arr = []
    train_acc_arr = []
    best_acc = 0
    for j = 1:mepoch
        progress!(opt(knetModel, dtrn, lr = lr)) #selected optimizer, train one epoch
        acc = Knet.accuracy(knetModel; data = test_minibatch_iterator)


        train_acc = Knet.accuracy(knetModel; data = dtrn)
        append!(train_acc_arr, train_acc)


        if acc > best_acc
            #TODO write to file, KnetNLPModel, w
            best_acc = acc
        end

        append!(acc_arr, acc) #TODO fix this to save the acc
        append!(iter_arr, j)

        if j % 2 == 0
            @info("epoch #", j, "  acc= ", train_acc)
        end
        # add!(acc_arr, (j, acc))
        #TODO wirte to file 
        # if acc > best_acc
        #     #TODO write to file, KnetNLPModel, w
        #     best_acc = acc
        # end
    end
    c = hcat(iter_arr, acc_arr, train_acc_arr)
    #after each epoch if the accuracy better, stop 
    return best_acc, c
end