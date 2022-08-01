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
function epoch!(modelNLP, solver, xtrn, ytrn, iter; verbose = true, epoch_verbose = true, mbatch = 64)
    reset_minibatch_train!(modelNLP)
    stats = solver(modelNLP; verbose=verbose)
    new_w = stats.solution
    set_vars!(modelNLP, new_w)
    if epoch_verbose
        @info("Epoch # ", iter)
        @info("Minibatch accuracy: ", KnetNLPModels.accuracy(modelNLP))
    end
    return KnetNLPModels.accuracy(modelNLP)
end

#run over all minibatched 
function epoch_all!(modelNLP, solver, xtrn, ytrn, epoch; verbose = true, epoch_verbose = true, mbatch = 64)
    #training loop to go over all the dataset
    @info("Epoch # ", epoch)
    for i = 0:(length(ytrn)/m)-1
        reset_minibatch_train!(modelNLP)
        stats = solver(modelNLP; verbose = verbose)
        new_w = stats.solution
        set_vars!(modelNLP, new_w)
        if epoch_verbose
            @info("Minibatch = ", i)
            @info("Minibatch accuracy: ", KnetNLPModels.accuracy(modelNLP))
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
    epoch_verbose = true
)

    acc_arr = []
    iter_arr = []
    best_acc = 0
    for j = 1:mepoch
        if all_data
            acc = epoch_all!(modelNLP, solver, xtrn, ytrn, j; verbose, epoch_verbose, mbatch)
        else
            acc = epoch!(modelNLP, solver, xtrn, ytrn, j; verbose, epoch_verbose, mbatch)
        end

        if acc > best_acc
            #TODO write to file, KnetNLPModel, w
            best_acc = acc
        end

        append!(acc_arr, best_acc)
        append!(iter_arr, j)

        if j % 10 == 0
            @info("epoch #", j, "  acc= ", acc)
        end
        # add!(acc_arr, (j, acc))
        #TODO wirte to file 
        if acc > best_acc
            #TODO write to file, KnetNLPModel, w
            best_acc = acc
        end
    end

    c = hcat(iter_arr, acc_arr)
    #after each epoch if the accuracy better, stop 
    # all_data =true, go over them all, false only select minibatch
    return best_acc, c
end


"""This function will plots some samples annd perdicted vs true tags
"""
function plotSamples(myModel, xtrn, ytrn, data_set; samples = 5)
    rp = randperm(10000)
    x = [xtrn[:, :, :,rp[i]] for i = 1:samples]
    A = cat(x..., dims = 4);
    buff = myModel.chain(A);
    pred_y = findmax.(eachcol(buff));
    imgs = [data_set.convert2image(xtrn[:, :, :,rp[i]]) for i = 1:samples]

    p = plot(layout = (1, samples)) # Step 1
    i = 1
    for item in imgs
        plot!(p[i], item, ticks = false,title = string("T:", ytrn[rp[i]], "\nP:", pred_y[i][2]))
        i = i + 1
    end
    display(p)
end


###############
#Knet

# # For running experiments
# function train_knet!(model,dtrn,dtest; epoch= 10 )
#    loss(x,y)=model(x,y)
#    lossgradient = grad(loss)
#    acc_arr = []
#    iter_arr = []
#    best_acc = 0
#    for j = 1:mepoch
#         acc = epoch_knet!(model, xtrn, ytrn, lossgradient; mbatch)
#        if acc > best_acc
#            #TODO write to file, KnetNLPModel, w
#            best_acc = acc
#        end

#        append!(acc_arr, best_acc)
#        append!(iter_arr, j)

#        if j % 10 == 0
#            @info("epoch #", j, "  acc= ", acc)
#        end
#        if acc > best_acc
#            best_acc = acc
#        end
#    end

#    c = hcat(iter_arr, acc_arr)
#    return best_acc, c
# end

# # Training
# function epoch_knet!(model, xtrn, ytrn,lossgrad;  mbatch=100)
#     data = minibatch(xtrn, ytrn, mbatch;
#                    shuffle=true,
#                    xtype=Knet.array_type[])
#     for (x, y) in data
#         g = lossgrad(w, x, y)
#         update!(w, g, o)
#     end
# end