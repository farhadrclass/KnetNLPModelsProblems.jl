# using Revise        # importantly, this must come before
# using Test
using JSOSolvers
using LinearAlgebra
# using Random
# using Printf
# using DataFrames
# using OptimizationProblems
# using NLPModels
# using ADNLPModels
# using OptimizationProblems.ADNLPProblems
# using SolverCore

# using Plots
# using Profile
# using StochasticRounding

include("struct_utils.jl")
function loaddata(data_flag)
    if (data_flag == 1)
        @info("Loading MNIST...")
        xtrn, ytrn = MNIST.traindata(T)
        ytrn[ytrn.==0] .= 10
        xtst, ytst = MNIST.testdata(Float32)
        ytst[ytst.==0] .= 10
        @info("Loaded MNIST")
    else # CIFAR data
        @info("Loading CIFAR 10...")
        xtrn, ytrn = CIFAR10.traindata()
        xtst, ytst = CIFAR10.testdata()
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
function epoch!(modelNLP, solver, epoch, verbose; mbatch = 64)
    reset_minibatch_train!(modelNLP)
    @info("Epoch # ", epoch)
    stats = solver(modelNLP; verbose = verbose)
    new_w = stats.solution
    set_vars!(modelNLP, new_w)
    @info("Minibatch Accracy: ", KnetNLPModels.accuracy(modelNLP))
    return KnetNLPModels.accuracy(modelNLP)
end

#run over all minibatched 
function epoch_all!(modelNLP, solver, epoch, verbose; mbatch = 64)
    #training loop to go over all the dataset
    @info("Epoch # ", epoch)
    for i = 0:(length(ytrn)/m)-1
        reset_minibatch_train!(modelNLP)
        @info("Minibatch = ", i)
        stats = solver(modelNLP; verbose = verbose)
        new_w = stats.solution
        set_vars!(modelNLP, new_w)
        @info("Minibatch Accracy: ", KnetNLPModels.accuracy(modelNLP))
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
    verbose = true
)

    acc_arr = []#todo fix this 
    best_acc = 0
    for j = 0:mepoch
        if all_data
            acc = epoch_all!(modelNLP, solver, j, verbose; mbatch)
        else
            acc = epoch!(modelNLP, solver, j, verbose; mbatch)
        end
        if j % 10 == 0
            @info("epoch #", j, "  acc= ", acc)
        end
        # TODO add acc to a List 
        add!(acc_arr, (j, acc))
        #TODO wirte to file 
        if acc > best_acc
            #TODO write to file, KnetNLPModel, w
            best_acc = acc
        end
        return acc_arr, best_acc
    end
    #after each epoch if the accracy better, stop 
    # all_data =true, go over them all, false only select minibatch

end
