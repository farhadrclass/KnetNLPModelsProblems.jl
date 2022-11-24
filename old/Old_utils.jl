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
using Statistics
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


"""
    all_accuracy(nlp::AbstractKnetNLPModel)
Compute the accuracy of the network `nlp.chain` given the data in `nlp.tests`.
uses the whole test data sets"""
all_accuracy(nlp::AbstractKnetNLPModel) = Knet.accuracy(nlp.chain; data = nlp.data_test)
"""
Gets the next minibatch
Input:
nlp:: KnetNLPModel 
i:: current location in the iterator #TODO maybe do it in the knetnlpmodels as a current location
"""
function minibatch_next_train!(nlp::AbstractKnetNLPModel, i)
    i += nlp.size_minibatch # update the i by mini_batch size
    if (i >= nlp.training_minibatch_iterator.imax)
        # reset to the begining and return xero 
        nlp.current_training_minibatch = first(nlp.training_minibatch_iterator) # reset to the first one
        return 0
    else
        next = iterate(nlp.training_minibatch_iterator, i)
        nlp.current_training_minibatch = next[1]
        return i
    end
end



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
    s = "..//results//samples_" * Strings(samples) #TODO change to dynamic name wrtie to file
    png(s)
end




function zero_mean_inner(x, y, w = similar(x))
    w .= x .* y
    μw = mean(w)
    w .= w .- μw
    s = sum(w) + length(w) * μw
    return s
end

function build_inner_product(x, y)
    return x'y
end

function mat_zero_mean(M, N)
    # M = copy(V)
    # N = copy(U)
    # if (M isa AbstractVecOrMat)
    #     A = M;        
    # else
    #     A = M.value;
    # end
    # if (N isa AbstractVecOrMat)
    #     B = N;

    # else
    #     B = N.value;
    # end

    A = mat(M)
    B = mat(N)
    if A isa AbstractVecOrMat
        T = eltype(A)
        if size(A)[2] == size(B)[1]
            C = Array{T}(undef, size(A)[1], size(B)[2])
            i = 0
            for row in eachrow(A)
                i += 1
                j = 0
                for col in eachcol(B)
                    j += 1
                    C[i, j] = zero_mean_inner(row, col)
                end
            end
            return C
        else
            error("Size mismatched, cannot multiply A and B.")
            return -1
        end
    else
        C = M * N
    end
    return C

end


function mat_mult(V, U)
    M = copy(V)
    N = copy(U)
    if (M isa AbstractVecOrMat)
        A = M
    else
        A = M.value
    end
    if (N isa AbstractVecOrMat)
        B = N
    else
        B = N.value
    end

    T = eltype(A)
    if size(A)[2] == size(B)[1]
        C = Array{T}(undef, size(A)[1], size(B)[2])
        i = 0
        for row in eachrow(A)
            i += 1
            j = 0
            for col in eachcol(B)
                j += 1
                C[i, j] = build_inner_product(row, col)
            end
        end
        return C
    else
        error("Size mismatched, cannot multiply A and B.")
        return -1
    end
end
