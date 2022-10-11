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
    All_accuracy(nlp::AbstractKnetNLPModel)
Compute the accuracy of the network `nlp.chain` given the data in `nlp.tests`.
uses the whole test data sets"""
All_accuracy(nlp::AbstractKnetNLPModel) = Knet.accuracy(nlp.chain; data = nlp.data_test)

```
Goes through the whole mini_batch one item at a time and not randomly
TODO make a PR for KnetNLPmodels
```
function reset_minibatch_train_next!(nlp::AbstractKnetNLPModel)
    next = iterate(nlp.current_training_minibatch)
    if (next === nothing)
        nlp.current_training_minibatch = first(nlp.training_minibatch_iterator) # reset to the first one
        #TODO end of the batch 
        return -1
    else
        nlp.current_training_minibatch = next
    end
end




function cb(nlp, solver, stats,data::StochasticR2Data)
    endFlag = reset_minibatch_train_next!(nlp)
    best_acc = 0
    if endFlag == -1
        data.epochs = epochs + 1
        #TODO save the accracy
        new_w = stats.solution
        set_vars!(modelNLP, new_w)
        acc = KnetNLPModels.accuracy(modelNLP)
        if acc > best_acc
            #TODO write to file, KnetNLPModel, w
            best_acc = acc
        end
        # train accracy
        data_buff = create_minibatch(
            modelNLP.current_training_minibatch[1],
            modelNLP.current_training_minibatch[2],
            mbatch,
        ) # we need to create iterator it has one item
        train_acc = Knet.accuracy(modelNLP.chain; data = data_buff)
        append!(data.train_acc_arr, train_acc) #TODO fix this to save the acc


        append!(data.acc_arr, acc) #TODO fix this to save the acc
        append!(data.iter_arr, j)

        if j % 2 == 0
            @info("epoch #", j, "  acc= ", train_acc)
        end

    end

    if epochs == max_epoch
        stats.status = :user
    end
end

#runs over only one random one one step of R2Solver
# by changine max_iter we can see if more steps have effect
function train_knetNLPmodel!(
    modelNLP,
    solver,
    xtrn,
    ytrn;
    mbatch = 64,     #todo see if we need this , in future we can update the number of batch size in different epochs
    mepoch = 10,
    verbose = false,
    β = T(0.9),
    max_iter = 1000, # we can play with this and see what happens in R2, 1 means one itration but the relation is not 1-to-1, 
    #TODO  add max itration 
    epoch_verbose = true,
) where {T}

  
    mutable struct StochasticR2Data
        epoch::Int
        # other fields as needed...
        max_epoch::Int
        acc_arr::Vector{Float64}
        train_acc_arr::Vector{Float64}
        iter_arr::Vector{Float64}
      end
      
      stochastic_data = Stochasticr2Data(0,mepoch,[],[],[])
    solver_stats = solver(
        modelNLP;
        atol = 0.05,
        rtol = 0.05,
        verbose = verbose,
        # max_iter = max_iter,
        β = β,
        callback = (nlp, solver, stats) -> cb(nlp, solver, stats, stochastic_data),
    )

end


###############
#Knet


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