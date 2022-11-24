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

mutable struct StochasticR2Data
    epoch::Int
    i::Int
    # other fields as needed...
    max_epoch::Int
    acc_arr::Vector{Float64}
    train_acc_arr::Vector{Float64}
    epoch_arr::Vector{Float64}
end



function cb(nlp, solver, stats, data::StochasticR2Data)

    if stats.iter % 2 == 0  #TODO testing what happens 
        #TODO one potential problem is that we stop before doing the epoch if the stopping condition happens, needs to fix it so it run the epochs not check the ϵ 
        data.i = minibatch_next_train!(nlp, data.i)
        # TODO add \rho and n and structur , solver object struck ?
        #  Todo   
        # set_objective!(stats, fck) # old value
        #   grad!(nlp, x, ∇fk) #grad is wrong 
        #   norm_∇fk = norm(∇fk) # wrong 
        # todo accept or not accept the step? 
        # reset the 
        best_acc = 0
        if data.i == 0
            data.epoch += 1

            # if j % 2 == 0
            @info("epoch #", data.epoch, "  acc= ", train_acc)
            # end
            #TODO save the accracy
            ## new_w = stats.solution
            new_w = solver.x
            set_vars!(nlp, new_w)
            acc = KnetNLPModels.accuracy(nlp)
            if acc > best_acc
                #TODO write to file, KnetNLPModel, w
                best_acc = acc
            end
            # train accracy
            # data_buff = create_minibatch(
            #     nlp.current_training_minibatch[1],
            #     nlp.current_training_minibatch[2],
            #     mbatch,
            # )
            train_acc = Knet.accuracy(nlp.chain; data = nlp.training_minibatch_iterator) #TODO minibatch acc.
            append!(data.train_acc_arr, train_acc) #TODO fix this to save the acc
            append!(data.acc_arr, acc) #TODO fix this to save the acc
            append!(data.epoch_arr, data.epoch)


        end

        if data.epoch == data.max_epoch
            stats.status = :user
        end
    end
end

#runs over only one random one one step of R2Solver
# by changine max_iter we can see if more steps have effect
function train_knetNLPmodel!(
    modelNLP,
    solver,
    xtrn,
    ytrn;
    mbatch = 64,     #todo see if we need this , in future we can update the number of batch size in different epoch
    mepoch = 10,
    verbose = -1,
    β = T(0.9),
    atol = T(0.05),
    rtol = T(0.05),
    # max_iter = 1000, # we can play with this and see what happens in R2, 1 means one itration but the relation is not 1-to-1, 
    #TODO  add max itration 
) where {T}




    #x::V = nlp.meta.x0,



    # max_time::Float64 = 30.0,
    # max_eval::Int = -1,
    # β::T = T(0),


    stochastic_data = StochasticR2Data(0, 0, mepoch, [], [], [])
    solver_stats = solver(
        modelNLP;
        atol = atol,
        rtol = rtol,
        verbose = verbose,
        # max_iter = max_iter,
        max_time = 10000000.0,#TODO issue with this
        β = β,
        callback = (nlp, solver, stats) -> cb(nlp, solver, stats, stochastic_data),
    )

    return stochastic_data

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
    epoch_arr = []
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
        append!(epoch_arr, j)

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
    c = hcat(epoch_arr, acc_arr, train_acc_arr)
    #after each epoch if the accuracy better, stop 
    return best_acc, c
end
