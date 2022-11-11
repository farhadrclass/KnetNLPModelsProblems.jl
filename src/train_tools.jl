#TODO one potential problem is that we stop before doing the epoch if the stopping condition happens, needs to fix it so it run the epochs not check the ϵ 
# TODO add \rho and n and structur , solver object struck ?
#  Todo   
# set_objective!(stats, fck) # old value
#   grad!(nlp, x, ∇fk) #grad is wrong 
#   norm_∇fk = norm(∇fk) # wrong 
# todo accept or not accept the step? 
# reset the 



###############################
# Moving avarage
using MarketTechnicals

#calculate Exponential Moving Average
function ema_avg(arr, window)
    size_arr = length(arr)
    if size_arr >= window
        return last(ema(arr, window)) #$ using MarketTechnicals
    end
    return -1 # means the size is not reached yet
end


#calculate moving avarage
function mv_avg(arr, window)
    size_arr = length(arr)
    if size_arr >= window
        # return (sum(arr[size_arr-window+1:size_arr])/window)
        #  or
        return last(sma(arr, window)) #$ using MarketTechnicals
    end
    return -1 # means the size is not reached yet
end


################################################

function cb(nlp, solver, stats, param::AbstractParameterSet, data::StochasticR2Data)
    data.i = KnetNLPModels.minibatch_next_train!(nlp)
    if (data.i == 2)
        norm_∇fk = norm(solver.gx)
        data.ϵ = param.atol.value + param.rtol.value * norm_∇fk
    end
   #TODO resett the SR2 grad and values
    window = 5; #TODO change that 
    append!(data.grads_arr , solver.gx) # to keep the grads from each call 
    # avg_grad_mv = mv_avg(data.grads_arr, window)
    avg_grad_mv = ema_avg(data.grads_arr, window)
    if (avg_grad_mv <= data.ϵ)
        stats.status = :first_order #optimal TODO change this
    end
    best_acc = 0
    if data.i == 1
        
        # reset
        data.grads_arr = [] 
        data.epoch += 1
        acc = KnetNLPModels.accuracy(nlp)
        if acc > best_acc
            best_acc = acc
        end
        # TODO  make sure we calculate mini-batch accracy
        train_acc = Knet.accuracy(nlp.chain; data = nlp.training_minibatch_iterator) #TODO minibatch acc.
        @info("epoch #", data.epoch, "  acc= ", train_acc)
        append!(data.train_acc_arr, train_acc) #TODO fix this to save the acc
        append!(data.acc_arr, acc) #TODO fix this to save the acc
        append!(data.epoch_arr, data.epoch)
    end

    if data.epoch == data.max_epoch
        stats.status = :user
    end
    # we need to reset the ∇fk and σk here since the data has changes
    # we also do not need to know β since the momentum doesn't make sense in case where the data has changed?

    # set_objective!(stats, fck) # old value
    #   grad!(nlp, x, ∇fk) #grad is wrong 
    #   norm_∇fk = norm(∇fk) # wrong 

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
    β = 0.9,
    atol = 0.05,
    rtol = 0.05,
    R = Float32,

    # max_iter = 1000, # we can play with this and see what happens in R2, 1 means one itration but the relation is not 1-to-1, 
    #TODO  add max itration 
)


    # TODO add param here 
    param = R2ParameterSet{R}() #(√eps(R), √eps(R), 0.1, 0.3, 1.1, 1.9, zero(R), 0.9) # TODO add the param here
    stochastic_data = StochasticR2Data(0, 0, mepoch, [], [], [],[])
    solver_stats = solver(
        modelNLP;
        param = param,
        verbose = verbose,
        # max_time = 10000000.0,#TODO issue with this
        callback = (nlp, solver, stats) ->
            cb(nlp, solver, stats, param, stochastic_data),
    )

    return stochastic_data

end


###############
#  Knet SGD training


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