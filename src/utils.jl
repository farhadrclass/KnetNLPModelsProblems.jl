# using Revise        # importantly, this must come before
# using Test
using JSOSolvers
using LinearAlgebra
using Random
using Printf
# using DataFrames
# using OptimizationProblems
using NLPModels
# using ADNLPModels
# using OptimizationProblems.ADNLPProblems
# using SolverCore
using SolverCore
using Plots

using Knet, Images, MLDatasets




# using Plots
# using Profile
# using StochasticRounding

include("struct_utils.jl")
function loaddata(data_flag, T)
    if (data_flag == 1)
        @info("Loading MNITS...")
        xtrn, ytrn = MNIST.traindata(T)
        ytrn[ytrn.==0] .= 10
        xtst, ytst = MNIST.testdata(T)
        ytst[ytst.==0] .= 10
        @info("Loaded MNIST")
        # else if # MNIST FASHION
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
function epoch!(modelNLP, solver, iter; verbos = true, mbatch = 64)
    reset_minibatch_train!(modelNLP)
    stats = solver(modelNLP)#;verbos=verbos)
    new_w = stats.solution
    set_vars!(modelNLP, new_w)
    if verbos
        @info("Epoch # ", iter)
        @info("Minibatch accuracy: ", KnetNLPModels.accuracy(modelNLP))
    end
    return KnetNLPModels.accuracy(modelNLP)
end

#run over all minibatched 
function epoch_all!(modelNLP, solver, xtrn, ytrn; verbos = true, mbatch = 64)
    #training loop to go over all the dataset
    for i = 0:(length(ytrn)/m)-1
        reset_minibatch_train!(LeNetNLPModel)
        stats = solver(modelNLP)#;verbos= verbos)
        new_w = stats.solution
        set_vars!(modelNLP, new_w)
        if verbos
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
    verbos = true,
)

    acc_arr = []
    iter_arr = []
    best_acc = 0
    for j = 1:mepoch
        if all_data
            acc = epoch_all!(modelNLP, solver, xtrn, ytrn; verbos = verbos, mbatch = 64)
        else
            acc = epoch!(modelNLP, solver, j; verbos = verbos, mbatch = 64)
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
function plotSamples(myModel, xtrn, ytrn; samples = 5)
    rp = randperm(10000)
    x = [xtrn[:, :, rp[i]] for i = 1:samples]
    A = cat(x..., dims = 4)
    buff = myModel.chain(A)
    pred_y = findmax.(eachcol(buff.value))
    # println("True classes: ",ytrn[rp[1:samples]])
    # println("model perdicted classes: ",)
    imgs = [MNIST.convert2image(xtrn[:, :, rp[i]]) for i = 1:samples]

    p = plot(layout = (1, samples)) # Step 1
    i = 1
    for item in imgs
        plot!(p[i], item, ticks = false,title = string("T:", ytrn[rp[i]], "\nP:", pred_y[i][2]))
        i = i + 1
    end
    display(p)
end

# #From https://github.com/denizyuret/Knet.jl/blob/master/examples/dcgan-mnist/dcgan.jl
# function plot_generations(
#     wg, mg; z=nothing, gridsize=(8,8), scale=1.0, savefile=nothing)
#     if z == nothing
#         nimg = prod(gridsize)
#         zdim = size(wg[1],2)
#         atype = typeof(wg[1]) # wg[1] isa KnetArray ? KnetArray{Float32} : Array{Float32}
#         z = sample_noise(atype,zdim,nimg)
#     end
#     output = Array(0.5 .* (1 .+ gnet(wg,z,mg; training=false)))
#     images = map(i->output[:,:,:,i], 1:size(output,4))
#     grid = make_image_grid(images; gridsize=gridsize, scale=scale)
#     if savefile == nothing
#         display(colorview(Gray, grid))
#     else
#         Knet.save(savefile, grid)
#     end
# end

# function make_image_grid(images; gridsize=(8,8), scale=2.0, height=28, width=28)
#     shape = (height, width)
#     nchannels = size(first(images))[end]
#     @assert nchannels == 1 || nchannels == 3
#     shp = map(x->Int(round(x*scale)), shape)
#     y = map(x->Images.imresize(x,shp), images)
#     gridx, gridy = gridsize
#     outdims = (gridx*shp[1]+gridx+1,gridy*shp[2]+gridy+1)
#     out = zeros(outdims..., nchannels)
#     for k = 1:gridx+1; out[(k-1)*(shp[1]+1)+1,:,:] .= 1.0; end
#     for k = 1:gridy+1; out[:,(k-1)*(shp[2]+1)+1,:] .= 1.0; end

#     x0 = y0 = 2
#     for k = 1:length(y)
#         x1 = x0+shp[1]-1
#         y1 = y0+shp[2]-1
#         out[x0:x1,y0:y1,:] .= y[k]

#         y0 = y1+2
#         if k % gridy == 0
#             x0 = x1+2
#             y0 = 2
#         else
#             y0 = y1+2
#         end
#     end

#     out = convert(Array{Float64}, map(x->isnan(x) ? 0 : x, out))
#     if nchannels == 1
#         out = reshape(out, (size(out,1),size(out,2)))
#         out = permutedims(out, (2,1))
#     else
#         out = permutedims(out, (3,1,2))
#     end
#     return out
# end
