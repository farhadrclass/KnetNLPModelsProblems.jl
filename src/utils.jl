"""
    all_accuracy(nlp::AbstractKnetNLPModel)
Compute the accuracy of the network `nlp.chain` given the data in `nlp.tests`.
uses the whole test data sets"""
all_accuracy(nlp::AbstractKnetNLPModel) = Knet.accuracy(nlp.chain; data = nlp.data_test)

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
    s= "..//results//samples_" * Strings(samples) #TODO change to dynamic name wrtie to file
    png(s)
end
