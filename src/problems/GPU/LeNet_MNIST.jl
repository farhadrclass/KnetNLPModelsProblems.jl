
T = Float32
# Knet.atype() = Array{T}
if CUDA.functional() 
    Knet.array_type[] = CUDA.CuArray{T}
else 
    Knet.array_type[] = Array{T}
end


(xtrn, ytrn), (xtst, ytst) = loaddata(1, T)

# size of minibatch 
m = 125
max_epochs = 50

knetModel, myModel = lenet_prob(xtrn, ytrn, xtst, ytst, minibatchSize = m)
# random init the w 
# w = rand(eltype(myModel.meta.x0), size(myModel.meta.x0)[1])
# set_vars!(myModel, w)



println("Training SR2 with KNET")
trained_model = train_knetNLPmodel!(
    myModel,
    SR2,
    xtrn,
    ytrn;
    mbatch = m,
    mepoch = max_epochs,
    verbose = 1,
    # β = T(0.3),
    # atol = T(0.05),
    # rtol = T(0.05),
    # η1 = eps(T)
)


res = trained_model
epochs = res.epoch_arr
acc = res.acc_arr
train_acc = res.train_acc_arr


println("Training SGD with KNET")
# Train Knet
trained_model_knet = train_knet(knetModel, xtrn, ytrn, xtst, ytst;mbatch=m, mepoch = max_epochs) #TODO some reason when mepoch=max_epochs, will give error , maybe Int(max_epochs)
res_knet = trained_model_knet[2]
epochs_knet = res_knet[:, 1]
acc_knet = res_knet[:, 2]
train_acc_knet = res_knet[:, 3]


fig = plot(
    epochs,
    # title = " test accuracy vs Epoch",
    markershape = :star4,
    acc,
    label = "test accuracy R2",
    legend = :bottomright,
    xlabel = "epoch",
    ylabel = "accuracy",
)
plot!(fig, epochs, markershape = :star1, acc_knet, label = "test accuracy SGD")

# plotSamples(myModel, xtrn, ytrn, MNIST; samples=10)

fig = plot(
    epochs,
    title = "train accuracy vs Epoch on Float32",
    markershape = :star4,
    train_acc,
    label = "train accuracy R2",
    legend = :bottomright,
    xlabel = "epoch",
    ylabel = "accuracy",
)
plot!(fig, epochs, markershape = :star1, train_acc_knet, label = "train accuracy SGD")


# plotSamples(myModel, xtrn, ytrn, MNIST; samples=10)


#Plot all

fig = plot(
    epochs,
    # title = " All accuracy vs Epoch on Float32",
    markershape = :star1,
    acc,
    label = "test accuracy R2",
    legend = :bottomright,
    xlabel = "epoch",
    ylabel = "accuracy",
)



plot!(
    fig,
    epochs,
    markershape = :star1,
    train_acc,
    label = "train accuracy R2",
    legend = :bottomright,
    linestyle = :dash,
)
plot!(fig, epochs, markershape = :star4, acc_knet, label = "test accuracy SGD")

plot!(
    fig,
    epochs,
    markershape = :star4,
    train_acc_knet,
    label = "train accuracy SGD",
    linestyle = :dot,
)

savefig("run_GPU_LENET_MNIST.png")