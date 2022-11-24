using Plots
using CSV
using DataFrames

cifar10 = CSV.read(
    "/Users/nathanallaire/Desktop/GERAD/KnetNLPModelsProblems/examples/GPU/res/results/cifar10.csv",
    DataFrame,
)
mnist = CSV.read(
    "/Users/nathanallaire/Desktop/GERAD/KnetNLPModelsProblems/examples/GPU/res/results/mnist.csv",
    DataFrame,
)


cifar10_plot = cifar10[1:100:3000, :]
mnist_plot = mnist[1:20:400, :]

fig = plot(
    cifar10_plot[:, 1],
    cifar10_plot[:, 2],
    label = "accuracy test",
    color = "darkred",
    markershape = :circle,
    lw = 2,
    thickness_scaling = 1.3,
    legend = :topleft,
    size = (1000, 400),
    title = "Accuracy of LeNet on CIFAR10 dataset - GPU",
)

plot!(
    fig,
    cifar10_plot[:, 1],
    cifar10_plot[:, 3],
    label = "accuracy train",
    color = "tomato",
    markershape = :diamond,
    lw = 2,
    thickness_scaling = 1.3,
    legend = :topleft,
    size = (1000, 400),
)


fig2 = plot(
    mnist_plot[:, 1],
    mnist_plot[:, 2],
    label = "accuracy test",
    color = "blue4",
    markershape = :circle,
    lw = 2,
    thickness_scaling = 1.3,
    legend = :topleft,
    size = (1000, 400),
    title = "Accuracy of LeNet on MNIST dataset - GPU",
)

plot!(
    fig2,
    mnist_plot[:, 1],
    mnist_plot[:, 3],
    label = "accuracy train",
    color = "dodgerblue",
    markershape = :diamond,
    lw = 2,
    thickness_scaling = 1.3,
    legend = :topleft,
    size = (1000, 400),
)
