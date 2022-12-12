# module KnetNLPModelsProblems



using CUDA, IterTools
using JSOSolvers
using LinearAlgebra
using Random
using Printf
using NLPModels
using SolverCore
using Plots
using Knet, Images, MLDatasets
using StochasticRounding
using Statistics: mean
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
# Write your package code here.
include("solver_pram.jl")
include("JSOSolver_SR2.jl")
# include("SR2.jl") #TODO replace this 

include("struct_DNN.jl")
include("utils.jl")

include("train_tools.jl")

# including the models for DNN
include("models/fc_mnist.jl")
include("models/lenet_cifar10.jl")
include("models/lenet_mnist.jl")


# using Pkg
# Pkg.instantiate()
# end


# For now I will add the problems here but later I move them to their own folder 
#TODO use condition on which one to run
# include("problems/CPU/LeNet_MNIST.jl")
# include("problems/GPU/LeNet_MNIST.jl")
# include("problems/GPU/LeNet_CIFAR10.jl")


#TODO 
# RENSET
