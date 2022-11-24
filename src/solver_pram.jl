using SolverParameters

#testing it 

# B = R2ParameterSet{R}(√eps(R), √eps(R), 0.1, 0.3, 1.1, 1.9, zero(R), 0.9)
# println(B.γ1.value)

# used in the callback of R2 for training deep learning model 
mutable struct StochasticR2Data
    epoch::Int
    i::Int
    # other fields as needed...
    max_epoch::Int
    acc_arr::Vector{Float64}
    train_acc_arr::Vector{Float64}
    epoch_arr::Vector{Float64}
    grads_arr::Vector{Float64} #TODO fix this type to be dynamic
    ϵ::Float64 #TODO Fix with type T
end
