# module JSOSolvers_inexcat

# stdlib
using LinearAlgebra, Logging, Printf

# JSO packages
using Krylov, LinearOperators, NLPModels, NLPModelsModifiers, SolverCore, SolverTools

# import SolverCore.solve!
# import Krylov.solve!
# export solve!

function get_status(
    nlp;
    elapsed_time = 0.0,
    optimal = false,
    unbounded = false,
    max_eval = Inf,
    max_time = Inf,
)
    if optimal
        :first_order
    elseif unbounded
        :unbounded
    elseif neval_obj(nlp) > max_eval ≥ 0
        :max_eval
    elseif elapsed_time > max_time
        :max_time
    else
        :unknown
    end
end

function get_status(
    nls::AbstractNLSModel;
    elapsed_time = 0.0,
    optimal = false,
    unbounded = false,
    max_eval = Inf,
    max_time = Inf,
)
    if optimal
        :first_order
    elseif unbounded
        :unbounded
    elseif neval_residual(nls) > max_eval ≥ 0
        :max_eval
    elseif elapsed_time > max_time
        :max_time
    else
        :unknown
    end
end

# Unconstrained solvers

include("SR2.jl")


# end
