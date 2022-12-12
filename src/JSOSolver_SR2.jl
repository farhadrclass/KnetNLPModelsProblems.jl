# module JSOSolvers_inexcat

# stdlib
using LinearAlgebra, Logging, Printf

# JSO packages
using Krylov, LinearOperators, NLPModels, NLPModelsModifiers, SolverCore, SolverTools


##################################################################



# used to send Parameter to R2 solver
struct R2ParameterSet{T<:AbstractFloat} <: AbstractParameterSet  #TODO change it to include  StochasticRounding

    atol::Parameter{T,RealInterval{T}}
    rtol::Parameter{T,RealInterval{T}}
    η1::Parameter{T,RealInterval{T}}
    η2::Parameter{T,RealInterval{T}}
    γ1::Parameter{T,RealInterval{T}}
    γ2::Parameter{T,RealInterval{T}}
    σmin::Parameter{T,RealInterval{T}}
    β::Parameter{T,RealInterval{T}}
    # β::Parameter{Float64, RealInterval{Float64}}


    function R2ParameterSet{T}(;
        atol::T = √eps(T),
        rtol::T = √eps(T),
        η1 = eps(T)^(1 / 4),#TODO check if this is too big
        η2 = T(0.95),
        γ1 = T(1 / 2),
        γ2 = 1 / γ1,
        σmin = zero(T),# change this
        β::T = T(0),
    ) where {T<:AbstractFloat} # empthy constructor  

        atol >= 0 || throw(DomainError("invalid atol, atol>=0"))
        rtol >= 0 || throw(DomainError("invalid rtol, rtol >=0"))
        0 < η1 <= η2 <= 1 || throw(DomainError("invalid: 0 < η1 <= η2 <= 1"))
        0 <= β < 1 || throw(DomainError("invalid: β needs to be between [0,1)"))
        0 < γ1 < 1 <= γ2 || throw(DomainError("invalid 0 < γ1 < 1 <= γ2 "))
        new(
            Parameter(T(atol), RealInterval(T(-1000), T(1000)), "atol"), #TODO actual name 
            Parameter(T(rtol), RealInterval(T(-1000), T(1000)), "rtol"),
            Parameter(T(η1), RealInterval(T(-1000), T(1000)), "η1"),
            Parameter(T(η2), RealInterval(T(-10000), T(10000)), "η2"),
            Parameter(T(γ1), RealInterval(T(-10000), T(1000)), "γ1"),
            Parameter(T(γ2), RealInterval(T(-10000), T(1000)), "γ2"),
            Parameter(σmin, RealInterval(T(-10000), T(1000)), "σmin"),
            Parameter(T(β), RealInterval(T(-10000), T(1000)), "β"),
        )

    end

end




####################################################################




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
