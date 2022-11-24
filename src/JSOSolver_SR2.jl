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
    # monotone::Parameter{Int, IntegerRange{Int}}
    # β::Parameter{Float64, RealInterval{Float64}}

    # empty constructor 

    #TODO --kwargs...
    R2ParameterSet{T}() where {T<:AbstractFloat} = new(
        Parameter(√eps(T), RealInterval(T(0), T(1)), "atol"),
        Parameter(√eps(T), RealInterval(T(0), T(1)), "rtol"),
        Parameter(T(eps(T)^(1 / 4)), RealInterval(T(0), T(1)), "η1"),
        Parameter(T(0.95), RealInterval(T(0), T(10)), "η2"),
        Parameter(T(1 / 2), RealInterval(T(0), T(1)), "γ1"),
        Parameter(T(2), RealInterval(T(0), T(10)), "γ2"),
        Parameter(zero(T), RealInterval(T(0), T(1000)), "σmin"),
        Parameter(T(0), RealInterval(T(0), T(1000)), "β"),
    )


    function R2ParameterSet{T}(atol, rtol, η1, η2, γ1, γ2, σmin, β) where {T<:AbstractFloat} # empthy constructor 
        #TODO 
        atol >= 0 || throw(DomainError("invalid atol, atol>=0"))
        rtol >= 0 || throw(DomainError("invalid rtol, rtol >=0"))
        0 < η1 <= η2 <= 1 || throw(DomainError("invalid: 0 < η1 <= η2 <= 1"))


        if atol <= 0
            throw(DomainError("invalid atol"))
        elseif rtol <= 0
            throw(DomainError("invalid rtol"))
        elseif β >= 1 || β < 0 #TODO check range on beta
            throw(DomainError("invalid: β needs to be between [0,1)"))
        elseif γ1 < 1 || γ1 >= γ2
            throw(DomainError("invalid γ1 <= γ2"))
        elseif (η1 > η2 || η1 <= 0 || η1 > 1 || η2 > 1)
            throw(DomainError("invalid: 0 < η1 <= η2 <= 1"))
        else
            new(
                Parameter(T(atol), RealInterval(T(0), T(1)), "atol"), #TODO actual name 
                Parameter(T(rtol), RealInterval(T(0), T(1)), "rtol"),
                Parameter(T(η1), RealInterval(T(0), T(1)), "η1"),
                Parameter(T(η2), RealInterval(T(0), T(10)), "η2"),
                Parameter(T(γ1), RealInterval(T(1), T(10)), "γ1"),
                Parameter(T(γ2), RealInterval(T(1), T(10)), "γ2"),
                Parameter(σmin, RealInterval(T(0), T(1000)), "σmin"),
                Parameter(T(β), RealInterval(T(0), T(1000)), "β"),
            )
        end
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
