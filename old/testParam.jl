# using SolverParameters

# struct R2ParameterSet{T<:AbstractFloat} <: AbstractParameterSet  #TODO change it to include  StochasticRounding

#     atol::Parameter{T,RealInterval{T}}
#     rtol::Parameter{T,RealInterval{T}}
#     η1::Parameter{T,RealInterval{T}}
#     η2::Parameter{T,RealInterval{T}}
#     γ1::Parameter{T,RealInterval{T}}
#     γ2::Parameter{T,RealInterval{T}}
#     σmin::Parameter{T,RealInterval{T}}
#     β::Parameter{T,RealInterval{T}}
#     # monotone::Parameter{Int, IntegerRange{Int}}
#     # β::Parameter{Float64, RealInterval{Float64}}

#     # empty constructor 

#     R2ParameterSet{T}() where {T<:AbstractFloat} = new(
#         Parameter(√eps(T), RealInterval(T(0), T(1)), "real"),
#         Parameter(√eps(T), RealInterval(T(0), T(1)), "real"),
#         Parameter(T(eps(T)^(1 / 4)), RealInterval(T(0), T(1)), "real"),
#         Parameter(T(0.95), RealInterval(T(0), T(10)), "real"),
#         Parameter(T(1 / 2), RealInterval(T(0), T(1)), "real"),
#         Parameter(T(2), RealInterval(T(0), T(10)), "real"),
#         Parameter(zero(T), RealInterval(T(0), T(1000)), "real"),
#         Parameter(T(0), RealInterval(T(0), T(1000)), "real"),
#     )


#     function R2ParameterSet{T}(atol, rtol, η1, η2, γ1, γ2, σmin, β) where {T<:AbstractFloat} # empthy constructor 
#         if atol <= 0
#             error("invalid atol")
#         elseif rtol <= 0
#             error("invalid rtol")
#         elseif β >= 1 || β < 0 #TODO check range on beta
#             error("invalid: β needs to be between [0,1)")
#         elseif γ1 < 1 || γ1 >= γ2
#             error("invalid γ1 <= γ2")
#         elseif (η1 > η2 || η1 <= 0 || η1 > 1 || η2 > 1)
#             error("invalid: 0 < η1 <= η2 <= 1")
#         else
#             new(
#                 Parameter(T(atol), RealInterval(T(0), T(1)), "real"),
#                 Parameter(T(rtol), RealInterval(T(0), T(1)), "real"),
#                 Parameter(T(η1), RealInterval(T(0), T(1)), "real"),
#                 Parameter(T(η2), RealInterval(T(0), T(10)), "real"),
#                 Parameter(T(γ1), RealInterval(T(1), T(10)), "real"),
#                 Parameter(T(γ2), RealInterval(T(1), T(10)), "real"),
#                 Parameter(σmin, RealInterval(T(0), T(1000)), "real"),
#                 Parameter(T(β), RealInterval(T(0), T(1000)), "real"),
#             )
#         end
#     end

# end

# #testing it 

# A = R2ParameterSet{Float32}()
# println(A.atol.value)
# R = Float16
# B = R2ParameterSet{R}(√eps(R), √eps(R), 0.1, 0.3, 1.1, 1.9, zero(R), 0.9)
# println(B.γ1.value)


# C = R2ParameterSet{R}(√eps(R), √eps(R), 0.1, 0.01, 0.1, 1.9, zero(R), 0.9) # with issues
