using Knet
# x = Float16.(rand(5,5))
# y = Float16.(rand(5,5))

# println(typeof(x))


# A = param(x)
# B = param(y)
# println(typeof(A))

# # we have a problelm the param makes the type Float32!!
# x    Matrix{Float16}
# A    Param{Matrix{Float32}}  To fix it we have to set Knet.atype()

include("utils.jl")
T = Float16

Knet.atype() = Array{T}

x = T.(rand(5,5))
y = T.(rand(5,5))

println(typeof(x))


A = param(x)
B = param(y)
println(typeof(A))


mat_zero_mean(A,B)

mat_mult(A,B)