# export SR2, SR2Solver

"""
    SR2(nlp; kwargs...)

A first-order quadratic regularization method for unconstrained optimization.

For advanced usage, first define a `SR2Solver` to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = SR2Solver(nlp)
    solve!(solver, nlp; kwargs...)

# Arguments
- `nlp::AbstractNLPModel{T, V}` is the model to solve, see `NLPModels.jl`.

# Keyword arguments 
- `x::V = nlp.meta.x0`: the initial guess.
- `atol::T = √eps(T)`: absolute tolerance.
- `param.rtol.value::T = √eps(T)`: relative tolerance: algorithm stops when ‖∇f(xᵏ)‖ ≤ atol + param.rtol.value * ‖∇f(x⁰)‖.
- `param.η1.value = eps(T)^(1/4)`, `param.η2.value = T(0.95)`: step acceptance parameters.
- `param.γ1.value = T(1/2)`, `param.γ2.value = 1/param.γ1.value`: regularization update parameters.
- `param.σmin.value = eps(T)`: step parameter for SR2 algorithm.
- `max_eval::Int = -1`: maximum number of evaluation of the objective function.
- `max_time::Float64 = 30.0`: maximum time limit in seconds.
- `param.β.value = T(0) ∈ [0,1]` is the constant in the momentum term. If `param.β.value == 0`, SR2 does not use momentum.
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration.

# Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.

# Callback
The callback is called at each iteration.
The expected signature of the callback is `callback(nlp, solver, stats)`, and its output is ignored.
Changing any of the input arguments will affect the subsequent iterations.
In particular, setting `stats.status = :user` will stop the algorithm.
All relevant information should be available in `nlp` and `solver`.
Notably, you can access, and modify, the following:
- `solver.x`: current iterate;
- `solver.gx`: current gradient;
- `stats`: structure holding the output of the algorithm (`GenericExecutionStats`), which contains, among other things:
  - `stats.dual_feas`: norm of current gradient;
  - `stats.iter`: current iteration counter;
  - `stats.objective`: current objective function value;
  - `stats.status`: current status of the algorithm. Should be `:unknown` unless the algorithm has attained a stopping criterion. Changing this to anything will stop the algorithm, but you should use `:user` to properly indicate the intention.
  - `stats.elapsed_time`: elapsed time in seconds.

# Examples
```jldoctest
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3))
stats = SR2(nlp)

# output

"Execution stats: first-order stationary"
```

```jldoctest
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3))
solver = SR2Solver(nlp);
stats = solve!(solver, nlp)

# output

"Execution stats: first-order stationary"
```
"""


import SolverCore.solve!
import Krylov.solve!
export solve!

mutable struct SR2Solver{T,V} <: AbstractOptimizationSolver
    x::V
    gx::V
    cx::V
    d::V   # used for momentum term
end

function SR2Solver(nlp::AbstractNLPModel{T,V}) where {T,V}
    x = similar(nlp.meta.x0)
    gx = similar(nlp.meta.x0)
    cx = similar(nlp.meta.x0)
    d = fill!(similar(nlp.meta.x0), 0)
    return SR2Solver{T,V}(x, gx, cx, d)
end


@doc (@doc SR2Solver) function SR2(
    nlp::AbstractNLPModel{T,V};
    atol::T = √eps(T),
    rtol::T = √eps(T),
    η1 = T(eps(T)^(1 / 4)),#TODO check if this is too big
    η2 = T(0.95),
    γ1 = T(1 / 2),
    γ2 = 1 / γ1,
    σmin = zero(T),# change this
    β::T = T(0),
    kwargs...,
) where {T,V}
    solver = SR2Solver(nlp)
    nlp_param = R2ParameterSet{T}(
        atol = atol,
        rtol = rtol,
        η1 = η1,
        η2 = η2,
        γ1 = γ1,
        γ2 = γ2,
        σmin = σmin,
        β = β,
    ) #(√eps(R), √eps(R), 0.1, 0.3, 1.1, 1.9, zero(R), 0.9) # TODO add the param here
    return SolverCore.solve!(solver, nlp_param, nlp; kwargs...)
end

function SolverCore.reset!(solver::SR2Solver{T}) where {T}
    solver.d .= zero(T)
    solver
end
SolverCore.reset!(solver::SR2Solver, ::AbstractNLPModel) = reset!(solver)




#TODO rtol, Rtol  are out ,?


# function SolverCore.solve!(solver::AbstractOptimizationSolver, param::AbstractParameterSet, model::AbstractNLPModel; kwargs...)
#     stats = GenericExecutionStats(model)
#     solve!(solver, param, model, stats; kwargs...)
#   end

function SolverCore.solve!(
    solver::AbstractOptimizationSolver,
    param::AbstractParameterSet,
    model::AbstractNLPModel;
    kwargs...,
)
    stats = GenericExecutionStats(model)
    solve!(solver, param, model, stats; kwargs...)
end



#TODO make param poistional requirment 

function SolverCore.solve!(
    solver::SR2Solver{T,V},
    param::AbstractParameterSet,
    nlp::AbstractNLPModel{T,V},
    stats::GenericExecutionStats{T,V};
    x::V = nlp.meta.x0,
    max_time::Float64 = 30.0,
    max_eval::Int = -1,
    callback = (args...) -> nothing,
    verbose::Int = 0,
) where {T,V}
    unconstrained(nlp) || error("SR2 should only be called on unconstrained problems.")

    reset!(stats)
    start_time = time()
    set_time!(stats, 0.0)

    #TODO change all of these to solver.
    # x = solver.x .= x
    solver.x .= x
    #   ∇fk = solver.gx # todo pointer !!
    # ck = solver.cx
    # d = solver.d

    set_iter!(stats, 0)
    set_objective!(stats, obj(nlp, solver.x))

    grad!(nlp, solver.x, solver.gx)
    norm_∇fk = norm(solver.gx)
    set_dual_residual!(stats, norm_∇fk)

    σk = 2^round(log2(norm_∇fk + 1))

    # Stopping criterion: 
    ϵ = param.atol.value + param.rtol.value * norm_∇fk #TODO 
    optimal = norm_∇fk ≤ ϵ
    if optimal
        @info("Optimal point found at initial point")
        @info @sprintf "%5s  %9s  %7s  %7s " "iter" "f" "‖∇f‖" "σ"
        @info @sprintf "%5d  %9.2e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk σk
    end

    if verbose > 0 && mod(stats.iter, verbose) == 0
        @info @sprintf "%5s  %9s  %7s  %7s  %7s  %7s " "iter" "f" "‖∇f‖" "σ" "ρk" "ΔTk"
        infoline =
            @sprintf "%5d  %9.2e  %7.1e  %7.1e  %7.1e  %7.1e " stats.iter stats.objective norm_∇fk σk 0.0 0.0
    end

    set_status!(
        stats,
        get_status(
            nlp,
            elapsed_time = stats.elapsed_time,
            optimal = optimal,
            max_eval = max_eval,
            max_time = max_time,
        ),
    )

    callback(nlp, solver, stats, param)

    done = stats.status != :unknown
    while !done

        #added by Farhad for Deep learning
        # since we are not updating the solver.x, then grad should say the same but we might have noise 


        #TODO objective re-calculate since we need same x, with new minibatch

        set_objective!(stats, obj(nlp, solver.x))

        grad!(nlp, solver.x, solver.gx)
        norm_∇fk = norm(solver.gx)
        set_dual_residual!(stats, norm_∇fk)

        # σk = 2^round(log2(norm_∇fk + 1))


        if param.β.value == 0
            solver.cx .= solver.x .- (solver.gx ./ σk)
        else
            solver.d .= solver.gx .* (T(1) - param.β.value) .+ solver.d .* param.β.value
            solver.cx .= solver.x .- (d ./ σk)
        end


        ΔTk = norm_∇fk^2 / σk
        fck = obj(nlp, solver.cx)
        if fck == -Inf
            set_status!(stats, :unbounded)
            break
        end

        #TODO objective re-calculate since we need same x, with new minibatch
        # set_objective!(stats, obj(nlp,solver.x))



        ρk = (stats.objective - fck) / ΔTk

        # Update regularization parameters
        if ρk >= param.η2.value
            σk = max(param.σmin.value, param.γ1.value * σk)
        elseif ρk < param.η1.value
            σk = σk * param.γ2.value
        end

        # Acceptance of the new candidate
        if ρk >= param.η1.value
            solver.x .= solver.cx
            set_objective!(stats, fck)
            grad!(nlp, solver.x, solver.gx)
            norm_∇fk = norm(solver.gx)
        end

        set_iter!(stats, stats.iter + 1)
        set_time!(stats, time() - start_time)
        set_dual_residual!(stats, norm_∇fk)


        #TODO for now
        # optimal = norm_∇fk ≤ ϵ
        optimal = false

        if verbose > 0 && mod(stats.iter, verbose) == 0
            @info infoline
            infoline =
                @sprintf "%5d  %9.2e  %7.1e  %7.1e  %7.1e  %7.1e " stats.iter stats.objective norm_∇fk σk ρk ΔTk
            # infoline =
            # @sprintf "%5d  %9.2e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk σk
        end

        set_status!(
            stats,
            get_status(
                nlp,
                elapsed_time = stats.elapsed_time,
                optimal = optimal,
                max_eval = max_eval,
                max_time = max_time,
            ),
        )

        callback(nlp, solver, stats, param)

        # #####
        # # I am forcing it not to stop
        done = stats.status != :unknown
    end

    set_solution!(stats, solver.x)
    if verbose > 0
        @info @sprintf "%s: %s" "stats.status" stats.status
    end
    return stats
end
