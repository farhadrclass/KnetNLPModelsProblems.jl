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
mutable struct SR2Solver{T,V} <: AbstractOptimizationSolver
    x::V
    gx::V
    cx::V
    d::V   # used for momentum term
    # param::AbstractParameterSet{T} #TODO should I add it here?
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
    η1 = eps(T)^(1 / 4),#TODO check if this is too big
    η2 = T(0.95),
    γ1 = T(1 / 2),
    γ2 = 1 / γ1,
    σmin = zero(T),# change this
    β::T = T(0),
    kwargs...,
) where {T,V}
    solver = SR2Solver(nlp)
    nlp_param = R2ParameterSet{T}(atol, rtol, η1, η2, γ1, γ2, σmin, β) #(√eps(R), √eps(R), 0.1, 0.3, 1.1, 1.9, zero(R), 0.9) # TODO add the param here
    return SolverCore.solve!(solver, nlp; param = nlp_param, kwargs...)
end

function SolverCore.reset!(solver::SR2Solver{T}) where {T}
    solver.d .= zero(T)
    solver
end
SolverCore.reset!(solver::SR2Solver, ::AbstractNLPModel) = reset!(solver)


#TODO rtol, Rtol  are out ,?

# param.atol.value::T = √eps(T),
# param.rtol.value::T = √eps(T),

# param.η1.value = eps(T)^(1 / 4),
# param.η2.value = T(0.95),
# param.γ1.value = T(1 / 2),
# param.γ2.value = 1 / param.γ1.value,
# param.σmin.value = zero(T),
# param.β.value::T = T(0),

function SolverCore.solve!(
    solver::SR2Solver{T,V},
    nlp::AbstractNLPModel{T,V},
    stats::GenericExecutionStats{T,V};
    x::V = nlp.meta.x0,
    param::AbstractParameterSet,#TODO either defult constructor if empty or move it up 
    max_time::Float64 = 30.0,
    max_eval::Int = -1,
    callback = (args...) -> nothing,
    verbose::Int = 0,
) where {T,V}
    unconstrained(nlp) || error("SR2 should only be called on unconstrained problems.")

    reset!(stats)
    start_time = time()
    set_time!(stats, 0.0)

    x = solver.x .= x
    #   ∇fk = solver.gx
    ck = solver.cx
    d = solver.d

    set_iter!(stats, 0)
    set_objective!(stats, obj(nlp, x))

    grad!(nlp, x, solver.gx)
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
        @info @sprintf "%5s  %9s  %7s  %7s %7s %7s %7s %7s" "iter" "f" "‖∇f‖" "σ" "ρk" "ΔTk" "η1" "η2"
        infoline =
            @sprintf "%5d  %9.2e  %7.1e  %7.1e %7.1e %7.1e %7.1e %7.1e" stats.iter stats.objective norm_∇fk σk ρk ΔTk param.η1.value param.η2.value
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

    callback(nlp, solver, stats, nlp_param)

    done = stats.status != :unknown
    while !done
        if param.β.value == 0
            ck .= x .- (solver.gx ./ σk)
        else
            d .= solver.gx .* (T(1) - param.β.value) .+ d .* param.β.value
            ck .= x .- (d ./ σk)
        end


        ΔTk = norm_∇fk^2 / σk

        fck = obj(nlp, ck)


        # ΔTk = fck - (\phi _x )
        if fck == -Inf
            set_status!(stats, :unbounded)
            break
        end

        ρk = (stats.objective - fck) / ΔTk

        # Update regularization parameters
        if ρk >= param.η2.value
            σk = max(param.σmin.value, param.γ1.value * σk)
        elseif ρk < param.η1.value
            σk = σk * param.γ2.value
        end

        # Acceptance of the new candidate
        if ρk >= param.η1.value
            x .= ck
            set_objective!(stats, fck)
            grad!(nlp, x, solver.gx)
            norm_∇fk = norm(solver.gx)
        end

        set_iter!(stats, stats.iter + 1)
        set_time!(stats, time() - start_time)
        set_dual_residual!(stats, norm_∇fk)
        # optimal = norm_∇fk ≤ ϵ
        optimal = false #TODO for now

        if verbose > 0 && mod(stats.iter, verbose) == 0
            @info infoline
            infoline =
                @sprintf "%5d  %9.2e  %7.1e  %7.1e %7.1e %7.1e %7.1e %7.1e" stats.iter stats.objective norm_∇fk σk ρk ΔTk param.η1.value param.η2.value
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

        callback(nlp, solver, stats, nlp_param)
        ###TODO  not sure about this but  , move to cb and add more info
        # set_objective!(stats, obj(nlp, x))
        # grad!(nlp, x, solver.gx)
        # norm_∇fk = norm(solver.gx)
        # set_dual_residual!(stats, norm_∇fk)

        # σk = 2^round(log2(norm_∇fk + 1)) # let's not change

        #####
        # I am forcing it not to stop
        done = stats.status != :unknown
    end

    set_solution!(stats, x)
    return stats
end
