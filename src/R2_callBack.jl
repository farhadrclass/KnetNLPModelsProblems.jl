export R2, R2Solver

"""
    R2(nlp; kwargs...)
    solver = R2Solver(nlp;)
    solve!(solver, nlp; kwargs...)

    A first-order quadratic regularization method for unconstrained optimization

# Arguments
- `nlp::AbstractNLPModel{T, V}` is the model to solve, see `NLPModels.jl`.

# Keyword arguments 
- x0::V = nlp.meta.x0`: the initial guess
- atol = eps(T)^(1 / 3): absolute tolerance
- rtol = eps(T)^(1 / 3): relative tolerance: algorithm stop when ||∇f(x)|| ≤ ϵ_abs + ϵ_rel*||∇f(x0)||
- η1 = eps(T)^(1/4), η2 = T(0.95): step acceptance parameters
- γ1 = T(1/2), γ2 = 1/γ1: regularization update parameters
- σmin = eps(T): step parameter for R2 algorithm
- max_eval::Int: maximum number of evaluation of the objective function
- max_time::Float64 = 3600.0: maximum time limit in seconds
- verbose::Bool = false: prints iteration details if true.

# Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.

## Callback
The callback is called after each iteration.
The expected signature of the callback is `(nlp, solver)`, and its output is ignored.
Notice that changing any of the input arguments will affect the subsequent iterations.
In particular, setting `solver.output.status = :user` will stop the algorithm.
All relevant information should be available in `nlp` and `solver`.
Notably, you can access, and modify, the following:
- `solver.x`: current iterate.
- `solver.gx`: current gradient.
- `solver.output`: structure holding the output of the algorithm (`GenericExecutionStats`), which contains, among other things:
  - `solver.output.dual_feas`: norm of current gradient.
  - `solver.output.iter`: current iteration counter.
  - `solver.output.objective`: current objective function value.
  - `solver.output.status`: current status of the algorithm. Should be `:unknown` unless the algorithm has found a stopping criteria. Changing this to anything will stop the algorithm, but you should use `:user` to properly indicate the intention.
  - `solver.output.elapsed_time`: elapsed time in seconds.
# Examples
```jldoctest
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3))
stats = R2(nlp)
# output
"Execution stats: first-order stationary"
```
```jldoctest
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3))
solver = R2Solver(nlp);
stats = solve!(solver, nlp)
# output
"Execution stats: first-order stationary"

```
"""

# TODO: Move to SolverCore
function get_status(
  nlp;
  elapsed_time = 0.0,
  optimal = false,
  max_eval = Inf,
  max_time = Inf,
)
  if optimal
    :first_order
  elseif neval_obj(nlp) > max_eval ≥ 0
    :max_eval
  elseif elapsed_time > max_time
    :max_time
  else
    :unknown
  end
end


mutable struct R2Solver{V <: AbstractVector{T}} <: AbstractOptSolver{T, V}
  x::V
  gx::V
  cx::V
  output::GenericExecutionStats{T, V}
end

function R2Solver(nlp::AbstractNLPModel{T, V}) where {T, V<: AbstractVector{T}}
  x = similar(nlp.meta.x0)
  gx = similar(nlp.meta.x0)
  cx = similar(nlp.meta.x0)
  output = GenericExecutionStats(:unknown, nlp, solution = x)
  return R2Solver{V}(x, gx, cx, output)
end

@doc (@doc R2Solver) function R2(
  nlp::AbstractNLPModel{T,V};
  kwargs...,
) where {T, V}
  solver = R2Solver(nlp)
  return solve!(solver, nlp; kwargs...)
end

function solve!(
    solver::R2Solver{V},
    nlp::AbstractNLPModel{T, V};
    callback = (args...) -> nothing,
    x0::V = nlp.meta.x0,
    atol = eps(T)^(1/3),
    rtol = eps(T)^(1/3),
    η1 = eps(T)^(1/4),
    η2 = T(0.95),
    γ1 = T(1/2),
    γ2 = 1/γ1,
    σmin = zero(T), #todo add max iteration
    max_time::Float64 = 3600.0,
    max_eval::Int = -1,
    verbose::Bool = true,
    max_iter = -1,
    user_gamma = 0.9 , #### Momentum
  ) where {T, V}

  
  unconstrained(nlp) || error("R2 should only be called on unconstrained problems.")
  output = solver.output

  start_time = time()
  output.elapsed_time = 0.0
  gamma = T(user_gamma) #type change for defence , should we add this to output?

  
  x = solver.x .= x0
  ∇fk = solver.gx
  ck = solver.cx
  d =  similar(∇fk) * 0  # used for Momentum, start with zeros

  output.iter = 1
  output.objective = obj(nlp, x)
  
  grad!(nlp, x, ∇fk)
  norm_∇fk = norm(∇fk)
  # σk = norm(hess(nlp, x))
  σk = 2^round(log2(norm_∇fk + 1))


  # Stopping criterion: 
  ϵ = atol + rtol * norm_∇fk
  optimal = norm_∇fk ≤ ϵ
  if optimal
    @info("Optimal point found at initial point")
    @info @sprintf "%5s  %9s  %7s  %7s " "iter" "f" "‖∇f‖" "σ"
    @info @sprintf "%5d  %9.2e  %7.1e  %7.1e" output.iter output.objective norm_∇fk σk
  end
  tired = neval_obj(nlp) > max_eval ≥ 0 || elapsed_time > max_time || output.iter > max_iter ≥ 0
  if verbose
    @info @sprintf "%5s  %9s  %7s  %7s " "iter" "f" "‖∇f‖" "σ"
    infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e" output.iter output.objective norm_∇fk σk
  end

  status = :unknown

  # while !(optimal | tired)
  while !done

    d .= ∇fk .* (T(1) - gamma) + d .* gamma   
    ck .= x .- (d ./ σk)
    # ck .= x .- (∇fk ./ σk)
    ΔTk= norm_∇fk^2 / σk
    fck = obj(nlp, ck)
    if fck == -Inf
      output.status = :unbounded
      break
    end

    ρk = (output.objective - fck) / ΔTk 

    # Update regularization parameters
    if ρk >= η2
      σk = max(σmin, γ1 * σk)
    elseif ρk < η1
      σk = σk * γ2
    end

    # Acceptance of the new candidate
    if ρk >= η1
      x .= ck
      output.objective = fck
      grad!(nlp, x, ∇fk)
      norm_∇fk = norm(∇fk)
    end

    output.iter += 1
    output.elapsed_time = time() - start_time
    optimal = norm_∇fk ≤ ϵ
    # tired = neval_obj(nlp) > max_eval ≥ 0 || elapsed_time > max_time || output.iter > max_iter ≥ 0
  
    if verbose
      @info infoline
      infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e" output.iter output.objective norm_∇fk σk
    end

    output.status = get_status(
      nlp,
      elapsed_time = output.elapsed_time,
      optimal = optimal,
      max_eval = max_eval,
      max_time = max_time,
    )

    callback(nlp, solver)

    done = output.status != :unknown


  end
    
  output.dual_feas = norm_∇fk
  # output.solution = x # don't need this since we already include it 

  return output
end


### EXAMPLE OF CALL BACK

# @testset "Callback" begin
#   nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])

#   cb = (nlp, solver) -> begin  # we can update minibatch and epochs here?
#     if solver.output.iter == 7
#       solver.output.status = :user
#     end
#   end

#   @testset "Solver $solver" for solver in [R2]
#     reset!(nlp)
#     output = solver(nlp)
#     @test output.iter > 7

#     reset!(nlp)
#     output = solver(nlp, callback=cb)
#     @test output.iter == 7
#   end
# end