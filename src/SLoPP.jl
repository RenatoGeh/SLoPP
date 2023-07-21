module SLoPP

using DataFrames
using Clustering
using LogicCircuits
using ProbabilisticCircuits
using Random
using BlossomV
using Metis

@inline function mapleaves(n::Integer, V::Vtree)::Vector{StructProbLiteralNode}
  # Create all leaves before hand.
  L_V = Dict(variable(v) => v for v ∈ leafnodes(V))
  L = Vector{StructProbLiteralNode}(undef, n << 1)
  for i ∈ 1:n
    x = Lit(i)
    L[i] = StructProbLiteralNode(x, L_V[i])
    L[n+i] = StructProbLiteralNode(-x, L_V[i])
  end
  return L
end

@inline function getleaf(X::Lit, L::Vector{StructProbLiteralNode})::StructProbLiteralNode
  # If X is positive, then index is X, otherwise index is |L|+X.
  return L[(X > 0)*X + (X < 0)*(length(L)÷2 - X)]
end

function factorize(D::AbstractDataFrame, V::Vtree, L::Vector{StructProbLiteralNode})::StructProbCircuit
  if isleaf(V)
    X = Lit(variable(V))
    if D[1, 1] return getleaf(X, L) end
    return getleaf(-X, L)
  end
  return StructMulNode(factorize(view(D, :, Symbol.(variables(V.left))), V.left, L),
                       factorize(view(D, :, Symbol.(variables(V.right))), V.right, L), V)
end

function factorize(D::AbstractDataFrame, n::Integer, V::Vtree, L::Vector{StructProbLiteralNode})::StructProbCircuit
  # D is a mapcols(sum, data).
  if isleaf(V)
    X = Lit(variable(V))
    w = D[1, Symbol(X)]/n
    return StructSumNode([getleaf(X, L), getleaf(-X, L)],
                         log.([w, 1-w]), V)
  end
  return StructMulNode(factorize(D, n, V.left, L), factorize(D, n, V.right, L), V)
end

function partition(D::AbstractDataFrame, X::Symbol, Y::Vector{Symbol})::Tuple{AbstractDataFrame, AbstractDataFrame}
  I, J = Vector{Int}(), Vector{Int}()
  for i ∈ 1:nrow(D)
    if D[i, X] push!(J, i) else push!(I, i) end
  end
  return view(D, I, Y), view(D, J, Y)
end

function singleton_prime(D::AbstractDataFrame, X::Lit, Y::Vector{Symbol}, V::Vtree,
    k::Integer, umin::Integer, citer::Integer, L::Vector{StructProbLiteralNode})::StructProbCircuit
  s_X = Symbol(X)
  D_X = view(D, :, s_X)
  # Partition into X=true and X=false instances.
  D_0, D_1 = partition(D, s_X, Y)
  if nrow(D_1) == nrow(D)
    s = _slopp(D_1, V.right, k, umin, citer, L)
    return StructMulNode(getleaf(X, L), s, V)
  elseif nrow(D_0) == nrow(D)
    s = _slopp(D_0, V.right, k, umin, citer, L)
    return StructMulNode(getleaf(-X, L), s, V)
  end
  w = nrow(D_1)/nrow(D)
  s_0 = _slopp(D_0, V.right, k, umin, citer, L)
  s_1 = _slopp(D_1, V.right, k, umin, citer, L)
  return StructSumNode([StructMulNode(getleaf(X, L), s_1, V),
                        StructMulNode(getleaf(-X, L), s_0, V)], log.([w, 1-w]), V)
end

function _slopp(D::AbstractDataFrame, V::Vtree, k::Integer, umin::Integer, citer::Integer,
    L::Vector{StructProbLiteralNode})::StructProbCircuit
  n, m = size(D)
  # Leaf distribution node.
  if isleaf(V)
    X = Lit(variable(V))
    # Invariant: D is a matrix of the form n × 1.
    z = mapcols(sum, D)[1, 1]
    # Positive literal.
    if z == n return getleaf(X, L)
    # Negative literal.
    elseif z == 0 return getleaf(-X, L) end
    # Bernoulli.
    w = z/n
    return StructSumNode([getleaf(X, L), getleaf(-X, L)], log.([w, 1-w]), V)
  end
  # If the scope of the left hand side is a single variable, then just add a sum node.
  X, Y = collect(Lit, variables(V.left)), collect(Lit, variables(V.right))
  s_Y = Symbol.(Y)
  if length(X) == 1 return singleton_prime(D, first(X), s_Y, V, k, umin, citer, L) end
  # If we have reached the minimum number of unique examples in the data.
  #if nrow(unique(D; view = true)) < umin return factorize(mapcols(sum, D), nrow(D), V, L)
  s_X = Symbol.(X)
  if nrow(unique(D; view = true)) < umin
    p = _slopp(view(D, :, s_X), V.left, k, umin, citer, L)
    s = _slopp(view(D, :, s_Y), V.right, k, umin, citer, L)
    return StructMulNode(p, s, V)
  # If there is only one example, consider it as a conjunction over all assignments.
  elseif umin <= 0 && n == 1 return factorize(D, V, L) end
  # Cluster.
  R = kmeans(transpose(Matrix(view(D, :, s_X))), k; display = :none, maxiter = citer)
  # If all instances are the same, return a conjunction over all assignments.
  if (umin > 0) && any(x -> x == 0, R.counts) return factorize(D, V, L) end
  # Get indices.
  I = Vector{Vector{Int}}(undef, k)
  incr = zeros(UInt32, k)
  for i in 1:k I[i] = Vector{Int}(undef, R.counts[i]) end
  for i in 1:n
    j = R.assignments[i]
    I[j][incr[j] += 1] = i
  end
  # Elements (conjunction of prime and sub).
  Ch = Vector{StructMulNode}(undef, k)
  # Recurse over clusters.
  for i in 1:k
    S_p = view(D, I[i], s_X)
    S_s = view(D, I[i], s_Y)
    # Prime.
    p = _slopp(S_p, V.left, k, umin, citer, L)
    # Sub.
    s = _slopp(S_s, V.right, k, umin, citer, L)
    # Element.
    Ch[i] = StructMulNode(p, s, V)
  end
  # Set weights as proportions.
  W = log.(R.counts/n)
  return StructSumNode(Ch, W, V)
end

function slopp(D::AbstractDataFrame, k::Integer; V::Union{Vtree, Nothing} = nothing,
    seed::Integer = 0, alg::Symbol = :topdown, umin::Integer = 200, citer::Integer = 200)
  if isnothing(V) V = learn_vtree(D; alg) end
  Random.seed!(seed)
  L = mapleaves(ncol(D), V)
  return _slopp(D, V, k, umin, citer, L)
end
export slopp

function revise!(name::String, k::Integer, C::StructProbCircuit; args...)
  print("Pulling dataset...")
  train, valid, test = twenty_datasets(name)
  println(" OK!")
  return revise!(D, k, C, nrow(train); args...)
end
function revise!(D::AbstractDataFrame, k::Integer, C::StructProbCircuit, n_original::Int;
    seed::Integer = 0, umin::Integer = 200, citer::Integer = 200)
  if seed >= 0 Random.seed!(seed) end
  rename!(D, Symbol.(1:ncol(D)))
  D_v = view(D, findall(isinf, log_likelihood_per_instance(C, D)), :)
  if nrow(D_v) == 0 return C end
  L = mapleaves(ncol(D), V)
  Z = _slopp(D_v, vtree(C), k, umin, citer, L)
  n = nrow(D_v)
  N = n+n_original
  proportions = Vector{Float64}(undef, length(C.children)+length(Z.children))
  proportions[1:length(C.children)] .= (n_original .* exp.(C.log_probs)) ./ N
  proportions[length(C.children)+1:end] .= (n .* exp.(Z.log_probs)) ./ N
  C.log_probs = log.(proportions)
  append!(C.children, Z.children)
  return C
end
export revise!

function slopp_from_data(name::String, k::Integer; seed::Integer = 1234, alg::Symbol = :topdown,
    param::Symbol = :none, args...)::StructProbCircuit
  print("Pulling dataset...")
  train, valid, test = twenty_datasets(name)
  rename!(train, Symbol.(1:ncol(train)))
  print(" OK!\nLearning the structure...")
  C = slopp(train, k; seed, alg, args...)
  println(" OK!")
  if param != :none
    print("Learning the parameters...")
    if param == :em estimate_parameters_em!(C, train; args...)
    elseif param == :sgd estimate_parameters_sgd!(C, train; args...) end
    println(" OK!")
  end
  return C
end
export slopp_from_data

function evaluate(name::String, C::StructProbCircuit; ignore::Bool = true)::Tuple{Float64, Float64, Float64, Float64, Float64, Float64}
  print("Pulling dataset...")
  train, valid, test = twenty_datasets(name)
  function ev(circ, data)
    if ignore
      ll = log_likelihood_per_instance(circ, data)
      n = length(ll)-count(isinf, ll)
      return sum(filter(x -> !isinf(x), ll))/n
    end
    return log_likelihood_avg(circ, data)
  end
  println(" OK!")
  print("Evaluating train set...")
  ll_train, train_ign = ev(C, train), count(isinf, log_likelihood_per_instance(C, train))
  print(" OK!\nEvaluating valid set...")
  ll_valid, valid_ign = ev(C, valid), count(isinf, log_likelihood_per_instance(C, valid))
  print(" OK!\nEvaluating test set...")
  ll_test, test_ign = ev(C, test), count(isinf, log_likelihood_per_instance(C, valid))
  println(" OK!")
  return ll_train, ll_valid, ll_test, train_ign, valid_ign, test_ign
end
function evaluate(name::String, k::Integer; seed::Integer = 1234, alg::Symbol = :clt,
    param::Symbol = :none, ignore::Bool = true, args...)::Tuple{Float64, Float64, Float64, StructProbCircuit, Float64, Float64, Float64}
  C = slopp_from_data(name, k; seed, alg, param, args...)
  ll_train, ll_valid, ll_test, train_ign, valid_ign, test_ign = evaluate(name, C; ignore)
  return ll_train, ll_valid, ll_test, C, train_ign, valid_ign, test_ign
end
export evaluate

end # module SLoPP
