module SLoPP

using DataFrames
using Clustering
using LogicCircuits
using ProbabilisticCircuits
using Random
using BlossomV
using Metis

function factorize(D::AbstractDataFrame, V::Vtree)::StructProbCircuit
  if isleaf(V)
    X = Lit(variable(V))
    if D[1, 1] return StructProbLiteralNode(X, V) end
    return StructProbLiteralNode(-X, V)
  end
  return StructMulNode(factorize(view(D, :, Symbol.(variables(V.left))), V.left),
                       factorize(view(D, :, Symbol.(variables(V.right))), V.right), V)
end

function _slopp(D::AbstractDataFrame, V::Vtree, k::Integer)::StructProbCircuit
  n, m = size(D)
  # Leaf distribution node.
  if isleaf(V)
    X = Lit(variable(V))
    # Invariant: D is a matrix of the form n × 1.
    z = mapcols(sum, D)[1, 1]
    # Positive literal.
    if z == n return StructProbLiteralNode(X, V)
    # Negative literal.
    elseif z == 0 return StructProbLiteralNode(-X, V) end
    # Bernoulli.
    w = z/n
    return StructSumNode([StructProbLiteralNode(X, V), StructProbLiteralNode(-X, V)],
                         log.([w, 1-w]), V)
  # If there is only one example, consider it as a conjunction over all assignments.
  elseif n == 1 return factorize(D, V) end
  # Cluster.
  R = kmeans(transpose(Matrix(D)), k; display = :none, maxiter = 100)
  # If all instances are the same, return a conjunction over all assignments.
  if any(x -> x == 0, R.counts) return factorize(D, V) end
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
    S_p = view(D, I[i], Symbol.(variables(V.left)))
    S_s = view(D, I[i], Symbol.(variables(V.right)))
    # Prime.
    p = _slopp(S_p, V.left, k)
    # Sub.
    s = _slopp(S_s, V.right, k)
    # Element.
    Ch[i] = StructMulNode(p, s, V)
  end
  # Set weights as proportions.
  W = log.(R.counts/n)
  return StructSumNode(Ch, W, V)
end

function slopp(D::AbstractDataFrame, k::Integer; V::Union{Vtree, Nothing} = nothing,
    seed::Integer = 0, alg::Symbol = :topdown)
  if isnothing(V) V = learn_vtree(D; alg) end
  Random.seed!(seed)
  return _slopp(D, V, k)
end
export slopp

function revise!(name::String, k::Integer, C::StructProbCircuit; args...)
  print("Pulling dataset...")
  train, valid, test = twenty_datasets(name)
  println(" OK!")
  return revise!(D, k, C, size(train, 1); args...)
end
function revise!(D::AbstractDataFrame, k::Integer, C::StructProbCircuit, n_original::Int;
    seed::Integer = 0, alg::Symbol = :clt)
  if seed >= 0 Random.seed!(seed) end
  rename!(D, Symbol.(1:size(D, 2)))
  D_v = view(D, findall(isinf, log_likelihood_per_instance(C, D)), :)
  if size(D_v, 1) == 0 return C end
  Z = _slopp(D_v, vtree(C), k)
  n = size(D_v, 1)
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
  rename!(train, Symbol.(1:size(train, 2)))
  print(" OK!\nLearning the structure...")
  C = slopp(train, k; seed, alg)
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
  rename!(train, Symbol.(1:size(train, 2)))
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
  ll_train, train_ign = ev(C, train), count(isinf, log_likelihood_avg(C, train))
  print(" OK!\nEvaluating valid set...")
  ll_valid, valid_ign = ev(C, valid), count(isinf, log_likelihood_avg(C, valid))
  print(" OK!\nEvaluating test set...")
  ll_test, test_ign = ev(C, test), count(isinf, log_likelihood_avg(C, valid))
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
