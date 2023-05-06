module SLoPP

using DataFrames
using Clustering
using LogicCircuits
using ProbabilisticCircuits
using Random
using BlossomV

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
    elseif z == 0 return StructProbLiteralNode(-X, V)
    # Bernoulli.
    else
      w = z/n
      return StructSumNode([StructProbLiteralNode(X, V), StructProbLiteralNode(-X, V)],
                           log.([w, 1-w]), V)
    end
  end
  # Cluster.
  R = kmeans(transpose(Matrix(D)), k; display = :none)
  # Get indices.
  I = Vector{Vector{Int}}(undef, k)
  for i in 1:k I[i] = Vector{Int}() end
  for i in 1:n push!(I[R.assignments[i]], i) end
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

function evaluate(name::String, k::Integer; seed::Integer = 0, alg::Symbol = :topdown,
    param::Symbol = :none, args...)::Tuple{Float64, Float64, Float64, StructProbCircuit}
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
  print("Evaluating train set...")
  ll_train = log_likelihood_avg(C, train)
  print(" OK!\nEvaluating valid set...")
  ll_valid = log_likelihood_avg(C, valid)
  print(" OK!\nEvaluating test set...")
  ll_test = log_likelihood_avg(C, test)
  println(" OK!")
  return ll_train, ll_valid, ll_test, C
end
export evaluate

end # module SLoPP
