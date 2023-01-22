module CoupledFields

using LinearAlgebra
using Statistics
using StatsBase: sample, zscore
export InputSpace, ModelObj
export KernelParameters, GaussianKP, PolynomialKP
export gradvecfield, gradient
export bf, cca, gKCCA
export CVfn, Rsq_adj
export whiten, zscore
include("MLKernels.jl")

"""
    KernelParameters

An abstract type. \n
All KernelParameters types contain certain parameters which are later passed to internal functions `Kf` and `∇Kf`. \n
A KernelParameters type is set using e.g. `PolynomialKP(X::Matrix{Float64})` or `GaussianKP(X::Matrix{Float64})`. 
"""
abstract type KernelParameters end

"""
    GaussianKP(X)
    
For the gaussian kernel.
"""
struct GaussianKP <: KernelParameters
    xx::Matrix{Float64}
    varx::Float64
end

function GaussianKP(X::Matrix{Float64})
    xx = kernelmatrix(SquaredDistanceKernel(1.0), X, X)
    varx = Statistics.median(xx[xx.>10.0^-9])
    return GaussianKP(xx, varx)    
end


    function Kf(par::Array{Float64}, X::Matrix{Float64}, kpars::GaussianKP)
        sx2 = 2*par[1]*par[1]*kpars.varx
        return exp.(-kpars.xx/sx2)
    end

    function ∇Kf(par::Array{Float64}, X::Matrix{Float64}, kpars::GaussianKP)
        n,p = size(X)
        sx2 = 2*par[1]*par[1]*kpars.varx
        Gx = Kf(par, X, kpars)
        ∇K = Float64[2.0*Gx[i,k]*(X[i,j]-X[k,j])/sx2  for i in 1:n, k in 1:n, j in 1:p ]
        return ∇K 
    end


"""
    PolynomialKP(X)

For the polynomial kernel.
"""
struct PolynomialKP <: KernelParameters 
    xx::Matrix{Float64}    
    function PolynomialKP(X::Matrix{Float64})
        xx = kernelmatrix(LinearKernel(1.0, 1.0), X, X)
        return new(xx)
    end
end


    function Kf(par::Array{Float64}, X::Matrix{Float64}, kpars::PolynomialKP)
        return (kpars.xx).^par[1]
    end

    function ∇Kf(par::Array{Float64}, X::Matrix{Float64}, kpars::PolynomialKP)
        n,p = size(X)
        m = par[1]
        ∇K = Float64[m*X[k,j]*kpars.xx[i,k]^(m-1.0)  for i in 1:n, k in 1:n, j in 1:p ]
        return ∇K 
    end



"""
    ModelObj(W, R, A, T, evals, pars, method)
    
A type to hold statistical model results, such as the matrices `W, R, A, T`, where `R=XW` and `T=YA`.  
"""
struct ModelObj
    W::Matrix{Float64}
    R::Matrix{Float64}
    A::Matrix{Float64}
    T::Matrix{Float64}
    evals::Vector{Float64}
    pars::Array{Float64}
    method::String
end



"""
    InputSpace(X, Y, d, lat)

A type to hold the `X` and `Y` fields of the Input space.  
The fields are whitened if `d=[d1, d2]` is supplied.  
Area weighting is applied if `lat` is supplied.
"""
struct InputSpace
    X::Matrix{Float64} 
    Y::Matrix{Float64}
    function InputSpace(a::T, b::T) where T<:Matrix{Float64}
        new(zscore(a,1), zscore(b,1))
    end
    function InputSpace(a::T, b::T, d::Vector{Float64}) where T<:Matrix{Float64}
        new(whiten(a, d[1]), whiten(b, d[2]))
    end
    function InputSpace(a::T, b::T, d::V, lat::V) where {V<:Matrix{Float64}, T<:Vector{Float64}}
        new(whiten(a, d[1], lat=lat), whiten(b, d[2], lat=lat))
    end
end



"""
    gradvecfield(par::Array, X::Matrix, Y::Matrix, kpars::KernelParameters)

Compute the gradient vector or gradient matrix at each instance of the `X` and `Y` fields, by making use of a kernel feature space.
"""
function gradvecfield(par::Array{Float64}, X::Matrix{Float64}, Y::Matrix{Float64}, kpars::KernelParameters)
    n,p = size(X)
    Gx = Kf(par, X, kpars) + (10.0^par[2])*n*I
    ∇K = ∇Kf(par, X, kpars)
    return [∇K[i,:,:]' * (Gx \ Y) for i in 1:n]
end



"""
    bf(x::Vector, df::Int)

Compute a piecewise linear basis matrix for the vector x.
"""
function bf(x::Vector{Float64}, df::Int)
    if df<2 return x[:,1:1] end
    bp = Statistics.quantile(x, range(0, stop=1, length=df+1))
    n = length(x)
    a1 = repeat(x, inner=(1,df)) 
    a2 = repeat( permutedims(bp[1:df]), inner=(n,1)) 
      
    for j in 1:df
        ix1, ix2 = x.<bp[j], x.>bp[j+1] 
        a1[ix1,j] .= bp[j] 
        a1[ix2,j] .= bp[j+1] 
    end
        
    return a1-a2 
end

"""
    cca(v::Array, X::Matrix, Y::Matrix)

Regularized Canonical Correlation Analysis using SVD. 
"""
function cca(v::Array{Float64}, X::Matrix{Float64}, Y::Matrix{Float64})
#    n,p = size(X)
    q = size(Y, 2)
    
    Cxx_fac_inv = (Statistics.cov(X)+(10.0^v[1])*I)^-0.5
    Cyy_fac_inv = (q>1) ? (Statistics.cov(Y)+(10.0^v[2])*I)^-0.5 : Statistics.cov(Y)^(-0.5)
    M = Statistics.cov(X*Cxx_fac_inv, Y*Cyy_fac_inv)
    U, D, V = svd(M)

    Wx = Cxx_fac_inv * U
    Wy = Cyy_fac_inv * V
    Tx = X * Wx
    Ty = Y * Wy
    
    L = D.^2
    return ModelObj(Wx, Tx, Wy, Ty, L, v, "CCA")
end


cca(v::Array{Float64}, X::Matrix{Float64}, Y::Matrix{Float64}, kpars::KernelParameters) = cca(v, X, Y)


"""
    gKCCA(par::Array, X::Matrix, Y::Matrix, kpars::KernelParameters)

Compute the projection matrices and components for gKCCA.
"""
function gKCCA(par::Array{Float64}, X::Matrix{Float64}, Y::Matrix{Float64}, kpars::KernelParameters)
    q = size(Y,2)
    n, p = size(X)
    ∇g = gradvecfield(par, X, Y, kpars)
    M = mapreduce(x -> x*x', +, ∇g)/n
    values, vectors = eigen(Symmetric(M))
    Wx = reverse(vectors, dims=2)
    R = X * Wx
    
    dmin = minimum([p q 3])
    Ay = zeros(q, dmin)
    T = zeros(n, dmin)
    for j in 1:dmin
        ϕRj = bf(R[:,j], Int(par[3]))
        cc = cca([-9.0 -9.0], ϕRj, Y)
        Ay[:,j] = cc.A[:,1]
        T[:,j] = cc.T[:,1]
    end
    
    return ModelObj(Wx, R, Ay, T, reverse(values), par, "gKCCA")
end




"""
    whiten(X::Matrix, d::Float64; lat=nothing)
 
Whiten `X`.  \n
`d` (0-1) Percentage variance of components to retain. \n
`lat` Latitudinal area-weighting.
"""
function whiten(X::Matrix{Float64}, d::Float64; lat=nothing)
    m1 = zscore(X, 1)
#    lat != nothing && scale!(m1, 1.0./sqrt(cos(pi*lat/180)) )
    U,S,V = svd(m1)
    l = cumsum(S.^2)/sum(abs2, S)
    U = U[:, l.≤d]
    U ./= Statistics.std(U, dims=1)
    return U
end


"""
    CVfn(parm::Matrix, X::Matrix, Y::Matrix, modelfn::Function, kerneltype::DataType; verbose=true, dcv=2)

Cross-validation function
"""
function CVfn(parm::Matrix{Float64}, X::Matrix{Float64}, Y::Matrix{Float64}, modelfn::Function, kerneltype::DataType; verbose::Bool=true, dcv::Int64=2)
    
    # free parameters
    # df = par[3] # Number of segments in PWLM
    # dcv = 2 # Number of components used in CV
    
    # other params
    nrg = size(parm, 1)
    pargrid = Float64[parm zeros(nrg)]
    n, p = size(X)
     
    # Setup
    yr = div(0:n-1,12)+1
    yrs = unique(yr)
    ta = findall(in(sample(yrs, div(length(yrs),2)+1, replace=false, ordered=true)), yr)
    tb = setdiff(1:n,ta)

    
    xm1, xm2 = mean(X[ta,:], dims=1), mean(X[tb,:], dims=1)
    ym1, ym2 = mean(Y[ta,:], dims=1), mean(Y[tb,:], dims=1)
    
    X1train, Y1train = X[ta,:].-xm1, Y[ta,:].-ym1 
    X2train, Y2train = X[tb,:].-xm2, Y[tb,:].-ym2
    X1test, Y1test = X[ta,:].-xm2, Y[ta,:].-ym2 
    X2test, Y2test = X[tb,:].-xm1, Y[tb,:].-ym1 
    
    k1pars = kerneltype(X1train)
    k2pars = kerneltype(X2train)
    
    # Main loop
    
    for i in 1:nrg
        percent = round(Int, 100*i/nrg)
        if verbose
           print(string(percent,"% ")) 
        end
        par = parm[i,:]
        model1 = modelfn(par, X1train, Y1train, k1pars)
        model2 = modelfn(par, X2train, Y2train, k2pars)
        W1, W2 = model1.W[:,1:dcv], model2.W[:,1:dcv]        
        A1, A2 = model1.A[:,1:dcv], model2.A[:,1:dcv]
        W1 .*= sign.(W1[1:1,:])
        W2 .*= sign.(W2[1:1,:])
        A1 .*= sign.(A1[1:1,:])
        A2 .*= sign.(A2[1:1,:])
        Rcv = [X2test * W1; X1test * W2]
        Tcv = [Y2test * A1; Y1test * A2]
        pargrid[i,end] = Rsq_adj(Rcv, Tcv, Int(par[3]))
    end
    
    imax = argmax(pargrid[:,end])
    return pargrid[imax,:]
end

"""
    Rsq_adj(Tx::Array, Ty::Array, df::Int)

Cross-validation metric
"""
function Rsq_adj(Tx::Array{Float64}, Ty::Array{Float64}, df::Int)
    y = vec(Ty)
    n,p = size(Tx)
    o = ones(n*p)
    mL = [ bf(Tx[:,j], df) for j in 1:p  ]
#    mL = [ [ bf(Tx[:,j], df) o] for j in 1:p  ]
    Xs =  [ cat([1,2], mL...) o]   # Block diagonal
    px = p*df
    nx = p*n
    Σyhat = (y'*Xs / (Xs' * Xs) ) * (Xs' * y)
    Rsq = ((Σyhat) / (y'y))[1]
    return Rsq - (1-Rsq)*px/(nx-px-1)
end    


include("misc.jl")


end