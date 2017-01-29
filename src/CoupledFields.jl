module CoupledFields

using StatsBase: zscore, sample
export InputSpace, ModelObj
export KernelParameters, GaussianKP, PolynomialKP
export gradvecfield
export bf, cca, gKCCA
export CVfn, Rsq_adj
export whiten
include("MLKernels.jl")

"""
    KernelParameters: An abstract type.
All KernelParameters types contain certain parameters which are later passed to internal functions `Kf` and `∇Kf`. \n
A KernelParameters type is set using e.g. `PolynomialKP(X::Matrix{Float64})` or `GaussianKP(X::Matrix{Float64})`. 
"""
abstract KernelParameters

"""
    GaussianKP: For the gaussian kernel
"""
type GaussianKP <: KernelParameters
    xx::Matrix{Float64}
    varx::Float64
end

function GaussianKP(X::Matrix{Float64})
    xx = kernelmatrix(SquaredDistanceKernel(1.0), X, X)
    varx = median(xx[xx.>10.0^-9])
    return GaussianKP(xx, varx)    
end


    function Kf{T<:Float64}(par::Array{T}, X::Matrix{T}, kpars::GaussianKP)
        sx2 = 2*par[1]*par[1]*kpars.varx
        return exp(-kpars.xx/sx2)
    end

    function ∇Kf{T<:Float64}(par::Array{T}, X::Matrix{T}, kpars::GaussianKP)
        n,p = size(X)
        sx2 = 2*par[1]*par[1]*kpars.varx
        Gx = Kf(par, X, kpars)
        ∇K = Float64[2.0*Gx[i,k]*(X[i,j]-X[k,j])/sx2  for i in 1:n, k in 1:n, j in 1:p ]
        return ∇K 
    end


"""
    PolynomialKP: For the polynomial kernel 
"""
type PolynomialKP <: KernelParameters 
    xx::Matrix{Float64}    
    function PolynomialKP(X::Matrix{Float64})
        xx = kernelmatrix(LinearKernel(1.0, 1.0), X, X)
        return new(xx)
    end
end


    function Kf{T<:Float64}(par::Array{T}, X::Matrix{T}, kpars::PolynomialKP)
        return (kpars.xx).^par[1]
    end

    function ∇Kf{T<:Float64}(par::Array{T}, X::Matrix{T}, kpars::PolynomialKP)
        n,p = size(X)
        m = par[1]
        ∇K = Float64[m*X[k,j]*kpars.xx[i,k]^(m-1.0)  for i in 1:n, k in 1:n, j in 1:p ]
        return ∇K 
    end



"""
    ModelObj: A type to hold statistical model results
Such as the matrices `W, R, A, T`, where `R=XW` and `T=YA`.  
"""
type ModelObj
    W::Matrix{Float64}
    R::Matrix{Float64}
    A::Matrix{Float64}
    T::Matrix{Float64}
    evals::Vector{Float64}
    pars::Array{Float64}
    method::String
end



"""
    InputSpace: A type to hold the `X` and `Y` fields of the Input space
InputSpace(X, Y, d, lat): The fields are whitened if `d=[d1, d2]` is supplied.
Area weighting is applied if `lat` is supplied.      
"""
type InputSpace
    X::Matrix{Float64} 
    Y::Matrix{Float64}
    function InputSpace{T<:Matrix{Float64}}(a::T, b::T)
        new(zscore(a,1), zscore(b,1))
    end
    function InputSpace{T<:Matrix{Float64}}(a::T, b::T, d::Vector{Float64})
        new(whiten(a, d[1]), whiten(b, d[2]))
    end
    function InputSpace{T<:Matrix{Float64},V<:Vector{Float64}}(a::T, b::T, d::V, lat::V)
        new(whiten(a, d[1], lat=lat), whiten(b, d[2], lat=lat))
    end
end



"""
    gradvecfield{N<:Float64, T<:Matrix{Float64}}(par::Array{N}, X::T, Y::T, kpars::KernelParameters ):
Compute the gradient vector or gradient matrix at each instance of the `X` and `Y` fields, by making use of a kernel feature space.
"""
function gradvecfield{N<:Float64, T<:Matrix{Float64}}(par::Array{N}, X::T, Y::T, kpars::KernelParameters )
    n,p = size(X)
    Ix = (10.0^par[2])*n*eye(n)
    Gx = Kf(par, X, kpars)
    ∇K = ∇Kf(par, X, kpars)
    return [∇K[i,:,:]' * ((Gx+Ix) \ Y) for i in 1:n]
end



"""
    bf(x::Vector{Float64}, df::Int):
Compute a piecewise linear basis matrix for the vector x.
"""
function bf(x::Vector{Float64}, df::Int)
    if df<2 return x[:,1:1] end
    bp = quantile(x, linspace(0, 1, df+1))
    n = length(x)
    a1 = repmat(x,1,df) 
    a2 = repmat( bp[1:df]', n, 1) 
      
    for j in 1:df
        ix1 = x.<bp[j]; a1[ix1,j] = bp[j] 
        ix2 = x.>bp[j+1]; a1[ix2,j] = bp[j+1] 
    end
        
    return a1-a2 
end

"""
    cca{T<:Matrix{Float64}}(v::Array{Float64}, X::T,Y::T):
Regularized Canonical Correlation Analysis using SVD. 
"""
function cca{T<:Matrix{Float64}}(v::Array{Float64}, X::T,Y::T)
    n,p = size(X)
    q = size(Y)[2]
    Ix = (10.0^v[1])*eye(p)
    Iy = (10.0^v[2])*eye(q)
    
    Cxx_fac_inv = (cov(X)+Ix)^-0.5
    Cyy_fac_inv= if q>1 (cov(Y)+Iy)^-0.5 
        else cov(Y)^(-0.5)
    end
    M = cov(X*Cxx_fac_inv, Y*Cyy_fac_inv)
    U, D, V = svd(M)

    Wx = Cxx_fac_inv * U
    Wy = Cyy_fac_inv * V
    Tx = X * Wx
    Ty = Y * Wy
    
    L = D.^2
    return ModelObj(Wx, Tx, Wy, Ty, L, v, "CCA")
end


cca{T<:Matrix{Float64}}(v::Array{Float64}, X::T,Y::T, kpars::KernelParameters) = cca(v, X, Y)


"""
    gKCCA(par::Array{Float64}, X::Matrix{Float64}, Y::Matrix{Float64}, kpars::KernelParameters):
Compute the projection matrices and components for gKCCA.
"""
function gKCCA(par::Array{Float64}, X::Matrix{Float64}, Y::Matrix{Float64}, kpars::KernelParameters)
    q = size(Y,2)
    n, p = size(X)
    ∇g = gradvecfield(par, X, Y, kpars)
    M = mapreduce(x -> x*x', +, ∇g)/n
    ev = eigfact(Symmetric(M))
    Wx = flipdim(ev.vectors, 2)
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
    
    return ModelObj(Wx, R, Ay, T, flipdim(ev.values,1), par, "gKCCA")
end




"""
    whiten(x::Matrix{Float64}, d::Float64; lat=nothing): Whiten matrix
`d` (0-1) Percentage variance of components to retain. \n
`lat` Latitudinal area-weighting.
"""
function whiten(X::Matrix{Float64}, d::Float64; lat=nothing)
    m1 = zscore(X,1)
    if lat != nothing
        scale!(m1, 1.0./sqrt(cos(pi*lat/180)) )
    end
    sv = svdfact(m1)
    l = cumsum(sv[:S].^2)/sumabs2(sv[:S])
    U = sv[:U][:, l.≤d]
    U /= std(U[:,1])
    return U
end 


"""
    CVfn{T<:Matrix{Float64}}(parm::T, X::T, Y::T, modelfn::Function, kerneltype::DataType; verbose::Bool=true, dcv::Int64=2)
Cross-validation function
"""
function CVfn{T<:Matrix{Float64}}(parm::T, X::T, Y::T, modelfn::Function, kerneltype::DataType; verbose::Bool=true, dcv::Int64=2)
    
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
    ta = findin(yr, sample(yrs,div(length(yrs),2)+1,replace=false,ordered=true))
    tb = setdiff(1:n,ta)

    
    xm1 = mean(X[ta,:],1); xm2 = mean(X[tb,:],1)
    ym1 = mean(Y[ta,:],1); ym2 = mean(Y[tb,:],1)
    
    X1train = X[ta,:].-xm1; Y1train = Y[ta,:].-ym1 
    X2train = X[tb,:].-xm2; Y2train = Y[tb,:].-ym2
    X1test = X[ta,:].-xm2; Y1test = Y[ta,:].-ym2 
    X2test = X[tb,:].-xm1; Y2test = Y[tb,:].-ym1 
    
    k1pars = kerneltype(X1train)
    k2pars = kerneltype(X2train)
    
    # Main loop
    
    for i in 1:nrg
        percent = round(100*i/nrg)
        if verbose
           @printf " %03d%%" percent
        end
        par = parm[i,:]
        model1 = modelfn(par, X1train, Y1train, k1pars)
        model2 = modelfn(par, X2train, Y2train, k2pars)
        W1 = model1.W[:,1:dcv]; W2 = model2.W[:,1:dcv]        
        A1 = model1.A[:,1:dcv]; A2 = model2.A[:,1:dcv]
        scale!(W1, vec(sign(W1[1,:])))
        scale!(W2, vec(sign(W2[1,:])))
        scale!(A1, vec(sign(A1[1,:])))
        scale!(A2, vec(sign(A2[1,:])))
        Rcv = [X2test * W1; X1test * W2]
        Tcv = [Y2test * A1; Y1test * A2]
        pargrid[i,end] = Rsq_adj(Rcv, Tcv, Int(par[3]))
    end
    
    imax = indmax(pargrid[:,end])
    return pargrid[imax,:]
end

"""
    Rsq_adj{T<:Array{Float64}}(Tx::T, Ty::T, df::Int): 
Cross-validation metric
"""
function Rsq_adj{T<:Array{Float64}}(Tx::T, Ty::T, df::Int)
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




end