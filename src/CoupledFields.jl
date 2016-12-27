module CoupledFields

# using MLKernels
export InputSpace, ModelObj
export KernelParameters, GaussianKP, PolynomialKP
export gradvecfield
export bf, cca, gKCCA
export whiten
 include("MLKernels.jl")

"""
    KernelParameters: An abstract type.
All KernelParameters types contain Kf and ∇Kf i.e. a kernel function and its first derivative. Some instances also contain other parameters which are later passed to Kf and ∇Kf. \n
A KernelParameters type can be set using e.g. `PolynomialKP()` or `PolynomialKP(X::Matrix{Float64})`. The two forms are provided for consistency. 
"""
abstract KernelParameters

"""
    GaussianKP: For the gaussian kernel 
"""
type GaussianKP <: KernelParameters
    xx::Matrix{Float64}
    varx::Float64
    Kf::Function
    ∇Kf::Function
end

function GaussianKP(X::Matrix{Float64})
    xx = kernelmatrix(SquaredDistanceKernel(1.0), X, X)
    varx = median(xx[xx.>10.0^-9])
    function Kf{T<:Float64}(par::Array{T}, X::Matrix{T}, kpars::KernelParameters)
        sx2 = 2*par[1]*par[1]*kpars.varx
        return exp(-kpars.xx/sx2)
    end
    function ∇Kf{T<:Float64}(par::Array{T}, X::Matrix{T}, kpars::KernelParameters)
        n,p = size(X)
        sx2 = 2*par[1]*par[1]*kpars.varx
        Gx = kpars.Kf(par, X, kpars)
        ∇K = Float64[2.0*Gx[i,k]*(X[i,j]-X[k,j])/sx2  for i in 1:n, k in 1:n, j in 1:p ]
        return ∇K 
    end 
    return GaussianKP(xx, varx, Kf, ∇Kf)    
end


"""
    PolynomialKP: For the polynomial kernel 
"""
type PolynomialKP <: KernelParameters
    Kf::Function
    ∇Kf::Function
end


function PolynomialKP()
    function Kf{T<:Float64}(par::Array{T}, X::Matrix{T}, kpars::KernelParameters)
        kernelmatrix(PolynomialKernel(1.0, 1.0, par[1]), X, X)
    end
    function ∇Kf{T<:Float64}(par::Array{T}, X::Matrix{T}, kpars::KernelParameters)
        n,p = size(X)
        m = par[1]
        ∇K = Float64[m*X[k,j]*(vecdot(X[i,:],X[k,:])+1.0).^(m-1.0)  for i in 1:n, k in 1:n, j in 1:p ]
        return ∇K 
    end
    return PolynomialKP(Kf, ∇Kf)
end

PolynomialKP(X::Matrix{Float64}) = PolynomialKP()


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
    Gx = kpars.Kf(par, X, kpars)
    ∇K = kpars.∇Kf(par, X, kpars)
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






end