
#==========================================================================
 Functions from Julia package MLKernels
 Unfortunately MLKernels hasn't been updated yet to Julia 0.5
 Hence I've borrowed some functions from MLKernels here
==========================================================================#

export kernelmatrix, SquaredDistanceKernel, PolynomialKernel

abstract Kernel{T}

abstract StandardKernel{T<:AbstractFloat} <: Kernel{T}
abstract BaseKernel{T<:AbstractFloat} <: StandardKernel{T}
abstract CompositeKernel{T<:AbstractFloat} <: StandardKernel{T}


#==========================================================================
  Additive Kernel: k(x,y) = sum(k(x_i,y_i))    x ∈ ℝⁿ, y ∈ ℝⁿ
  Separable Kernel: k(x,y) = k(x)k(y)    x ∈ ℝ, y ∈ ℝ
==========================================================================#

abstract AdditiveKernel{T<:AbstractFloat} <: BaseKernel{T}
abstract SeparableKernel{T<:AbstractFloat} <: AdditiveKernel{T}

phi{T<:AbstractFloat}(κ::SeparableKernel{T}, x::T, y::T) = phi(κ,x) * phi(κ,y)


#==========================================================================
  Squared Distance Kernel
  k(x,y) = (x-y)²ᵗ    x ∈ ℝ, y ∈ ℝ, t ∈ (0,1]
==========================================================================#

immutable SquaredDistanceKernel{T<:AbstractFloat,CASE} <: AdditiveKernel{T} 
    t::T
    function SquaredDistanceKernel(t::T)
        0 < t <= 1 || error("Parameter t = $(t) must be in range (0,1]")
        new(t)
    end
end
function SquaredDistanceKernel{T<:AbstractFloat}(t::T = 1.0)
    CASE =  if t == 1
                :t1
            elseif t == 0.5
                :t0p5
            else
                :∅
            end
    SquaredDistanceKernel{T,CASE}(t)
end

@inline phi{T<:AbstractFloat}(κ::SquaredDistanceKernel{T,:t1}, x::Vector{T}, y::Vector{T}) = sumabs2(x-y)
@inline phi{T<:AbstractFloat}(κ::SquaredDistanceKernel{T,:t0p5}, x::Vector{T}, y::Vector{T}) = sumabs(x-y)
@inline phi{T<:AbstractFloat}(κ::SquaredDistanceKernel{T}, x::Vector{T}, y::Vector{T}) = sumabs2(x-y)^κ.t


#==========================================================================
  Scalar Product Kernel
  (Code preserved for future reasons)
==========================================================================#

immutable ScalarProductKernel{T<:AbstractFloat} <: SeparableKernel{T} end
ScalarProductKernel() = ScalarProductKernel{Float64}()

ismercer(::ScalarProductKernel) = true

convert{T<:AbstractFloat}(::Type{ScalarProductKernel{T}}, κ::ScalarProductKernel) = ScalarProductKernel{T}()

@inline phi{T<:AbstractFloat}(κ::ScalarProductKernel{T}, x::T) = x

convert{T<:AbstractFloat}(::Type{Kernel{T}}, κ::ScalarProductKernel) = convert(ScalarProductKernel{T}, κ)


#==========================================================================
  Polynomial Kernel
==========================================================================#

immutable PolynomialKernel{T<:AbstractFloat,CASE} <: CompositeKernel{T}
    k::BaseKernel{T}
    alpha::T
    c::T
    d::T
    function PolynomialKernel(κ::BaseKernel{T}, α::T, c::T, d::T)
        ismercer(κ) == true || error("Composed kernel must be a Mercer kernel.")
        α > 0 || error("α = $(α) must be greater than zero.")
        c >= 0 || error("c = $(c) must be non-negative.")
        (d > 0 && trunc(d) == d) || error("d = $(d) must be an integer greater than zero.")
        if CASE == :d1 && d != 1
            error("Special case d = 1 flagged but d = $(convert(Int64,d))")
        end
        new(κ, α, c, d)
    end
end

PolynomialKernel{T<:AbstractFloat}(κ::BaseKernel{T}, α::T = one(T), c::T = one(T), d::T = convert(T, 2)) = PolynomialKernel{T, d == 1 ? :d1 : :Ø}(κ, α, c, d)
PolynomialKernel{T<:AbstractFloat}(α::T = 1.0, c::T = one(T), d::T = convert(T, 2)) = PolynomialKernel(convert(Kernel{T},ScalarProductKernel()), α, c, d)


@inline phi{T<:AbstractFloat}(κ::PolynomialKernel{T}, x::Vector{T}, y::Vector{T}) = (κ.alpha*dot(x,y) + κ.c)^κ.d
@inline phi{T<:AbstractFloat}(κ::PolynomialKernel{T,:d1}, x::Vector{T}, y::Vector{T}) = κ.alpha*dot(x,y) + κ.c



#==========================================================================
  Generic Kernel Matrix Functions
==========================================================================#

function init_pairwise{T<:AbstractFloat}(X::Matrix{T}, Y::Matrix{T}, is_trans::Bool)
    n_dim = is_trans ? 2 : 1
    n = size(X, n_dim)
    m = size(Y, n_dim)
    Array(T, n, m)
end


function kernelmatrix{T<:AbstractFloat}(κ::Kernel{T}, X::Matrix{T}, Y::Matrix{T}, is_trans::Bool = false)
    kernelmatrix!(init_pairwise(X, Y, is_trans), κ, X, Y, is_trans)
end
# kernelmatrix{T<:AbstractFloat}(κ::Kernel{T}, X::Matrix{T}, Y::Matrix{T}; is_trans::Bool = false) = kernelmatrix(κ, X, Y, is_trans)


#==========================================================================
  Base and Composite Kernel Matrix Functions
==========================================================================#

function kernelmatrix!{T<:AbstractFloat}(K::Matrix{T}, κ::BaseKernel{T}, X::Matrix{T}, Y::Matrix{T}, is_trans::Bool)
    pairwise!(K, κ, X, Y, is_trans)
end

function kernelmatrix!{T<:AbstractFloat}(K::Matrix{T}, κ::CompositeKernel{T}, X::Matrix{T}, Y::Matrix{T}, is_trans::Bool)
    pairwise!(K, κ, X, Y, is_trans)
end


#===================================================================================================
  Default Pairwise Computation
===================================================================================================#

# Initiate pairwise matrices

function init_pairwise{T<:AbstractFloat}(X::Matrix{T}, is_trans::Bool)
    n = size(X, is_trans ? 2 : 1)
    Array(T, n, n)
end

# Pairwise definition

function pairwise!{T<:AbstractFloat}(K::Matrix{T}, κ::Kernel{T}, X::Matrix{T}, Y::Matrix{T}, is_trans::Bool)
    if is_trans
        pairwise_XtYt!(K, κ, X, Y)
    else
        pairwise_XY!(K, κ, X, Y)
    end
end



function pairwise_XY!{T<:AbstractFloat}(K::Matrix{T}, κ::Kernel{T}, X::Matrix{T}, Y::Matrix{T})
    (n = size(X,1)) == size(K,1) || throw(DimensionMismatch("Dimension 1 of X must match dimension 1 of K."))
    (m = size(Y,1)) == size(K,2) || throw(DimensionMismatch("Dimension 1 of Y must match dimension 2 of K."))
    size(X,2) == size(Y,2) || throw(DimensionMismatch("Dimension 2 of Y must match dimension 2 of X."))
    for j = 1:m
        y = Y[j,:]
        for i = 1:n
            K[i,j] = phi(κ, X[i,:], y)
        end
    end
    K
end


function pairwise_XtYt!{T<:AbstractFloat}(K::Matrix{T}, κ::Kernel{T}, X::Matrix{T}, Y::Matrix{T})
    (n = size(X,2)) == size(K,1) || throw(DimensionMismatch("Dimension 2 of X must match dimension 1 of K."))
    (m = size(Y,2)) == size(K,2) || throw(DimensionMismatch("Dimension 2 of Y must match dimension 2 of K."))
    size(X,1) == size(Y,1) || throw(DimensionMismatch("Dimension 1 of Y must match dimension 1 of X."))
    for j = 1:m
        y = Y[:,j]
        for i = 1:n
            K[i,j] = phi(κ, X[:,i], y)
        end
    end
    K
end



