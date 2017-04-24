
#==========================================================================
 Functions from Julia package MLKernels.jl
 Unfortunately MLKernels hasn't tagged a new release for a while
 Hence I've borrowed (and modified) some functions from vesion 0.1.0
 of MLKernels here, see
  https://github.com/trthatcher/MLKernels.jl/blob/master/LICENSE.md:
"The MIT License (MIT)
Copyright (c) 2015 Tim Thatcher
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."
==========================================================================#

export Kernel
export kernelmatrix, SquaredDistanceKernel, PolynomialKernel, LinearKernel
export GaussianKernel

@compat abstract type Kernel{T} end

@compat abstract type StandardKernel{T<:AbstractFloat} <: Kernel{T} end
@compat abstract type BaseKernel{T<:AbstractFloat} <: StandardKernel{T} end
@compat abstract type CompositeKernel{T<:AbstractFloat} <: StandardKernel{T} end


#==========================================================================
  Additive Kernel: k(x,y) = sum(k(x_i,y_i))    x ∈ ℝⁿ, y ∈ ℝⁿ
  Separable Kernel: k(x,y) = k(x)k(y)    x ∈ ℝ, y ∈ ℝ
==========================================================================#

@compat abstract type AdditiveKernel{T<:AbstractFloat} <: BaseKernel{T} end
@compat abstract type SeparableKernel{T<:AbstractFloat} <: AdditiveKernel{T} end

phi{T<:AbstractFloat}(κ::SeparableKernel{T}, x::T, y::T) = phi(κ,x) * phi(κ,y)
isnonnegative(κ::Kernel) = kernelrange(κ) == :Rp


#==========================================================================
  Squared Distance Kernel
  k(x,y) = (x-y)²ᵗ    x ∈ ℝ, y ∈ ℝ, t ∈ (0,1]
==========================================================================#

immutable SquaredDistanceKernel{T<:AbstractFloat,CASE} <: AdditiveKernel{T} 
    t::T
    function (::Type{SquaredDistanceKernel{T,CASE}}){T,CASE}(t::T)
        0 < t <= 1 || error("Parameter t = $(t) must be in range (0,1]")
        new{T,CASE}(t)
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

isnegdef(::SquaredDistanceKernel) = true
kernelrange(::SquaredDistanceKernel) = :Rp

phi{T<:AbstractFloat}(κ::SquaredDistanceKernel{T,:t1}, x::Vector{T}, y::Vector{T}) = sum(abs2, x-y)
phi{T<:AbstractFloat}(κ::SquaredDistanceKernel{T,:t0p5}, x::Vector{T}, y::Vector{T}) = sum(abs, x-y)
phi{T<:AbstractFloat}(κ::SquaredDistanceKernel{T}, x::Vector{T}, y::Vector{T}) = sum(abs2, x-y)^κ.t


#==========================================================================
  Scalar Product Kernel
  (Code preserved for future reasons)
==========================================================================#

immutable ScalarProductKernel{T<:AbstractFloat} <: SeparableKernel{T} end
ScalarProductKernel() = ScalarProductKernel{Float64}()

ismercer(::ScalarProductKernel) = true

convert{T<:AbstractFloat}(::Type{ScalarProductKernel{T}}, κ::ScalarProductKernel) = ScalarProductKernel{T}()

phi{T<:AbstractFloat}(κ::ScalarProductKernel{T}, x::T) = x

convert{T<:AbstractFloat}(::Type{Kernel{T}}, κ::ScalarProductKernel) = convert(ScalarProductKernel{T}, κ)


#==========================================================================
  Polynomial Kernel
==========================================================================#

immutable PolynomialKernel{T<:AbstractFloat,CASE} <: CompositeKernel{T}
    k::BaseKernel{T}
    alpha::T
    c::T
    d::T
    function (::Type{PolynomialKernel{T,CASE}}){T,CASE}(κ::BaseKernel{T}, α::T, c::T, d::T)
        ismercer(κ) == true || error("Composed kernel must be a Mercer kernel.")
        α > 0 || error("α = $(α) must be greater than zero.")
        c >= 0 || error("c = $(c) must be non-negative.")
        (d > 0 && trunc(d) == d) || error("d = $(d) must be an integer greater than zero.")
        if CASE == :d1 && d != 1
            error("Special case d = 1 flagged but d = $(convert(Int64,d))")
        end
        new{T,CASE}(κ, α, c, d)
    end
end

PolynomialKernel{T<:AbstractFloat}(κ::BaseKernel{T}, α::T = one(T), c::T = one(T), d::T = convert(T, 2)) = PolynomialKernel{T, d == 1 ? :d1 : :Ø}(κ, α, c, d)
PolynomialKernel{T<:AbstractFloat}(α::T = 1.0, c::T = one(T), d::T = convert(T, 2)) = PolynomialKernel(convert(Kernel{T},ScalarProductKernel()), α, c, d)
LinearKernel{T<:AbstractFloat}(α::T = 1.0, c::T = one(T)) = PolynomialKernel(ScalarProductKernel(), α, c, one(T))

phi{T<:AbstractFloat}(κ::PolynomialKernel{T}, x::Vector{T}, y::Vector{T}) = (κ.alpha*dot(x,y) + κ.c)^κ.d
phi{T<:AbstractFloat}(κ::PolynomialKernel{T,:d1}, x::Vector{T}, y::Vector{T}) = κ.alpha*dot(x,y) + κ.c


#==========================================================================
  Exponential Kernel
==========================================================================#

immutable ExponentialKernel{T<:AbstractFloat,CASE} <: CompositeKernel{T}
    k::BaseKernel{T}
    alpha::T
    gamma::T
    function (::Type{ExponentialKernel{T,CASE}}){T,CASE}(κ::BaseKernel{T}, α::T, γ::T)
        isnegdef(κ) == true || error("Composed kernel must be negative definite.")
        isnonnegative(κ) || error("Composed kernel must attain only non-negative values.")
        α > 0 || error("α = $(α) must be greater than zero.")
        0 < γ <= 1 || error("γ = $(γ) must be in the interval (0,1].")
        if CASE == :γ1 &&  γ != 1
            error("Special case γ = 1 flagged but γ = $(γ)")
        end
        new{T,CASE}(κ, α, γ)
    end
end
ExponentialKernel{T<:AbstractFloat}(κ::BaseKernel{T}, α::T = one(T), γ::T = one(T)) = ExponentialKernel{T, γ == 1 ? :γ1 : :Ø}(κ, α, γ)
ExponentialKernel{T<:AbstractFloat}(α::T = 1.0, γ::T = one(T)) = ExponentialKernel(convert(Kernel{T}, SquaredDistanceKernel()), α, γ)

GaussianKernel{T<:AbstractFloat}(α::T = 1.0) = ExponentialKernel(SquaredDistanceKernel(one(T)), α)
RadialBasisKernel{T<:AbstractFloat}(α::T = 1.0) = ExponentialKernel(SquaredDistanceKernel(one(T)),α)
LaplacianKernel{T<:AbstractFloat}(α::T = 1.0) = ExponentialKernel(SquaredDistanceKernel(one(T)),α, convert(T, 0.5))


function convert{T<:AbstractFloat}(::Type{ExponentialKernel{T}}, κ::ExponentialKernel)
    ExponentialKernel(convert(Kernel{T}, κ.k), convert(T, κ.alpha), convert(T, κ.gamma))
end

phi{T<:AbstractFloat}(κ::ExponentialKernel{T}, x::Vector{T}, y::Vector{T}) = exp.(-κ.alpha * sum(abs2, x-y)^κ.gamma)
phi{T<:AbstractFloat}(κ::ExponentialKernel{T,:γ1}, x::Vector{T}, y::Vector{T}) = exp.(-κ.alpha * sum(abs2, x-y))


#==========================================================================
  Generic Kernel Matrix Functions
==========================================================================#

function init_pairwise{T<:AbstractFloat}(X::Matrix{T}, Y::Matrix{T}, is_trans::Bool)
    n_dim = is_trans ? 2 : 1
    n = size(X, n_dim)
    m = size(Y, n_dim)
    return Array{T}(n, m)
end


function kernelmatrix{T<:AbstractFloat}(κ::Kernel{T}, X::Matrix{T}, Y::Matrix{T}, is_trans::Bool = false)
    kernelmatrix!(init_pairwise(X, Y, is_trans), κ, X, Y, is_trans)
end


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



