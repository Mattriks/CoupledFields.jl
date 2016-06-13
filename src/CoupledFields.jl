module CoupledFields

using MLKernels

export ModelParameters
export distm, gradvecfield

"""
ModelParameters(xx::Matrix{Float64}, varx::Float64): A ModelParameters object
"""
type ModelParameters 
    xx::Matrix{Float64}
    varx::Float64
end


"""
distm(X::Matrix{Float64}): Returns distance matrix and median distance
"""
function distm(X::Matrix{Float64})
    xx = kernelmatrix(SquaredDistanceKernel(1.0), X, X)
    varx = median(xx[xx.>10.0^-9])
    return ModelParameters(xx, varx)    
end


"""
gradvecfield{N<:Float64, T<:Matrix{Float64}}(par::Array{N}, X::T, Y::T, modelpars::ModelParameters ):
Returns a gradient vector or gradient matrix at each instance of the X and Y fields, by making use of a kernel feature space.
There are two parameters in par: the gaussian parameter σ and the regularization parameter ϵ.
"""
function gradvecfield{N<:Float64, T<:Matrix{Float64}}(par::Array{N}, X::T, Y::T, modelpars::ModelParameters )
    n,p = size(X)
    Ix = (10.0^par[2])*n*eye(n)
    sx2 = 2*par[1]*par[1]*modelpars.varx
    Gx = exp(-modelpars.xx/sx2)
    H = [ 2.0*Gx[i,k]*(X[i,j]-X[k,j])/sx2 for i in 1:n, k in 1:n, j in 1:p ]
    return [squeeze(H[i,:,:],1)' * ((Gx+Ix) \ Y) for i in 1:n]
end


end # module
