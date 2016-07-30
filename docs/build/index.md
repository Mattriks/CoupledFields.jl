
<a id='CoupledFields.jl-Documentation-1'></a>

# CoupledFields.jl Documentation


<a id='Types-1'></a>

## Types

<a id='CoupledFields.InputSpace' href='#CoupledFields.InputSpace'>#</a>
**`CoupledFields.InputSpace`** &mdash; *Type*.



```
InputSpace: A type to hold the `X` and `Y` fields of the Input space
```

InputSpace(X, Y, d, lat): The fields are whitened if `d=[d1, d2]` is supplied. Area weighting is applied if `lat` is supplied.      

<a id='CoupledFields.ModelObj' href='#CoupledFields.ModelObj'>#</a>
**`CoupledFields.ModelObj`** &mdash; *Type*.



```
ModelObj: A type to hold statistical model results
```

Such as the matrices `W, R, A, T`, where `R=XW` and `T=YA`.  

<a id='CoupledFields.KernelParameters' href='#CoupledFields.KernelParameters'>#</a>
**`CoupledFields.KernelParameters`** &mdash; *Type*.



```
KernelParameters: An abstract type.
```

All KernelParameters types contain Kf and ∇Kf i.e. a kernel function and its first derivative. Some instances also contain other parameters which are later passed to Kf and ∇Kf. 

A KernelParameters type can be set using e.g. `PolynomialKP()` or `PolynomialKP(X::Matrix{Float64})`. The two forms are provided for consistency. 

<a id='CoupledFields.GaussianKP' href='#CoupledFields.GaussianKP'>#</a>
**`CoupledFields.GaussianKP`** &mdash; *Type*.



```
GaussianKP: For the gaussian kernel
```

<a id='CoupledFields.PolynomialKP' href='#CoupledFields.PolynomialKP'>#</a>
**`CoupledFields.PolynomialKP`** &mdash; *Type*.



```
PolynomialKP: For the polynomial kernel
```


<a id='Functions-1'></a>

## Functions

<a id='CoupledFields.bf-Tuple{Array{Float64,1},Int64}' href='#CoupledFields.bf-Tuple{Array{Float64,1},Int64}'>#</a>
**`CoupledFields.bf`** &mdash; *Method*.



```
bf(x::Vector{Float64}, df::Int):
```

Compute a piecewise linear basis matrix for the vector x.

<a id='CoupledFields.cca-Tuple{Array{Float64,N},T<:Array{Float64,2},T<:Array{Float64,2}}' href='#CoupledFields.cca-Tuple{Array{Float64,N},T<:Array{Float64,2},T<:Array{Float64,2}}'>#</a>
**`CoupledFields.cca`** &mdash; *Method*.



```
cca{T<:Matrix{Float64}}(v::Array{Float64}, X::T,Y::T):
```

Regularized Canonical Correlation Analysis using SVD. 

<a id='CoupledFields.gKCCA-Tuple{Array{Float64,N},Array{Float64,2},Array{Float64,2},CoupledFields.KernelParameters}' href='#CoupledFields.gKCCA-Tuple{Array{Float64,N},Array{Float64,2},Array{Float64,2},CoupledFields.KernelParameters}'>#</a>
**`CoupledFields.gKCCA`** &mdash; *Method*.



```
gKCCA(par::Array{Float64}, X::Matrix{Float64}, Y::Matrix{Float64}, kpars::KernelParameters):
```

Compute the projection matrices and components for gKCCA.

<a id='CoupledFields.gradvecfield-Tuple{Array{N<:Float64,N},T<:Array{Float64,2},T<:Array{Float64,2},CoupledFields.KernelParameters}' href='#CoupledFields.gradvecfield-Tuple{Array{N<:Float64,N},T<:Array{Float64,2},T<:Array{Float64,2},CoupledFields.KernelParameters}'>#</a>
**`CoupledFields.gradvecfield`** &mdash; *Method*.



```
gradvecfield{N<:Float64, T<:Matrix{Float64}}(par::Array{N}, X::T, Y::T, kpars::KernelParameters ):
```

Compute the gradient vector or gradient matrix at each instance of the `X` and `Y` fields, by making use of a kernel feature space.

<a id='CoupledFields.whiten-Tuple{Array{Float64,2},Float64}' href='#CoupledFields.whiten-Tuple{Array{Float64,2},Float64}'>#</a>
**`CoupledFields.whiten`** &mdash; *Method*.



```
whiten(x::Matrix{Float64}, d::Float64; lat=nothing): Whiten matrix
```

`d` (0-1) Percentage variance of components to retain. 

`lat` Latitudinal area-weighting.

