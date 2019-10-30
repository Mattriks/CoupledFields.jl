# CoupledFields.jl

## Summary

A julia package for working with coupled fields. This is work in progress. 
The main function `gradvecfield` calculates the gradient vector or gradient matrix for each instance of the coupled fields.

For 𝒀 = g(𝑿), `CoupledFields.gradvecfield([a b], X, Y, kernelpars)` returns 𝑛 gradient matrices, for 𝑛 random points in 𝑿.
For parameters [𝑎 𝑏]: 𝑎 is a smoothness parameter, and 𝑏 is a ridge parameter.

```julia
using CoupledFields
g(x,y,z) = x * exp(-x^2 - y^2 - z^2)
X = -2 .+ 4*rand(100, 3)
Y = g.(X[:,1], X[:,2], X[:,3])

 kernelpars = GaussianKP(X)
 ∇g = gradvecfield([0.5 -7], X, Y[:,1:1], kernelpars)
```
Also CoupledFields doesn’t require a closed-form function, it can be used if you have only the observed fields 𝑿 and 𝒀.


## Installation

Within Julia:
```julia
using Pkg
julia> Pkg.add("CoupledFields")
```

## Documentation

- [**LATEST**][docs-latest-url]


[docs-latest-url]: https://Mattriks.github.io/CoupledFields.jl/latest
[docs-stable-url]: https://Mattriks.github.io/CoupledFields.jl/stable

