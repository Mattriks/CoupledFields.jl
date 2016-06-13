
<a id='CoupledFields.jl-Documentation-1'></a>

# CoupledFields.jl Documentation


<a id='Types-and-Functions-1'></a>

## Types and Functions

<a id='CoupledFields.ModelParameters' href='#CoupledFields.ModelParameters'>#</a>
**`CoupledFields.ModelParameters`** &mdash; *Type*.



ModelParameters(xx::Matrix{Float64}, varx::Float64): A ModelParameters object

<a id='CoupledFields.distm' href='#CoupledFields.distm'>#</a>
**`CoupledFields.distm`** &mdash; *Function*.



distm(X::Matrix{Float64}): Returns distance matrix and median distance

<a id='CoupledFields.gradvecfield' href='#CoupledFields.gradvecfield'>#</a>
**`CoupledFields.gradvecfield`** &mdash; *Function*.



gradvecfield{N<:Float64, T<:Matrix{Float64}}(par::Array{N}, X::T, Y::T, modelpars::ModelParameters ): Returns a gradient vector or gradient matrix at each instance of the X and Y fields, by making use of a kernel feature space. There are two parameters in par: the gaussian parameter σ and the regularization parameter ϵ.


<a id='Example-1'></a>

## Example


This example requires the Gadfly branch [Geom_segment](https://github.com/Mattriks/Gadfly.jl/tree/Geom_segment) for plotting the vectorfield. Note that `CoupledFields` by itself does not require this branch. 


```julia
using StatsBase: zscore
using DataFrames, Gadfly
using CoupledFields

function rHypersphere(n::Int, k::Int)
    Q = qrfact(randn(k,k))[:Q]
    return Q[:,1:n]  
end

function simfn(n::Int, p::Int,sc::Float64, sige::Float64)
    Wx = rHypersphere(2,p)
    Wy = rHypersphere(2,2)
    X = sc*rand(n,p)-(sc/2)
    E = sige*randn(n,1)
    xstar = X * Wx
    ystar = zscore([6.3*xstar[:,1].*exp(-0.1*xstar[:,1].^2) randn(n,1)],1)
    Y =  ystar / Wy
    return zscore(X,1), Y, xstar, ystar
end

function createDF{T<:Matrix{Float64}}(c::Float64, X::T, Y::T, modelpars::ModelParameters)
    ∇g = hcat(gradvecfield([c -7.0], X, Y, modelpars)...)'
    sc = 0.05
    vecf = [X-∇g*sc X+∇g*sc] 
    DataFrame(x=X[:,1], y=X[:,2], x1=vecf[:,1], y1=vecf[:,2], x2=vecf[:,3],y2=vecf[:,4], col=Y[:,1],
        par="σ=$c"*"σ<sub>med</sub>")
end

srand(1234)
X, Y, xstar, ystar = simfn(200, 2,30.0, 0.1)
mpars = distm(X)

D1 = vcat([createDF(c, X, ystar[:,1:1], mpars) for c in [0.5 0.05]  ]...) 

colscale = Scale.color_continuous(minvalue=-2.0, maxvalue=2.0)
coord = Coord.cartesian(xmin=-2.0, xmax=2.0, ymin=-2.0, ymax=2.0)

p = plot(D1,  xgroup=:par,
    Geom.subplot_grid(coord,
        layer(x=:x1, y=:y1,xend=:x2, yend=:y2, color=:col, Geom.vector),
        layer(x=:x, y=:y, color=:col, Geom.point, Theme(default_point_size=2pt)) 
    ),
    colscale,
    Guide.xlabel("X<sub>1</sub>"),
    Guide.ylabel("X<sub>2</sub>"),
    Guide.colorkey("Y")
)

```


![](Fig_vecfield.png)

