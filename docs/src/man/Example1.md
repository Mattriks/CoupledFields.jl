# Example 1

```@example
using CoupledFields, DataFrames, Gadfly
using LinearAlgebra, Random
set_default_plot_size(21cm,12cm)

function rHypersphere(n::Int, k::Int)
    Q = qr(randn(k,k)).Q
    return Q[:,1:n]
end

function simfn(n::Int, p::Int, sc::Float64)
    Wx = rHypersphere(2,p)
    Wy = rHypersphere(2,2)
    X = sc*rand(n,p) .- 0.5*sc
    xstar = X * Wx
    ystar = zscore([6.3*xstar[:,1].*exp.(-0.1*xstar[:,1].^2) randn(n)],1)
    Y =  ystar / Wy
    return zscore(X,1), Y, xstar, ystar
end

function createDF(c::Float64, X::Matrix, Y::Matrix, kpars::KernelParameters; sc=1.0)
    ∇g = hcat(gradvecfield([c -7.0], X, Y, kpars)...)'
    vecf = [X-∇g*sc X+∇g*sc] 
    DataFrame(x=X[:,1], y=X[:,2], x1=vecf[:,1], y1=vecf[:,2], x2=vecf[:,3],
        y2=vecf[:,4], col=Y[:,1], par="σ = $c"*"σ<sub>med</sub>")
end

Random.seed!(1234)
X, Y, xstar, ystar = simfn(200, 2, 30.0)

kpars = GaussianKP(X)
D = vcat([createDF(c, X, ystar[:,1:1], kpars, sc=0.05) for c in [0.5 0.05]]...)


colscale = Scale.color_continuous(minvalue=-2.0, maxvalue=2.0)
coord = Coord.cartesian(xmin=-2.0, xmax=2.0, ymin=-2.0, ymax=2.0)

plot(D, x=:x, y=:y, color=:col,  xgroup=:par,
    Geom.subplot_grid(coord,
        layer(x=:x1, y=:y1, xend=:x2, yend=:y2, color=:col, Geom.vector),
        layer(Geom.point, Theme(point_size=2pt)) 
    ), colscale,
    Scale.x_continuous(minvalue=-2.0, maxvalue=2.0),
    Scale.y_continuous(minvalue=-2.0, maxvalue=2.0),
    Guide.xlabel("<b>X₁</b>"),
    Guide.ylabel("<b>X₂</b>"),
    Guide.colorkey(title="Y")
)
```

