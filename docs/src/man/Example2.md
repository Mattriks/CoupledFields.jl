# Example 2

```@example
#=
using DataFrames, Compose, Gadfly
using CoupledFields, LinearAlgebra, Random, Statistics
set_default_graphic_size(21cm, 14cm)


function rHypersphere(n::Int, k::Int)
    Q = qr(randn(k,k)).Q
    return Q[:,1:n]  
end

function simfn(n::Int, p::Int, sc::Float64)
    Wx = rHypersphere(2,p)
    Wy = rHypersphere(2,2)
    X = sc*rand(n,p) .- 0.5*sc
    xstar = X*Wx
    ystar = zscore([6.3*xstar[:,1].*exp.(-0.1*xstar[:,1].^2) randn(n)],1)
    Y =  ystar / Wy
    return zscore(X,1), Y, xstar, ystar
end

createDF = function(df::Int, R::Vector, Y::Matrix{Float64})
    Xs = bf(R, df)
    CCAm = cca([-9. -9.], Xs, Y)
    return DataFrame(x=R, y= CCAm.T[:,1], y2 = CCAm.R[:,1].-mean(CCAm.R[:,1]), df="df=$df")
end    

Random.seed!(1234)
X, Y, xstar, ystar = simfn(200, 2, 30.0)

kpars = GaussianKP(X)
gKCCAm = gKCCA([0.4, -5, 1], X, Y, kpars )

plotfn = function(v)
    mlfs = 10pt
    D1= vcat([createDF(df, gKCCAm.R[:,1], Y) for df in v]...)
    
plot(D1, xgroup=:df,
    Geom.subplot_grid(Coord.cartesian(ymin=-3, ymax=3),
        layer(x=:x, y=:y2, Geom.line, Theme(default_color=colorant"red")),
        layer(x=:x, y=:y, Geom.point)
    ),
    Guide.ylabel("YA"),
    Theme(plot_padding=[0mm], major_label_font_size=mlfs)
 )
end

pb = plotfn(4:6)
push!(pb, Guide.xlabel("XW (gKCCA)" ))

compose(context(),
    (context(0, 0, 1.0, 0.45), render(plotfn(1:3))),
    (context(0, 0.45, 1.0, 0.55), render(pb))
)
=#
```


