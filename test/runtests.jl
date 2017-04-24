using StatsBase, CoupledFields
# using Base.Test

# write your own tests here
# @test 1 == 1


function simfields(t::Range, p::Int64, σₑ::Float64)
    n = length(t)
    Xa = sin.(2π*repmat(t, 1, p)/5.0) * diagm(sign.(randn(p)))
    Xb = sin.(2π*repmat(t, 1, p)/2.5+1.5π)* diagm(sign.(randn(p)))
    Xa = StatsBase.zscore(Xa,1) + 0.05*randn(n, p)
    Xb = StatsBase.zscore(Xb,1) + σₑ*randn(n, p)
    return Xa, Xb
end


t1 = (1950+1/24):(1/12):2015
lpfield, hpfield = simfields(t1, 10, 1.0)
Z = InputSpace(lpfield, hpfield, [1.1,1.1])
kpars = GaussianKP(Z.X)
model = gKCCA([1.0 -5.0 2], Z.X, Z.Y, kpars)


