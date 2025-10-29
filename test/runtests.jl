using Test, Random
using DelvesLynessBeyn

Random.seed!(0)

function compare(expecteds, results; rtol=sqrt(eps()))
  corrects = zeros(Bool, length(expecteds))
  for result in results, (i, expected) in enumerate(expecteds)
    if isapprox(result, expected, rtol=rtol)
      corrects[i] = true
    end
  end
  return corrects
end

function setup(nzeros, npoles)
  expectedzeros = sort(randn(ComplexF64, nzeros), by=abs)
  expectedpoles = sort(randn(ComplexF64, npoles), by=abs)
  realradius = 1.5 * maximum(abs, vcat(expectedpoles, expectedzeros))
  imagradius = 1.5 * maximum(abs, vcat(expectedpoles, expectedzeros))
  f(z) = prod((z - r) for r in expectedzeros) / prod((z - p) for p in expectedpoles; init=1.0+0im)
  return expectedzeros, expectedpoles, realradius, imagradius, f
end

@testset "DelvesLynessBeyn.jl" begin
@testset "Single pass" begin
  for npoles in (0, 1, 2, 4), nzeros in (1, 2, 3, 4), N in (64, 128, 256)
    @testset "nzeros = $nzeros, npoles = $npoles, N = $N" begin
      expectedzeros, expectedpoles, realradius, imagradius, f = setup(nzeros, npoles)
           
      res = delveslynessbeyn(f; N=N, centre=0+0im, realradius=realradius, imagradius=imagradius, npoles=npoles)
      @test all(compare(expectedzeros, res.zeros; rtol=1e-3))
      npoles > 0 && @test all(compare(expectedpoles, res.poles; rtol=1e-3))
    end
  end
end
@testset "Adaptive" begin
  for npoles in 0:1, nzeros in 1:6
    @testset "nzeros = $nzeros, npoles = $npoles" begin
      expectedzeros, expectedpoles, realradius, imagradius, f = setup(nzeros, npoles)
      res = delveslynessbeyn(f; rtol=sqrt(eps()), N=8, centre=0+0im, realradius=realradius, imagradius=imagradius, npoles=npoles, maxiters=14)
      @test all(compare(expectedzeros, res.zeros; rtol=1e-6))
      npoles > 0 && @test all(compare(expectedpoles, res.poles; rtol=1e-6))
    end
  end
end
end
 
