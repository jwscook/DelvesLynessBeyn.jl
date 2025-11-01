![CI](https://github.com/jwscook/DelvesLynessBeyn.jl/workflows/CI/badge.svg)

# DelvesLynessBeyn.jl

Find the zeros and poles (if you know how many there are) of a function within a contour by evaluating only around the contour!

In the example below, all the randomly placed 10 zeros and 2 poles are found by 64 function evaluations. In general, the API takes an `rtol` which then adapatively increases the number of samples until the positions of the zeros and poles has stabilised.

```julia
using DelvesLynessBeyn, Plots, Random

Random.seed!(0)

function setup(nzeros, npoles)
  expectedzeros = sort(randn(ComplexF64, nzeros), by=abs)
  expectedpoles = sort(randn(ComplexF64, npoles), by=abs)
  realradius = 1.1 * maximum(abs, vcat(expectedpoles, expectedzeros))
  imagradius = 1.1 * maximum(abs, vcat(expectedpoles, expectedzeros))
  f(z) = prod((z - r) for r in expectedzeros) / prod((z - p) for p in expectedpoles; init=1.0+0im)
  return expectedzeros, expectedpoles, realradius, imagradius, f
end

function run(;nzeros=10, npoles=2)
  contourpoints = ComplexF64[]
  expectedzeros, expectedpoles, realradius, imagradius, objective = setup(nzeros, npoles)
  function wrappedobjective(x)
    push!(contourpoints, x)
    output = objective(x)
    return output
  end
  solutions = delveslynessbeyn(wrappedobjective; N=64, npoles=npoles, centre=0+0im,
                               realradius=realradius, imagradius=imagradius)
  return contourpoints, solutions, expectedzeros, expectedpoles
end

contourpoints, solutions, expectedzeros, expectedpoles = run()

h = plot()
scatter!(h, real.(contourpoints), imag.(contourpoints), mc=:black, label="Contour")
scatter!(h, real.(expectedzeros), imag.(expectedzeros), m=:+, markersize=5, mc=:blue, label="Expected Zeros")
scatter!(h, real.(expectedpoles), imag.(expectedpoles), m=:+, markersize=5, mc=:red, label="Expected Poles")
scatter!(h, real.(solutions.zeros), imag.(solutions.zeros), markersize=3, mc=:blue, label="Result Zeros")
scatter!(h, real.(solutions.poles), imag.(solutions.poles), markersize=3, mc=:red, label="Result Poles")
plot!(xlabel="Real axis", ylabel="Imaginary axis", legend=:outerright, legendcolumns=1)
savefig("DelvesLynessBeyn.png")
```
<img width="600" height="400" alt="DelvesLynessBeyn" src="https://github.com/user-attachments/assets/655b3f38-786a-4b46-b455-192f6aaa8530" />

