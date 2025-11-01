module DelvesLynessBeyn

using FFTW, LinearAlgebra

export delveslynessbeyn
const DEFAULT_SVDTOL = 100eps()
"""
    delveslynessbeyn(f::F; N=8, centre, realradius, imagradius, npoles::Int=0, svdtol=1e-12, maxrefinements=10, rtol=Inf, atol=0.0, errornoconvergence=true) where F

Find all the zeros of a function `f`(z) using Delves-Lyness-Beyn method by evaluating the objective function at `N` 
equally spaced positions on a contour with specified real and imaginary radius and centre;
`centre`, `realradius` and `imagradius`. If there are poles, specify the number as `npoles` to aid the algorithm.

If `rtol` is set less than Inf, then the roots will be found via refinement where at each iteration `N` doubles and the
algorithm will stop when the relative (absolute) error is less than rtol (atol). It will error if it does not reach
convergence inside `maxrefinements` iterations unless `errornoconvergence` is set to false.

Arguments:
- `f`: Function to find roots of (f(z) = 0)
- `centre`: Center of contour
- `realradius`: Real radius of contour
- `imagradius`: Imaginary radius of contour
- `N (default=8)`: Initial number of points to sample along contour, which doubles with each refinement
- `npoles (default=0)`: Number of known poles in the contour
- `svdtol (default=100eps())`: SVD truncation tolerance, which acts as a regularization parameter.
- `maxrefinements (default=10)`: Maximum number of refinement iterations
- `rtol (default=Inf)`: Relative tolerance (default is Inf, meaning that no refinement takes place)
- `atol (default=0)`: Absolute tolerance
- `errornoconvergence`: Throw error if no convergence
"""
function delveslynessbeyn(f::F; centre, realradius, imagradius, N=8, npoles::Integer=0, svdtol=DEFAULT_SVDTOL,
    maxrefinements=10, rtol=Inf, atol=0.0, errornoconvergence=true) where F
  t0 = float(centre + realradius + 0im)
  cache = Dict(t0=>f(t0)) 
  lastsols = delveslynessbeyn!(f, cache; N, centre, realradius, imagradius, npoles, svdtol)
  rtol == Inf && return lastsols
  iter = 0
  while iter < maxrefinements
    iter += 1
    nextsols = delveslynessbeyn!(f, cache; N=N*2^iter, centre=centre, realradius=realradius, imagradius=imagradius, npoles=npoles, svdtol=svdtol)
    if length(nextsols.allroots) == length(lastsols.allroots)
      all(isapprox.(nextsols.allroots, lastsols.allroots; rtol=rtol, atol=atol)) && return nextsols
    end
    lastsols = nextsols
  end
  if errornoconvergence
    throw(ErrorException("Delves-Lyness-Beyn did not converge"))
  end
  return lastsols
end

function delveslynessbeyn!(f::F, cache::Dict{K, V}; N, centre, realradius, imagradius, npoles::Integer,
    svdtol=DEFAULT_SVDTOL) where {F, K, V}

  N < 8 && throw(ArgumentError("N too small; 8 is tiny, 64 is sensible"))

  Δt = 2π / N
  t = collect(0:Δt:2π-Δt)

  # parametric ellipse and sampling
  z = @. centre + realradius * cos(t) + im * imagradius * sin(t)

  fvals = [get!(cache, zi, f(zi)) for zi in z]
  minabs, maxabs = extrema(abs, fvals)
  minabs < 10eps() && @warn "f has very small values on contour (min|f| = $minabs). Results may be unstable."
  fvals ./= maxabs

  # FFT differentiation: df/dt
  dfdt = im .* FFTW.fftfreq(N, N) .* fft(fvals) 
  ifft!(dfdt)

  # ratio = (d/dt f) / f
  ratio = dfdt ./ fvals

  # s0 estimate (#zeros - #poles)
  s0 = Δt * sum(ratio) / (2π * im)
  n_est = round(Int, real(s0)) # zeros - poles
  nz_est = n_est + npoles

  # Total number of exponentials L = nz + np
  L = nz_est + npoles
  T = eltype(ratio)
  L <= 0 && return (zeros=T[], poles=T[], allroots=T[], weights=T[])

  # compute moments s_k for k = 0..(2L-1)
  zk = [Δt * sum(z.^k .* ratio) / (2π * im) for k in 0:2L-1]  # zk[k+1] = s_k

  # Build Hankel matrices H0 and H1 (L x L):
  # H0[i,j] = s_{i+j-2}  => zk[i+j-1]
  # H1[i,j] = s_{i+j-1}  => zk[i+j]
  H0 = Matrix{ComplexF64}(undef, L, L)
  H1 = Matrix{ComplexF64}(undef, L, L)
  for i in 1:L, j in 1:L
    H0[i,j] = zk[i + j - 1]
    H1[i,j] = zk[i + j]
  end

  # Solve generalized eigenproblem H1 v = λ H0 v  (matrix-pencil method)
  # This finds λ = r_p (the bases), robust when moments are consistent
  # Use eigen for generalized eigenvalues; handle possible singular H0 by SVD truncation fallback
  allroots, eigvectors = try
    ev = eigen(H1, H0)   # may throw if H0 singular
    ev.values, ev.vectors
  catch err
    # Fall back: small regularization to H0 (Tikhonov-like), then eigen
    H0r = H0 + I(L) * norm(H0) * 100eps()
    ev = eigen(H1, H0r)
    ev.values, ev.vectors
  end
  sort!(allroots, by=imag, rev=true)

  # Solve for residues w: s_k = sum_p w_p * r_p^k for k=0..L-1
  # Build Vandermonde M[k+1, p] = r_p^k
  M = transpose(allroots).^(0:L-1)
  svec = zk[1:L]

  # Solve V * weights = zk[1:L] with SVD-regularized pseudoinverse not weights = M \ zk[1:L]
  U, Sv, Vt = svd(M)
  svmax = maximum(Sv)
  Sinv = Diagonal([(s > svdtol * svmax) ? 1 / s : zero(s) for s in Sv])
  weights = Vector{T}(Vt * (Sinv * (U' * zk[1:L])))

  inds = sortperm(@. abs(angle(weights)))
  zerosinds = inds[1:nz_est]
  polesinds = inds[nz_est + 1:end]

  zeros_list = T[allroots[i] for i in zerosinds]
  poles_list = T[allroots[i] for i in polesinds]

  return (zeros=zeros_list, poles=poles_list, allroots=allroots, weights=weights)
end


end # module DelvesLynessBeyn
