# Auxiliary numerical constants

Code accompanying

> A. Tankman, *Stability-Constrained Approximation in Spline KANs: Exact Layer
> Balancing and Budget-Compatible Saturation.*

**None of the theorems in the paper depends on the numbers produced here.** The
proofs use only the positivity of the constants, which is established
analytically. This script reproduces the illustrative table of Appendix B.

## Run

```
pip install -r requirements.txt
python3 reproduce-constants.py                     # the table
python3 reproduce-constants.py --refinement-check  # each quantity on N, 2N, 4N, 8N
```

## What is computed

| quantity | meaning | exact benchmark |
|---|---|---|
| `second_moment` | second moment of the Schoenberg weights, uniform grid | `(k+1)/12` for `k >= 2` (Lemma 7.1); fails for `k = 1`, as stated |
| `c_k` | `dist_Linf[-1/2,1/2](G_*, P_max(k-1,1))`, `G_* = x^2/2 - (x_+)^2` | `(3-2*sqrt2)/8` for `k <= 3` |
| `gamma_d` | `min ||q||_1 / ||q||_inf` over `P_d` | `sqrt2 - 1` for `d = 1` |
| `Lambda_k`, `kappa_k` | Lebesgue constant of the Chebyshev–Lobatto nodes; norm of the B-spline coefficient functional | `Lambda_1 = kappa_1 = 1` |
| `C_{0,k}`, `C_{1,k}` | quasi-interpolant constants of Lemma 2.5 | `9/4` and `15/4` for `k = 1` |

## Certification

Values are numerical approximations, not certified enclosures. The sampled
minimax value for `c_k` is a lower bound; the sampled feasible-set relaxation
for `kappa_k` is an upper bound; for `gamma_d` and for the sampled Lebesgue
maximum no one-sided certification is claimed. Use `--refinement-check` to see
the behaviour under grid refinement.

## Sample output

See `sample-output.txt`, produced by the two commands above.
