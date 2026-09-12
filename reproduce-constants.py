"""
Auxiliary numerical constants of Appendix B of

    A. Tankman, "Stability-Constrained Approximation in Spline KANs:
    Exact Layer Balancing and Budget-Compatible Saturation".

None of the theorems of the paper depends on the decimal values computed here;
the proofs use only the positivity of these constants, which is established
analytically.  This script reproduces the illustrative table.

Quantities:

  M_2      second moment of the Schoenberg weights on a uniform grid;
           Lemma 7.1 proves M_2 == (k+1)/12 for k >= 2, and the script
           checks that identity numerically (it fails for k = 1, as it must).
  c_k      dist_{Linf[-1/2,1/2]}(G_*, P_{max(k-1,1)}) with G_* = x^2/2 - (x_+)^2.
           Closed form (3 - 2*sqrt(2))/8 for k <= 3.
  gamma_d  min over q in P_d, q != 0, of ||q||_1 / ||q||_inf on an interval.
           Closed form sqrt(2) - 1 for d = 1.
  Lambda_k Lebesgue constant of the Chebyshev-Lobatto nodes of [0,1].
  kappa_k  norm of the B-spline coefficient functional c_i on P_k
           with respect to the sup norm on the support interval.
  C_{0,k}, C_{1,k}  the resulting quasi-interpolant constants.

Reported values are numerical approximations, not certified enclosures.  The
sampled minimax value for c_k is a lower bound and the sampled feasible-set
relaxation for kappa_k is an upper bound; for gamma_d and for the sampled
Lebesgue maximum no one-sided certification is claimed.

Usage:
    python3 reproduce-constants.py                    # table
    python3 reproduce-constants.py --refinement-check # N, 2N, 4N, 8N
"""

import sys
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import linprog, minimize
from math import factorial, cos, pi, sqrt

# --------------------------------------------------------------- second moment
def second_moment(k, npts=2001):
    """min and max of M_2 over one period of a uniform grid with h = 1."""
    m = 40
    t = np.arange(-m, m + 1, dtype=float)
    nb = len(t) - k - 1
    xi = np.array([t[i + 1:i + k + 1].mean() for i in range(nb)])
    ts = np.linspace(0.0, 1.0, npts)
    M2 = np.zeros_like(ts); S = np.zeros_like(ts); S1 = np.zeros_like(ts)
    for i in range(nb):
        c = np.zeros(nb); c[i] = 1.0
        N = BSpline(t, c, k)(ts)
        M2 += N * (xi[i] - ts) ** 2; S += N; S1 += N * xi[i]
    assert np.allclose(S, 1, atol=1e-9) and np.allclose(S1, ts, atol=1e-8)  # Marsden
    return M2.min(), M2.max()

# -------------------------------------------------- best uniform approximation
def best_unif(f, d, a=-0.5, b=0.5, ngrid=6001):
    """min over p in P_d of max over the grid of |f - p|;  a lower bound for c_k."""
    x = np.linspace(a, b, ngrid); y = f(x)
    V = np.vander(x, d + 1, increasing=True)
    A = np.vstack([np.hstack([V, -np.ones((ngrid, 1))]),
                   np.hstack([-V, -np.ones((ngrid, 1))])])
    r = linprog(np.r_[np.zeros(d + 1), 1.0], A_ub=A, b_ub=np.r_[y, -y],
                bounds=[(None, None)] * (d + 1) + [(0, None)], method="highs")
    return r.x[-1]

Gstar = lambda x: 0.5 * x ** 2 - np.maximum(x, 0.0) ** 2

# ------------------------------------------------------------------- gamma_d
def gamma_d(d, ngrid=4001, ntries=60, seed=0):
    """min over all q in P_d of ||q||_1 / ||q||_inf, by direct minimisation.

    The ratio is scale invariant, so no normalising constraint is imposed and
    the minimisation runs over the whole coefficient space; this matches the
    definition in the paper, with no restriction of the maximiser to selected
    grid points.
    """
    x = np.linspace(0, 1, ngrid)
    V = np.vander(x, d + 1, increasing=True)
    w = np.ones(ngrid) / (ngrid - 1); w[0] *= 0.5; w[-1] *= 0.5

    def ratio(c):
        q = V @ c
        mx = np.abs(q).max()
        return 1e6 if mx < 1e-12 else (w @ np.abs(q)) / mx

    # deterministic start from the shifted Chebyshev polynomial, which is the
    # natural candidate extremiser, plus random restarts
    ts = np.linspace(0, 1, 2000)
    cheb = np.zeros(d + 1); cheb[d] = 1.0
    c_cheb = np.polyfit(ts, np.polynomial.chebyshev.chebval(2 * ts - 1, cheb), d)[::-1]
    rng = np.random.default_rng(seed)
    starts = [c_cheb] + [rng.standard_normal(d + 1) for _ in range(ntries)]
    best = np.inf
    for c0 in starts:
        r = minimize(ratio, c0, method="Nelder-Mead",
                     options=dict(maxiter=8000, xatol=1e-11, fatol=1e-13))
        best = min(best, r.fun)
    return best

# --------------------------------------------------------- Lambda_k, kappa_k
def ref_nodes(k):
    """Chebyshev-Lobatto nodes of [0,1]."""
    return np.array([0.0, 1.0]) if k == 1 else \
           np.array([(1 - cos(r * pi / k)) / 2 for r in range(k + 1)])

def lebesgue(k, ngrid=20001):
    s = ref_nodes(k); x = np.linspace(0, 1, ngrid); L = np.zeros(ngrid)
    for r in range(k + 1):
        num = np.ones(ngrid); den = 1.0
        for m in range(k + 1):
            if m != r:
                num *= (x - s[m]); den *= (s[r] - s[m])
        L += np.abs(num / den)
    return L.max()

def kappa(k, ngrid=4001):
    """sup of |c_i(q)| over the grid-relaxed unit ball;  an upper bound."""
    m = 30; t = np.arange(-m, m + 1, dtype=float); nb = len(t) - k - 1; i = nb // 2
    xs = np.linspace(t[i] + 1e-9, t[i + 1] - 1e-9, 200); idx = list(range(i - k, i + 1))
    A = np.zeros((len(xs), len(idx)))
    for col, j in enumerate(idx):
        c = np.zeros(nb); c[j] = 1.0
        A[:, col] = BSpline(t, c, k)(xs)
    a = np.zeros(k + 1)
    for deg in range(k + 1):
        sol, *_ = np.linalg.lstsq(A, xs ** deg, rcond=None)
        a[deg] = sol[idx.index(i)]
    x = np.linspace(t[i], t[i + k + 1], ngrid); V = np.vander(x, k + 1, increasing=True)
    r = linprog(-a, A_ub=np.vstack([V, -V]), b_ub=np.ones(2 * ngrid),
                bounds=[(None, None)] * (k + 1), method="highs")
    return -r.fun

def qi_constants(k):
    Lam, kap = lebesgue(k), kappa(k)
    R = (2 * k + 1) / 2
    T0, T1 = R ** (k + 1) / factorial(k + 1), R ** k / factorial(k)
    return Lam, kap, (1 + Lam * kap) * T0, 2 * k * k * Lam * kap * T0 + T1

# ------------------------------------------------------------------- reports
def table():
    print("second moment on a uniform grid, h = 1   (Lemma 7.1: (k+1)/12 for k >= 2)")
    for k in range(1, 7):
        lo, hi = second_moment(k)
        flag = "" if k >= 2 else "   <-- not constant, as the lemma states"
        print(f"  k={k}: min={lo:.6f} max={hi:.6f}   (k+1)/12={(k+1)/12:.6f}{flag}")

    print("\nc_k = dist(G_*, P_{max(k-1,1)}) on [-1/2,1/2]")
    print(f"  closed form for k <= 3: (3-2*sqrt2)/8 = {(3-2*sqrt(2))/8:.9f}")
    for k in range(1, 7):
        print(f"  k={k}: {best_unif(Gstar, max(k - 1, 1)):.9f}")

    print("\ngamma_d = min ||q||_1 / ||q||_inf over P_d")
    print(f"  closed form for d = 1: sqrt2-1 = {sqrt(2)-1:.9f}")
    for d in range(1, 7):
        print(f"  d={d}: {gamma_d(d):.4f}")

    print("\n k   Lambda_k   kappa_k      C_{0,k}       C_{1,k}")
    for k in range(1, 6):
        Lam, kap, C0, C1 = qi_constants(k)
        print(f"{k:2d}  {Lam:9.4f} {kap:9.4f} {C0:12.4f} {C1:12.4f}")

def refinement_check():
    """Print each grid-dependent constant on N, 2N, 4N, 8N."""
    print("refinement check: each value on N, 2N, 4N, 8N\n")

    print("c_k  (k = 4, d = 3), grid on [-1/2,1/2]")
    N = 3001
    for j in range(4):
        print(f"  N={N*2**j:7d}: {best_unif(Gstar, 3, ngrid=N * 2 ** j):.9f}")

    print("\nkappa_k  (k = 4)")
    N = 2001
    for j in range(4):
        print(f"  N={N*2**j:7d}: {kappa(4, ngrid=N * 2 ** j):.9f}")

    print("\nLambda_k  (k = 4)")
    N = 5001
    for j in range(4):
        print(f"  N={N*2**j:7d}: {lebesgue(4, ngrid=N * 2 ** j):.9f}")

    print("\ngamma_d  (d = 3)")
    N = 1001
    for j in range(4):
        print(f"  N={N*2**j:7d}: {gamma_d(3, ngrid=N * 2 ** j, ntries=25):.9f}")

    print("\nsecond moment, min and max over a period  (k = 4)")
    N = 501
    for j in range(4):
        lo, hi = second_moment(4, npts=N * 2 ** j)
        print(f"  N={N*2**j:7d}: min={lo:.12f} max={hi:.12f}")

if __name__ == "__main__":
    if "--refinement-check" in sys.argv:
        refinement_check()
    else:
        table()
