"""
Shared utilities for the revision experiments (E1-E4).

Conventions
-----------
* A graph signal is X in R^{n x d_in}; a linear graph layer is
      F(X) = sum_k B_k X M_k ,  B_k in R^{n x n},  M_k in R^{d_in x d_out}.
* Column-major vectorisation gives  vec(F(X)) = O vec(X)  with the
  "operator matrix"  O = sum_k M_k^T (kron) B_k  in R^{n d_out x n d_in}.
* If the entries of X are i.i.d. N(0,1), then
      E || F(X) - F*(X) ||_F^2 = || O - O* ||_F^2 ,
  so the Frobenius operator error equals the expected squared error on
  independent Gaussian test signals.  We always report the NORMALISED
  operator error  ||O - O*||_F^2 / ||O*||_F^2  (= expected relative test MSE).

Only numpy / scipy / networkx / autograd are needed (no GPU, no torch).
"""
import numpy as np
import networkx as nx
import autograd.numpy as anp
from autograd import grad
from scipy.optimize import minimize
from scipy.linalg import eigh, cholesky, svd


# --------------------------------------------------------------------- graphs
def adjacency(G):
    nodes = sorted(G.nodes())
    return nx.to_numpy_array(G, nodelist=nodes)


def distance_matrix(G):
    nodes = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    Dm = np.full((n, n), np.inf)
    for u, lengths in nx.all_pairs_shortest_path_length(G):
        for v, d in lengths.items():
            Dm[idx[u], idx[v]] = d
    return Dm


def relation_matrices(G, K):
    """Distance matrices A_0..A_K (pairs at distance > K are not represented)."""
    Dm = distance_matrix(G)
    return [(Dm == i).astype(float) for i in range(K + 1)]


def cheb_basis(A, K, scale=None):
    """T_0..T_K evaluated at A/scale (default scale = spectral radius of A)."""
    n = A.shape[0]
    if scale is None:
        scale = np.max(np.abs(np.linalg.eigvalsh(A)))
    An = A / scale
    T = [np.eye(n), An.copy()]
    for _ in range(2, K + 1):
        T.append(2 * An @ T[-1] - T[-2])
    return T[: K + 1]


def connected_ws(n, k, p, seed, max_tries=1000):
    """Watts-Strogatz graph; re-draws with seed+1, seed+2, ... until connected.
    Returns (G, seed_used)."""
    for t in range(max_tries):
        G = nx.watts_strogatz_graph(n, k, p, seed=seed + t)
        if nx.is_connected(G):
            return G, seed + t
    raise RuntimeError("no connected graph found")


# ---------------------------------------------------- distance-regularity stats
def shell_nonuniformity(G, K=None):
    """delta_shell: mean over i=1..K of Var_u |N_u^(i)| (the statistic of Fig. 3)."""
    Dm = distance_matrix(G)
    K = int(np.max(Dm)) if K is None else K
    return float(np.mean([np.var((Dm == i).sum(1)) for i in range(1, K + 1)]))


def intersection_nonuniformity(G, K=None):
    """delta_int: mean over i=1..K of the variance, across ordered vertex pairs
    (u,v) with d(u,v)=i, of the intersection counts
        c_i(u,v) = |{w ~ v : d(u,w)=i-1}|,
        a_i(u,v) = |{w ~ v : d(u,w)=i  }|,
        b_i(u,v) = |{w ~ v : d(u,w)=i+1}|.
    With K = diameter, delta_int(G) = 0  <=>  G is distance-regular."""
    A = adjacency(G)
    Dm = distance_matrix(G)
    K = int(np.max(Dm)) if K is None else K
    Ai = lambda j: (Dm == j).astype(float)
    vals = []
    for i in range(1, K + 1):
        mask = Dm == i
        if not mask.any():
            continue
        c = (Ai(i - 1) @ A)[mask]
        a = (Ai(i) @ A)[mask]
        b = (Ai(i + 1) @ A)[mask]
        vals.append(np.var(c) + np.var(a) + np.var(b))
    return float(np.mean(vals))


# ----------------------------------------------------------- operator algebra
def operator_matrix(B_list, M_list):
    """O = sum_k M_k^T kron B_k."""
    return sum(np.kron(M.T, B) for B, M in zip(B_list, M_list))


def rearrange(O, d_out, d_in, n):
    """ rearrangement: R(Y kron Z) = vec(Y) vec(Z)^T for
    Y in R^{d_out x d_in}, Z in R^{n x n} (column-major vec)."""
    R = np.empty((d_out * d_in, n * n))
    for j in range(d_in):
        for i in range(d_out):
            R[j * d_out + i] = O[i * n:(i + 1) * n, j * n:(j + 1) * n].reshape(-1, order="F")
    return R


def gram(B_list):
    V = np.stack([B.reshape(-1, order="F") for B in B_list], axis=1)
    return V.T @ V, V


# ------------------------------------------- closed-form best approximations
def best_shared_common_basis(B_list, Theta_star, H=1):
    """Common node basis {B_k}; target O* = sum_k Theta*_k^T kron B_k.
    Model with H shared-transformation 'heads' and UNCONSTRAINED scalar
    mixing:  sum_k B_k X (sum_h g_{kh} Theta_h)  (H=1: shared Theta, real c_k).
    ||O - O*||^2 = || R (C - C*) ||_F^2 with G = R^T R (Gram of basis) and C the
    (K+1) x (d_in d_out) matrix whose rows are vec(M_k).  The model class is
    {rank(C) <= H}, so by Eckart-Young the exact minimum is
        sum_{j > H} sigma_j^2 (R C*).
    Returns the normalised minimal error."""
    G, _ = gram(B_list)
    R = cholesky(G, lower=False)
    Cstar = np.stack([T.reshape(-1, order="F") for T in Theta_star])
    s = svd(R @ Cstar, compute_uv=False)
    return float(np.sum(s[H:] ** 2) / np.sum(s ** 2))


def best_shared_any_basis(O_star, P_basis, d_out, d_in, n):
    """Target O* arbitrary; model P = sum_k c_k P_k (c real) with ONE shared
    Theta.  For fixed P the optimal Theta is a Kronecker least-squares solution
    and the profiled error is ||O*||^2 - ||R(O*) vec P||^2 / ||vec P||^2.
    Maximising over c is a generalised symmetric eigenproblem."""
    Rm = rearrange(O_star, d_out, d_in, n)
    G, V = gram(P_basis)
    S = V.T @ Rm.T @ Rm @ V
    w = eigh(S, G, eigvals_only=True)
    return float(1.0 - w[-1] / np.sum(O_star ** 2))


# --------------------------------------- constrained (softmax) best approximations
def _multistart(fun, dim, n_starts, rng, scale=2.0):
    g = grad(fun)
    best = (np.inf, None)
    for _ in range(n_starts):
        x0 = rng.normal(scale=scale, size=dim)
        r = minimize(fun, x0, jac=g, method="L-BFGS-B", options=dict(maxiter=2000))
        if r.fun < best[0]:
            best = (float(r.fun), r.x)
    return best


def softmax_cols(W):
    W = W - anp.max(W, axis=0, keepdims=True)
    E = anp.exp(W)
    return E / anp.sum(E, axis=0, keepdims=True)


def best_softmax_heads_common_basis(B_list, Theta_star, H, rng, n_starts=30):
    """H distance-invariant softmax heads in the common basis {B_k}:
    effective C = Gamma T, with Gamma in R^{(K+1) x H} having positive columns
    (softmax).  For fixed Gamma, T is a least-squares solution, so the profiled
    error is || (I - Proj_{R Gamma}) R C* ||^2.  Gamma optimised by multistart
    L-BFGS.  (Column normalisation is absorbed by T, so only positivity matters.)"""
    G, _ = gram(B_list)
    R = cholesky(G, lower=False)
    RC = R @ np.stack([T.reshape(-1, order="F") for T in Theta_star])
    tot = np.sum(RC ** 2)
    m = len(B_list)

    def fun(w):
        Gam = softmax_cols(anp.reshape(w, (m, H)))
        U = anp.dot(R, Gam)
        M = anp.dot(U.T, U) + 1e-10 * anp.eye(H)
        proj = anp.dot(U, anp.linalg.solve(M, anp.dot(U.T, RC)))
        return anp.sum((RC - proj) ** 2) / tot

    return _multistart(fun, m * H, n_starts, rng)


def best_rowsoftmax_attention(O_star, A_list, d_out, d_in, n, rng, n_starts=30):
    """Actual distance-invariant attention on an arbitrary graph:
        P(w)_{uv} = exp(w_{d(u,v)}) / sum_{z: d(u,z)<=K} exp(w_{d(u,z)}),
    i.e. row-softmax over relation classes 0..K (pairs farther than K masked),
    followed by ONE shared Theta (profiled out in closed form).  On a
    non-distance-regular graph the row normalisers differ, so P(w) is in
    general NOT in span{A_k}."""
    Rm = rearrange(O_star, d_out, d_in, n)
    tot = np.sum(O_star ** 2)
    Ast = np.stack(A_list)

    def P_of(w):
        E = anp.exp(w - anp.max(w))
        Wm = anp.tensordot(E, Ast, axes=1)
        return Wm / anp.sum(Wm, axis=1, keepdims=True)

    def fun(w):
        p = anp.reshape(anp.transpose(P_of(w)), (-1,))   # column-major vec
        q = anp.dot(Rm, p)
        return 1.0 - anp.sum(q ** 2) / anp.sum(p ** 2) / tot

    return _multistart(fun, len(A_list), n_starts, rng)


def best_positive_shared_any_basis(O_star, P_basis, d_out, d_in, n, rng, n_starts=30):
    """Shared Theta with POSITIVE scalar coefficients c = softmax(w) on an
    arbitrary (non-normalised) basis, Theta profiled out."""
    Rm = rearrange(O_star, d_out, d_in, n)
    G, V = gram(P_basis)
    S = V.T @ Rm.T @ Rm @ V
    tot = np.sum(O_star ** 2)

    def fun(w):
        e = anp.exp(w - anp.max(w))
        c = e / anp.sum(e)
        return 1.0 - anp.dot(c, anp.dot(S, c)) / anp.dot(c, anp.dot(G, c)) / tot

    return _multistart(fun, len(P_basis), n_starts, rng)


# ------------------------------------------------------------------ Adam
def adam(fun, x0, steps=3000, lr=0.02, b1=0.9, b2=0.999, eps=1e-8, log_every=0):
    g = grad(fun)
    x = x0.copy(); m = np.zeros_like(x); v = np.zeros_like(x)
    hist = []
    for t in range(1, steps + 1):
        gt = g(x)
        m = b1 * m + (1 - b1) * gt
        v = b2 * v + (1 - b2) * gt ** 2
        x = x - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        if log_every and t % log_every == 0:
            hist.append(float(fun(x)))
    return x, hist
