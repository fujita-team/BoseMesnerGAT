"""
E7 - Illustrations requested in the second round (Reviewer 3, comment 2).
No new experiments: uses the same operator classes and solvers as E1/E3.

(a) Approximation result for ONE fixed target on C_10 (full order K = D = 5,
    d_in = d_out = 2), in the orthonormalised Chebyshev basis
    T_k(A/2)/||T_k(A/2)||_F, so that the squared entries of the displayed
    matrices sum exactly to the normalised operator error of Eq. (22):
        target          W* = R C* / ||R C*||_F          (rows: relation k,
        residual        (R C - R C*) / ||R C*||_F         columns: vec(Theta) entry)
    Models: single head with real coefficients (exact minimum, Eq. (23));
    single softmax head; H = 2 and H = 3 summed softmax heads (best fits found by
    restarted L-BFGS, Theta eliminated exactly).  The exact rank-H minimum is
    reported next to each softmax fit as a lower bound.
(b) Supplementary: C_20 and one rewired Watts-Strogatz graph (k = 2, p = 0.2) under
    IDENTICAL distance logits w = (2, 1, 0.5, 0) and cutoff K = 3:
    distance-shell sizes n_r(u), row-softmax attention matrix, and the coupling
    energy between adjacency eigenspaces
        M_jk = ||E_j P E_k||_F^2 / ||P||_F^2      (sum_jk M_jk = 1),
    whose off-diagonal mass is the off-spectral energy of Sec. 7.7.
Writes results_E7.json.
"""
import json
import numpy as np, networkx as nx
import autograd.numpy as anp
from scipy.linalg import cholesky
from common import (adjacency, cheb_basis, gram, best_shared_common_basis,
                    best_softmax_heads_common_basis, softmax_cols, relation_matrices,
                    connected_ws)

SEED = 71
n, D, d = 10, 5, 2
STARTS = 30


def heads_fit(T, Th, H, rng):
    """Best softmax H-head fit: returns (eps, C) with C = Gamma T_h."""
    eps, x = best_softmax_heads_common_basis(T, Th, H, rng, n_starts=STARTS)
    G, _ = gram(T); R = cholesky(G, lower=False)
    Cstar = np.stack([M.reshape(-1, order='F') for M in Th])
    Gam = np.array(softmax_cols(x.reshape(len(T), H)))
    U = R @ Gam
    Tm = np.linalg.lstsq(U, R @ Cstar, rcond=None)[0]
    return float(eps), Gam @ Tm


def rank1_exact(T, Th):
    G, _ = gram(T); R = cholesky(G, lower=False)
    Cstar = np.stack([M.reshape(-1, order='F') for M in Th])
    U, s, Vt = np.linalg.svd(R @ Cstar, full_matrices=False)
    RC1 = s[0] * np.outer(U[:, 0], Vt[0])
    return np.linalg.solve(R, RC1)


def part_a(rng):
    A = adjacency(nx.cycle_graph(n))
    T = cheb_basis(A, D, scale=2.0)
    G, _ = gram(T); R = cholesky(G, lower=False)
    # draw candidate targets and keep the one whose rank-1 minimum is the median
    cands = [[rng.normal(size=(d, d)) for _ in range(D + 1)] for _ in range(21)]
    vals = [best_shared_common_basis(T, Th, 1) for Th in cands]
    Th = cands[int(np.argsort(vals)[len(vals) // 2])]
    Cstar = np.stack([M.reshape(-1, order='F') for M in Th])
    nrm = np.linalg.norm(R @ Cstar)
    W = lambda C: (R @ C) / nrm
    out = dict(gram_diag=np.diag(G).tolist(), gram_offdiag_max=float(np.max(np.abs(G - np.diag(np.diag(G))))),
               target=W(Cstar).tolist())
    C1 = rank1_exact(T, Th)
    out['H1_real'] = dict(eps=float(np.sum((W(C1) - W(Cstar)) ** 2)),
                          eps_exact=best_shared_common_basis(T, Th, 1), residual=(W(C1) - W(Cstar)).tolist())
    for H in (1, 2, 3):
        eps, C = heads_fit(T, Th, H, rng)
        out[f'H{H}_soft'] = dict(eps_fit=eps, eps_check=float(np.sum((W(C) - W(Cstar)) ** 2)),
                                 lower_bound_exact=best_shared_common_basis(T, Th, H),
                                 residual=(W(C) - W(Cstar)).tolist())
        print(H, out[f'H{H}_soft']['eps_fit'], out[f'H{H}_soft']['eps_check'], out[f'H{H}_soft']['lower_bound_exact'], flush=True)
    print('H1 real', out['H1_real']['eps'], out['H1_real']['eps_exact'])
    return out


def coupling(P, A):
    ev, U = np.linalg.eigh(A)
    groups = []
    for i, e in enumerate(ev):
        if groups and abs(e - groups[-1][0]) < 1e-6: groups[-1][1].append(i)
        else: groups.append([e, [i]])
    E = [U[:, g] @ U[:, g].T for _, g in groups]
    M = np.array([[np.sum((Ej @ P @ Ek) ** 2) for Ek in E] for Ej in E]) / np.sum(P ** 2)
    return M, [float(e) for e, _ in groups]


def part_b():
    w = np.array([2.0, 1.0, 0.5, 0.0]); K = 3
    G0 = nx.cycle_graph(20)
    G1, seed = connected_ws(20, 2, 0.2, seed=10_000 + 200)
    out = {}
    for name, G in (('C20', G0), ('rewired', G1)):
        A = adjacency(G); Al = relation_matrices(G, K)
        Wm = sum(np.exp(w[r]) * Al[r] for r in range(K + 1))
        P = Wm / Wm.sum(1, keepdims=True)
        M, ev = coupling(P, A)
        shells = np.stack([Al[r].sum(1) for r in range(K + 1)], 1)
        out[name] = dict(seed=seed if name == 'rewired' else None, edges=[list(map(int, e)) for e in G.edges()],
                         shells=shells.tolist(), P=P.tolist(), coupling=M.tolist(), eigenvalues=ev,
                         offspectral=float(1 - np.trace(M)), row_sum_range=[float(Wm.sum(1).min()), float(Wm.sum(1).max())],
                         symmetric=bool(np.allclose(P, P.T)))
        print(name, 'off-spectral', out[name]['offspectral'], 'normaliser range', out[name]['row_sum_range'])
    out['logits'] = w.tolist(); out['K'] = K
    return out


if __name__ == '__main__':
    rng = np.random.default_rng(SEED)
    json.dump(dict(partA=part_a(rng), partB=part_b(), config=dict(seed=SEED, starts=STARTS)),
              open('results_E7.json', 'w'), indent=1)
