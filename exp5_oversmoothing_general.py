"""
E5 - Oversmoothing beyond distance-regular graphs (Theorem 6.6 and the weaker
     norm-based condition of Sec. 6.3).  No training anywhere.  [Reviewer 3 comment 10]

Graphs: C_20, Petersen (DRGs); Zachary karate club, Watts-Strogatz (24, 4, 0.3, seed 7).
Metric: R_L = ||H^(L) - E0 H^(L)||_F^2 / ||H^(0)||_F^2.

E5a  Fixed symmetric operator (corrected version of the original Fig. 4):
       K = (Z + Z^T)/2, Z_uv ~ N(0,1);  W = exp(K) * (A + I);  S = diag(W 1);
       P = S^{-1/2} W S^{-1/2};  v0 = S^{1/2} 1 / ||S^{1/2} 1||,  E0 = v0 v0^T
     (P v0 = v0 exactly).  Theta = I, d = 6, L = 40, 20 draws of (Z, H^(0)).
     Bound rho^{2L}, rho = max_{i>=1} |mu_i(P)|.
     The single draw of the original Fig. 4 (np.random.seed(0)) is also reported.
E5b  Non-commuting layer-dependent operators with a common invariant subspace:
       W_uv = exp(z_uv) on edges, z ~ N(0,1), redrawn every layer;
       P^(l) = I - L_W / (max deg_W + 1)  (symmetric, >= 0, doubly stochastic,
       E0 = J/n), Theta^(l) Gaussian rescaled so that
       c_l = ||Q P^(l) Q||_2 ||Theta^(l)||_2 <= 0.95.  Bound prod_l c_l^2 ||Q H0||^2.
"""
import json
import numpy as np, networkx as nx
from common import adjacency

SEED = 41
REPS = 20


def graphs():
    return {'C20': nx.cycle_graph(20), 'Petersen': nx.petersen_graph(),
            'Karate club': nx.karate_club_graph(),
            'Small-world': nx.watts_strogatz_graph(24, k=4, p=0.3, seed=7)}


def sym_operator(A, Z):
    n = A.shape[0]; mask = (A > 0).astype(float) + np.eye(n)
    K = (Z + Z.T) / 2; W = np.exp(K) * mask
    s = W.sum(1); P = W / np.sqrt(np.outer(s, s))
    v0 = np.sqrt(s) / np.linalg.norm(np.sqrt(s))
    return P, np.outer(v0, v0), K[mask > 0]


def summary(c):
    c = np.array(c)
    return dict(median=np.median(c, 0).tolist(), q10=np.quantile(c, .1, 0).tolist(),
                q90=np.quantile(c, .9, 0).tolist())


def e5a(rng):
    L, d = 40, 6; out = {}
    # original single draw (seed 0, original draw order) for reporting
    np.random.seed(0); orig = {}
    for name, G in graphs().items():
        A = adjacency(G); n = A.shape[0]
        P, E0, _ = sym_operator(A, np.random.randn(n, n)); _ = np.random.randn(n, 6)
        ev = np.sort(np.abs(np.linalg.eigvalsh(P)))
        orig[name] = float(ev[-2])
    for name, G in graphs().items():
        A = adjacency(G); n = A.shape[0]
        curves, rhos, lmin, scores = [], [], [], []
        for _ in range(REPS):
            P, E0, sc = sym_operator(A, rng.normal(size=(n, n)))
            assert np.allclose(P @ E0, E0)
            ev = np.linalg.eigvalsh(P)
            rho = float(np.sort(np.abs(ev))[-2]); rhos.append(rho); lmin.append(float(ev.min()))
            scores.append(sc)
            H0 = rng.normal(size=(n, d)); Q = np.eye(n) - E0; h0 = np.sum(H0 ** 2); H = H0.copy()
            c = [np.sum((Q @ H) ** 2) / h0]
            for _ in range(L):
                H = P @ H; c.append(np.sum((Q @ H) ** 2) / h0)
            curves.append(c)
        rho_max = max(rhos)
        bounds = np.array([[r ** (2 * l) for l in range(L + 1)] for r in rhos])
        sc = np.concatenate(scores)
        out[name] = dict(n=n, edges=G.number_of_edges(), rho_median=float(np.median(rhos)),
                         rho_range=[float(min(rhos)), float(max(rhos))], lambda_min_median=float(np.median(lmin)),
                         score_mean=float(sc.mean()), score_std=float(sc.std()), rho_original_fig4=orig[name],
                         bound_median=np.median(bounds, 0).tolist(),
                         violations=int(np.sum(np.array(curves) > bounds + 1e-12)), **summary(curves))
        print('E5a', name, 'rho median', np.median(rhos), 'orig', orig[name], 'viol', out[name]['violations'], flush=True)
    return out


def e5b(rng):
    L, d = 30, 4; out = {}
    for name in ('Karate club', 'Small-world'):
        A = (adjacency(graphs()[name]) > 0).astype(float); n = A.shape[0]
        E0 = np.ones((n, n)) / n; Q = np.eye(n) - E0
        curves, bounds, comms = [], [], []
        for _ in range(REPS):
            Ps = []
            for _ in range(L):
                Z = np.triu(rng.normal(size=(n, n)), 1); Z = Z + Z.T
                W = np.exp(Z) * A
                Ps.append(np.eye(n) - (np.diag(W.sum(1)) - W) / (W.sum(1).max() + 1.0))
            assert np.allclose(Ps[0].sum(1), 1) and np.allclose(Ps[0], Ps[0].T) and Ps[0].min() >= 0
            q = np.array([np.linalg.norm(Q @ P @ Q, 2) for P in Ps]); s = 0.95 / q.max()
            Ths = []
            for _ in range(L):
                M = rng.normal(size=(d, d)); Ths.append(s * M / np.linalg.norm(M, 2))
            H0 = rng.normal(size=(n, d)); h0 = np.sum(H0 ** 2); H = H0.copy()
            c = [np.sum((Q @ H) ** 2) / h0]; b = [np.sum((Q @ H0) ** 2) / h0]
            for l in range(L):
                H = Ps[l] @ H @ Ths[l]; c.append(np.sum((Q @ H) ** 2) / h0); b.append(b[-1] * (q[l] * s) ** 2)
            curves.append(c); bounds.append(b)
            comms.append(float(np.linalg.norm(Ps[0] @ Ps[1] - Ps[1] @ Ps[0]) /
                               (np.linalg.norm(Ps[0]) * np.linalg.norm(Ps[1]))))
        out[name] = dict(n=n, bound_median=np.median(bounds, 0).tolist(),
                         violations=int(np.sum(np.array(curves) > np.array(bounds) + 1e-12)),
                         rel_commutator_median=float(np.median(comms)), **summary(curves))
        print('E5b', name, 'viol', out[name]['violations'], 'comm', np.median(comms), flush=True)
    return out


if __name__ == '__main__':
    rng = np.random.default_rng(SEED)
    json.dump(dict(E5a=e5a(rng), E5b=e5b(rng), config=dict(seed=SEED, reps=REPS)),
              open('results_E5.json', 'w'), indent=1)
