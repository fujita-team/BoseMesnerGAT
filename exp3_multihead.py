"""
E3 - Multi-head ablation on the exact cycle C_10 (D = 5, full order).
     [Reviewer 2 comment 3; Reviewer 3 comment 9]

Summed H-head distance-invariant global attention on C_2D:
    H^{l+1} = sum_h sum_k a_{k,h} T_k(A/2) X Theta_h ,  a_{.,h} = softmax(w_h)
so the effective relation-to-feature map is C = Gamma T (Gamma >= 0, (D+1) x H).
Target: random ChebNet layer of order D (independent Theta*_k).
Reported (normalised operator error, N_TGT targets, d in {2, 3}):
  heads-real : unconstrained Gamma  -> exact floor sum_{j>H} sigma_j^2 (Eckart-Young)
  heads-soft : softmax Gamma         -> multistart L-BFGS, T profiled out
  constructive check at H = D+1: Gamma = column-normalised (I + eps J), T = Gamma^{-1} C*
"""
import json
import numpy as np, networkx as nx
from scipy.linalg import cholesky
from common import (adjacency, cheb_basis, operator_matrix, best_shared_common_basis,
                    best_softmax_heads_common_basis)

SEED = 11
n, D = 10, 5
N_TGT, STARTS = 30, 25

if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    A = adjacency(nx.cycle_graph(n))
    T = cheb_basis(A, D, scale=2.0)
    out = {}
    for d in (2, 3):
        res = {H: dict(real=[], soft=[]) for H in range(1, D + 2)}
        construct = []
        for t in range(N_TGT):
            Th = [rng.normal(size=(d, d)) for _ in range(D + 1)]
            for H in range(1, D + 2):
                res[H]['real'].append(best_shared_common_basis(T, Th, H=H))
                res[H]['soft'].append(best_softmax_heads_common_basis(T, Th, H, rng, STARTS)[0])
            # constructive representation with D+1 positive heads
            eps = 0.1
            Gam = np.eye(D + 1) + eps * np.ones((D + 1, D + 1)); Gam /= Gam.sum(0, keepdims=True)
            Cstar = np.stack([M.reshape(-1, order='F') for M in Th])
            Tm = np.linalg.solve(Gam, Cstar)                      # head transformations
            Thetas_h = [Tm[h].reshape(d, d, order='F') for h in range(D + 1)]
            O = sum(operator_matrix(T, [Gam[k, h] * Thetas_h[h] for k in range(D + 1)]) for h in range(D + 1))
            Os = operator_matrix(T, Th)
            construct.append(float(np.sum((O - Os) ** 2) / np.sum(Os ** 2)))
        out[f"d{d}"] = {str(H): dict(real_mean=float(np.mean(v['real'])), real_std=float(np.std(v['real'])),
                                     soft_mean=float(np.mean(v['soft'])), soft_std=float(np.std(v['soft'])),
                                     soft_zero_frac=float(np.mean(np.array(v['soft']) < 1e-6)))
                        for H, v in res.items()}
        out[f"d{d}"]["constructive_max_err"] = float(np.max(construct))
        print(d, json.dumps(out[f"d{d}"], indent=0), flush=True)
    out["config"] = dict(seed=SEED, n=n, D=D, n_targets=N_TGT, starts=STARTS)
    json.dump(out, open("results_E3.json", "w"), indent=1)
