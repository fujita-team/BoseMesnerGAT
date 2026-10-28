"""
E1 - Expressivity restriction on the exact cycle C_10 (D = 5), with
     (i) an exact best-approximation floor, (ii) independent train/test
     signals, and (iii) separation of shared-Theta coupling from softmax
     positivity.   [Reviewer 3, comments 5 and 7]

All models act in the SAME node basis {T_k(A/2)}_{k=0..K}:
  ChebNet      : sum_k T_k X Theta_k                     (independent Theta_k)
  Shared-real  : sum_k c_k T_k X Theta,  c_k in R        (shared Theta only)
  Shared-soft  : sum_k a_k T_k X Theta,  a = softmax(w)  (= distance-invariant
                 global GAT on C_2D when K = D, Sec. 5.3)
Target: fixed operator sum_k T_k X Theta*_k, Theta*_k ~ N(0,1) entries, held
fixed across training and evaluation signals.

Part A (exact): normalised operator error ||O-O*||^2/||O*||^2 = expected
relative test MSE on i.i.d. Gaussian inputs, minimised in closed form
(Shared-real) or by multistart L-BFGS with Theta profiled out (Shared-soft),
over 50 target draws, for K in {3, 5} and d in {1, 2}.
Part B (training): Adam on N_train = 200 signals, evaluation on 2000
independent signals; compared with the Part-A floor.
"""
import json, sys
import numpy as np, networkx as nx
import autograd.numpy as anp
from common import (adjacency, cheb_basis, operator_matrix, best_shared_common_basis,
                    best_softmax_heads_common_basis, adam)

SEED = 2026
n, D = 10, 5
N_TARGETS = 50
N_TRAIN, N_TEST = 200, 2000
TRAIN_TARGETS = 5
STEPS, LR = 4000, 0.02


def part_a(rng):
    A = adjacency(nx.cycle_graph(n))
    out = {}
    for K in (3, 5):
        T = cheb_basis(A, K, scale=2.0)
        for d in (1, 2):
            real, soft = [], []
            for _ in range(N_TARGETS):
                Th = [rng.normal(size=(d, d)) for _ in range(K + 1)]
                real.append(best_shared_common_basis(T, Th, H=1))
                soft.append(best_softmax_heads_common_basis(T, Th, 1, rng, n_starts=20)[0])
            key = f"K{K}_d{d}"
            out[key] = dict(cheb=0.0,
                            shared_real_mean=float(np.mean(real)), shared_real_std=float(np.std(real)),
                            shared_soft_mean=float(np.mean(soft)), shared_soft_std=float(np.std(soft)),
                            frac_soft_gt_real=float(np.mean(np.array(soft) > np.array(real) + 1e-6)))
            print(key, out[key], flush=True)
    return out


def part_b(rng):
    A = adjacency(nx.cycle_graph(n))
    K, d = 3, 2
    T = cheb_basis(A, K, scale=2.0)
    Tst = np.stack(T)
    res = []
    for t in range(TRAIN_TARGETS):
        Th = [rng.normal(size=(d, d)) for _ in range(K + 1)]
        Xtr = rng.normal(size=(N_TRAIN, n, d)); Xte = rng.normal(size=(N_TEST, n, d))
        f = lambda X, B, M: anp.einsum('kij,sjp,kpq->siq', B, X, M)
        Ytr = f(Xtr, Tst, np.stack(Th)); Yte = f(Xte, Tst, np.stack(Th))
        ytr2, yte2 = np.mean(Ytr ** 2), np.mean(Yte ** 2)

        def models(kind):
            if kind == 'cheb':
                unpack = lambda x: anp.reshape(x, (K + 1, d, d))
                dim = (K + 1) * d * d
            else:
                def unpack(x):
                    c = x[:K + 1]
                    if kind == 'soft':
                        e = anp.exp(c - anp.max(c)); c = e / anp.sum(e)
                    Th_ = anp.reshape(x[K + 1:], (d, d))
                    return c[:, None, None] * Th_[None]
                dim = K + 1 + d * d
            return unpack, dim

        row = dict(target=t, floor_shared_real=best_shared_common_basis(T, Th, 1),
                   floor_shared_soft=best_softmax_heads_common_basis(T, Th, 1, rng, 20)[0])
        for kind in ('cheb', 'real', 'soft'):
            unpack, dim = models(kind)
            loss = lambda x: anp.mean((f(Xtr, Tst, unpack(x)) - Ytr) ** 2) / ytr2
            x0 = 0.1 * rng.normal(size=dim)
            if kind != 'cheb':
                x0[:K + 1] = 0.0          # uniform initial attention, as in Fig. 1
            x, hist = adam(loss, x0, steps=STEPS, lr=LR, log_every=50)
            te = float(np.mean((f(Xte, Tst, unpack(x)) - Yte) ** 2) / yte2)
            row[kind] = dict(train=float(loss(x)), test=te, curve=hist)
        print({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != 'curve'})
               for k, v in row.items()}, flush=True)
        res.append(row)
    return res


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    out = dict(config=dict(seed=SEED, n=n, D=D, n_targets=N_TARGETS, n_train=N_TRAIN, n_test=N_TEST,
                           steps=STEPS, lr=LR, train_targets=TRAIN_TARGETS),
               partA=part_a(rng), partB=part_b(rng))
    json.dump(out, open("results_E1.json", "w"), indent=1)
