"""
E2 - Perturbation experiment re-designed so that the compared models share
     the SAME node basis.  [Reviewer 3, comments 5, 7 and 8]

Graphs: C_20 rewired by Watts-Strogatz (k=2) with p in PS; for each p,
N_GRAPHS connected realisations (a disconnected draw is replaced by the next
seed; the seed actually used is stored).  Order K = 3, d_in = d_out = 2.
For every graph, N_TGT target operators are drawn with i.i.d. N(0,1)
component matrices Theta*_k, held fixed for evaluation.

Settings (all errors = normalised operator error = expected relative test
MSE on independent Gaussian signals; Theta always profiled out exactly):
  S1  Chebyshev basis  B_k = T_k(A / rho(A)) for target AND models
        indep : independent Theta_k                    -> 0 by construction
        real  : shared Theta, real c_k                 (closed form, Eckart-Young)
        pos   : shared Theta, positive c_k             (multistart L-BFGS)
  S2  Distance basis   B_k = A_k (pairs at distance <= K) for target AND models
        indep, real, pos as above, and
        attn  : actual distance-invariant row-softmax attention
                P(w)_uv = e^{w_d(u,v)} / sum_z e^{w_d(u,z)}  (not in span{A_k}
                when the shell sizes differ)  with one shared Theta
  S3  Original (mismatched) protocol of Fig. 3: Chebyshev target, shared-Theta
        models in the distance basis ('real' closed form, 'attn').
Graph statistics: delta_shell and delta_int (intersection-count variation;
zero iff distance-regular), both over all distances 1..diam.
"""
import json
import numpy as np, networkx as nx
from common import (adjacency, relation_matrices, cheb_basis, connected_ws, operator_matrix,
                    shell_nonuniformity, intersection_nonuniformity,
                    best_shared_common_basis, best_shared_any_basis,
                    best_positive_shared_any_basis, best_rowsoftmax_attention)

SEED = 7
N, K, d = 20, 3, 2
PS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
N_GRAPHS, N_TGT, STARTS = 5, 5, 8

if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    rows = []
    for p in PS:
        for g in range(N_GRAPHS):
            if p == 0.0:
                G, used = nx.cycle_graph(N), None
            else:
                G, used = connected_ws(N, 2, p, seed=10_000 * g + int(round(1000 * p)))
            A = adjacency(G)
            Tb = cheb_basis(A, K)
            Ab = relation_matrices(G, K)
            Ab = [M for M in Ab if M.sum() > 0]      # drop empty classes (diam < K)
            stats = dict(p=p, graph=g, seed_used=used, diameter=nx.diameter(G),
                         d_shell=shell_nonuniformity(G), d_int=intersection_nonuniformity(G))
            acc = {k: [] for k in ['S1_real', 'S1_pos', 'S2_real', 'S2_pos', 'S2_attn',
                                   'S3_real', 'S3_attn']}
            for _ in range(N_TGT):
                ThT = [rng.normal(size=(d, d)) for _ in Tb]
                ThA = [rng.normal(size=(d, d)) for _ in Ab]
                OT, OA = operator_matrix(Tb, ThT), operator_matrix(Ab, ThA)
                acc['S1_real'].append(best_shared_common_basis(Tb, ThT))
                acc['S1_pos'].append(best_positive_shared_any_basis(OT, Tb, d, d, N, rng, STARTS)[0])
                acc['S2_real'].append(best_shared_common_basis(Ab, ThA))
                acc['S2_pos'].append(best_positive_shared_any_basis(OA, Ab, d, d, N, rng, STARTS)[0])
                acc['S2_attn'].append(best_rowsoftmax_attention(OA, Ab, d, d, N, rng, STARTS)[0])
                acc['S3_real'].append(best_shared_any_basis(OT, Ab, d, d, N))
                acc['S3_attn'].append(best_rowsoftmax_attention(OT, Ab, d, d, N, rng, STARTS)[0])
            stats.update({k: float(np.mean(v)) for k, v in acc.items()})
            rows.append(stats)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()}, flush=True)
    json.dump(dict(config=dict(seed=SEED, N=N, K=K, d=d, ps=PS, n_graphs=N_GRAPHS,
                               n_targets=N_TGT, starts=STARTS), rows=rows),
              open("results_E2.json", "w"), indent=1)
