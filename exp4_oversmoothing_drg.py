"""
E4 - Oversmoothing on a distance-regular graph (Petersen).  No training anywhere.
     [Reviewer 3 comment 10; Reviewer 2 comment 4]

Metric: R_L = ||H^(L) - E0 H^(L)||_F^2 / ||H^(0)||_F^2,  E0 = J/n.

E4a  Fixed operators of the original Fig. 2 (corrected normalisation):
       global  gamma = (0.2, 0.5, 0.3)/3.5   (gamma0 + 3 gamma1 + 6 gamma2 = 1)
       local   (gamma0, gamma1) = (0.3, 0.7/3)
     Theta = I, d = 4, 20 draws of H^(0) ~ N(0,1); bound rho^{2L}.
E4b  Layer-dependent attention + non-identity Theta, near-worst case:
       w^(l) = w_bar + N(0, 0.03^2), Theta^(l) = s * random orthogonal (d = 4).
E4c  Same, generic: w^(l) ~ N(0,1), Theta^(l) Gaussian rescaled to ||.||_2 = s.
     In E4b/E4c, s is chosen so that rho*s in {0.6, 0.9, 1.1}, rho = max_l rho_l.
E4d  Strict per-layer mixing vs uniform contraction: local attention, Theta = I,
     rho_l = 1 - delta_l with delta_l = 0.1, 0.5/(l+1), 0.5/(l+1)^2.
"""
import json
import numpy as np, networkx as nx
from common import adjacency, relation_matrices

SEED = 31
REPS, d = 20, 4
L_FIXED, L_LAYER, L_SCHED = 20, 30, 400
W_BAR = {'global': np.array([2.0, 0.5, 0.0]), 'local': np.array([1.0, 0.0, 0.0])}


def nonprincipal_radius(P, E0):
    Q = np.eye(P.shape[0]) - E0
    return float(np.linalg.norm(Q @ P @ Q, 2))


def spec_scaled(rng, d, s):
    M = rng.normal(size=(d, d)); return s * M / np.linalg.norm(M, 2)


def orth_scaled(rng, d, s):
    Qm, _ = np.linalg.qr(rng.normal(size=(d, d))); return s * Qm


def propagate(Ps, Ths, H0, E0):
    Q = np.eye(H0.shape[0]) - E0; h0 = np.sum(H0 ** 2); H = H0.copy()
    out = [np.sum((Q @ H) ** 2) / h0]
    for P, T in zip(Ps, Ths):
        H = P @ H @ T
        out.append(np.sum((Q @ H) ** 2) / h0)
    return out


def summary(curves):
    c = np.array(curves)
    return dict(median=np.median(c, 0).tolist(), q10=np.quantile(c, .1, 0).tolist(),
                q90=np.quantile(c, .9, 0).tolist(), max=c.max(0).tolist())


def e4a(rng):
    G = nx.petersen_graph(); A = adjacency(G); n = 10
    I, J = np.eye(n), np.ones((n, n)); E0 = J / n
    g = np.array([0.2, 0.5, 0.3]); g = g / (g @ np.array([1, 3, 6]))
    Pg = g[0] * I + g[1] * A + g[2] * (J - I - A)
    g0 = 0.3; g1 = (1 - g0) / 3; Pl = g0 * I + g1 * A
    out = {}
    for name, P in (('global', Pg), ('local', Pl)):
        rho = nonprincipal_radius(P, E0)
        curves = [propagate([P] * L_FIXED, [np.eye(d)] * L_FIXED, rng.normal(size=(n, d)), E0)
                  for _ in range(REPS)]
        bound = [rho ** (2 * l) for l in range(L_FIXED + 1)]
        out[name] = dict(coeffs=(g.tolist() if name == 'global' else [g0, g1]),
                         spectrum=sorted(set(np.round(np.linalg.eigvalsh(P), 6).tolist())),
                         row_sums=float(P.sum(1).mean()), rho=rho, bound=bound,
                         violations=int(np.sum(np.array(curves) > np.array(bound) + 1e-12)),
                         **summary(curves))
        print('E4a', name, 'rho', rho, 'violations', out[name]['violations'], flush=True)
    return out


def petersen_op(rng, kind, regime):
    Al = relation_matrices(nx.petersen_graph(), 2)
    w = rng.normal(size=3) if regime == 'generic' else W_BAR[kind] + 0.03 * rng.normal(size=3)
    Wm = sum(np.exp(w[i]) * Al[i] for i in range(3)) if kind == 'global' else \
        np.exp(w[0]) * Al[0] + np.exp(w[1]) * Al[1]
    return Wm / Wm.sum(1, keepdims=True)


def e4bc(rng):
    n = 10; E0 = np.ones((n, n)) / n; out = {}
    for regime, th in (('tight', orth_scaled), ('generic', spec_scaled)):
        for kind in ('global', 'local'):
            for rs in (0.6, 0.9, 1.1):
                curves, rhos = [], []
                for _ in range(REPS):
                    Ps = [petersen_op(rng, kind, regime) for _ in range(L_LAYER)]
                    assert np.allclose(Ps[0] @ Ps[1], Ps[1] @ Ps[0])      # stay in M
                    rho = max(nonprincipal_radius(P, E0) for P in Ps)
                    Ths = [th(rng, d, rs / rho) for _ in range(L_LAYER)]
                    curves.append(propagate(Ps, Ths, rng.normal(size=(n, d)), E0)); rhos.append(rho)
                bound = [rs ** (2 * l) for l in range(L_LAYER + 1)]
                key = f'{regime}_{kind}_{rs}'
                out[key] = dict(regime=regime, kind=kind, rho_s=rs, rho_median=float(np.median(rhos)),
                                rho_range=[float(min(rhos)), float(max(rhos))], bound=bound,
                                violations=int(np.sum(np.array(curves) > np.array(bound) + 1e-12)),
                                **summary(curves))
                print('E4bc', key, 'rho', round(np.median(rhos), 3), 'R_30', out[key]['median'][-1],
                      'viol', out[key]['violations'], flush=True)
    return out


def e4d(rng):
    A = adjacency(nx.petersen_graph()); n = 10; E0 = np.ones((n, n)) / n
    sched = {'uniform': lambda l: 0.1, 'harmonic': lambda l: 0.5 / (l + 1),
             'summable': lambda l: 0.5 / (l + 1) ** 2}
    H0 = rng.normal(size=(n, d)); out = {}
    for name, f in sched.items():
        Ps = []
        for l in range(L_SCHED):
            g0 = 1 - 1.5 * f(l); Ps.append(g0 * np.eye(n) + (1 - g0) / 3 * A)
        rl = [nonprincipal_radius(P, E0) for P in Ps]
        c = propagate(Ps, [np.eye(d)] * L_SCHED, H0, E0)
        out[name] = dict(curve=c, rho_l=rl, sup_rho=float(max(rl)), final=float(c[-1]),
                         prod_rho2=float(np.prod(np.array(rl) ** 2)))
        print('E4d', name, 'sup rho', max(rl), 'R_400', c[-1], flush=True)
    return out


if __name__ == '__main__':
    rng = np.random.default_rng(SEED)
    res = dict(E4a=e4a(rng), E4bc=e4bc(rng), E4d=e4d(rng),
               config=dict(seed=SEED, reps=REPS, d=d, L_fixed=L_FIXED, L_layer=L_LAYER, L_sched=L_SCHED))
    json.dump(res, open('results_E4.json', 'w'), indent=1)
