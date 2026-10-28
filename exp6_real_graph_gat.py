"""
E6 - Trained feature-dependent GATs on a real graph: testing the observable
     hypotheses of Sec. 7 (Reviewer 2 comments 1-2; Reviewer 3 comment 9).

Data: Zachary's karate club (n = 34, 78 edges, real 2-class labels 'club').
Features X = I_34 (one-hot node identity).  20 stratified random splits, 30%
of the nodes for training and 70% for testing.  Full-batch Adam (lr 0.01,
L2 5e-4 on transformations), 300 epochs, ELU between layers, no dropout,
no residual connections.

Models (2 layers, hidden width 16):
  GCN        P = D^{-1/2}(A+I)D^{-1/2}
  DI-local   P = row-softmax over the closed neighbourhood of (w0 self, w1 neighbour)
  DI-global  P = row-softmax over all distance classes, logits w_r
  DI-global-4h  4 summed distance-invariant global heads (Remark 5.4)
  GAT-1h     standard feature-dependent GAT (LeakyReLU scores), 1 head
  GAT-4h     standard GAT, 4 heads (concatenated in the hidden layer)

Diagnostics on the realised (trained) propagation operators P^(l):
  aniso(P)  = ||P - Pi(P)||_F / ||P||_F, Pi = average of P over each distance
              class within each row  (0 for any distance-invariant operator
              with equal row normalisers along a row)
  offspec(P)= sum_{j != k} ||E_j P E_k||_F^2 / ||P||_F^2, E_j = eigenprojectors of A
              (0 iff P commutes with A, i.e. P is a spectral filter of A)
  rho_l     = ||Q P Q||_2,  c_l = rho_l ||Theta^(l)||_2   (Q = I - E0; E0 = J/n for
              row-stochastic P, E0 = v0 v0^T, v0 ~ D^{1/2} 1, for GCN)
  erank(C)  = effective rank (exp of singular-value entropy) of the relation-to-
              feature map C of DI-global-4h (rows: relation classes)
Depth study: GCN, DI-local, GAT-1h with L in {2, 4, 8, 16} layers (10 splits):
  test accuracy and the non-principal energy fraction of the last hidden layer
  phi = ||Q H||_F^2 / ||H||_F^2, together with prod_l c_l (hidden layers), both
  at initialisation and after training.
"""
import json
import numpy as np, networkx as nx
import autograd.numpy as anp
from autograd import grad
from autograd.misc import flatten
from common import adjacency, distance_matrix

SEED = 51
N_SPLITS, N_SPLITS_DEPTH = 20, 10
EPOCHS, LR, WD, HID = 300, 0.01, 5e-4, 16
DEPTHS = [2, 4, 8, 16]


# ------------------------------------------------------------------ data
def load():
    G = nx.karate_club_graph()
    A = (adjacency(G) > 0).astype(float)
    y = np.array([0 if G.nodes[v]['club'] == 'Mr. Hi' else 1 for v in sorted(G.nodes())])
    return G, A, np.eye(A.shape[0]), y


def splits(y, rng, k, frac=0.3):
    out = []
    for _ in range(k):
        tr = []
        for c in np.unique(y):
            idx = rng.permutation(np.where(y == c)[0]); tr += list(idx[:max(1, int(round(frac * len(idx))))])
        tr = np.array(sorted(tr)); te = np.setdiff1d(np.arange(len(y)), tr)
        out.append((tr, te))
    return out


# ------------------------------------------------------------------ layers
def elu(x):
    return anp.where(x > 0, x, anp.exp(anp.minimum(x, 0)) - 1)


def rowsoftmax_masked(E, mask):
    E = E - 1e9 * (1 - mask)
    E = E - anp.max(E, axis=1, keepdims=True)
    W = anp.exp(E) * mask
    return W / anp.sum(W, axis=1, keepdims=True)


class Ctx:
    def __init__(self, G, A):
        n = A.shape[0]
        self.A, self.n = A, n
        self.mask = A + np.eye(n)
        Dm = distance_matrix(G); self.Dm = Dm
        self.diam = int(Dm.max())
        self.Ast = np.stack([(Dm == r).astype(float) for r in range(self.diam + 1)])
        d = self.mask.sum(1); self.S = self.mask / np.sqrt(np.outer(d, d))
        v0 = np.sqrt(d) / np.linalg.norm(np.sqrt(d)); self.E0_gcn = np.outer(v0, v0)
        self.E0 = np.ones((n, n)) / n
        ev, U = np.linalg.eigh(A); groups = []
        for i, e in enumerate(ev):
            if groups and abs(e - groups[-1][0]) < 1e-6: groups[-1][1].append(i)
            else: groups.append([e, [i]])
        self.Eproj = [U[:, g] @ U[:, g].T for _, g in groups]


def init_model(kind, L, din, dout, rng, ctx, heads=1):
    dims = [din] + [HID] * (L - 1) + [dout]
    ps = []
    for l in range(L):
        a, b = dims[l], dims[l + 1]
        last = (l == L - 1)
        if kind in ('gat1', 'gat4'):
            h = 1 if (kind == 'gat1' or last) else 4
            fo = b if last else b // h
            ps.append(dict(T=[rng.normal(scale=np.sqrt(2 / (a + fo)), size=(a, fo)) for _ in range(h)],
                           a_s=[0.1 * rng.normal(size=fo) for _ in range(h)],
                           a_d=[0.1 * rng.normal(size=fo) for _ in range(h)]))
        elif kind == 'diglobal4':
            h = 1 if last else 4
            ps.append(dict(T=[rng.normal(scale=np.sqrt(2 / (a + b)), size=(a, b)) for _ in range(h)],
                           w=[0.1 * rng.normal(size=ctx.diam + 1) for _ in range(h)]))
        else:
            p = dict(T=[rng.normal(scale=np.sqrt(2 / (a + b)), size=(a, b))])
            if kind == 'dilocal': p['w'] = [0.1 * rng.normal(size=2)]
            if kind == 'diglobal': p['w'] = [0.1 * rng.normal(size=ctx.diam + 1)]
            ps.append(p)
    return ps


def layer_ops(kind, p, H, ctx):
    """Returns list of (P_h, Theta_h) for this layer (P may depend on H)."""
    ops = []
    for h, T in enumerate(p['T']):
        if kind == 'gcn':
            P = ctx.S
        elif kind == 'dilocal':
            w = p['w'][h]; E = w[0] * np.eye(ctx.n) + w[1] * ctx.A
            P = rowsoftmax_masked(E, ctx.mask)
        elif kind in ('diglobal', 'diglobal4'):
            E = anp.tensordot(p['w'][h], ctx.Ast, axes=1)
            P = rowsoftmax_masked(E, np.ones((ctx.n, ctx.n)))
        else:
            Hp = anp.dot(H, T)
            e = anp.dot(Hp, p['a_s'][h])[:, None] + anp.dot(Hp, p['a_d'][h])[None, :]
            e = anp.where(e > 0, e, 0.2 * e)
            P = rowsoftmax_masked(e, ctx.mask)
        ops.append((P, T))
    return ops


def forward(kind, ps, X, ctx, record=False):
    H = X; rec = []
    for l, p in enumerate(ps):
        ops = layer_ops(kind, p, H, ctx)
        outs = [anp.dot(P, anp.dot(H, T)) for P, T in ops]
        if record:
            rec.append(dict(ops=[(np.array(P), np.array(T)) for P, T in ops], H_in=np.array(H)))
        if kind == 'diglobal4':
            Z = sum(outs)
        else:
            Z = anp.concatenate(outs, axis=1) if len(outs) > 1 else outs[0]
        H = elu(Z) if l < len(ps) - 1 else Z
    return (H, rec) if record else H


def train(kind, L, X, y, tr, ctx, rng, return_init=False):
    ps = init_model(kind, L, X.shape[1], 2, rng, ctx)
    flat, unflat = flatten(ps)
    Y = np.eye(2)[y]

    def loss(f):
        q = unflat(f); out = forward(kind, q, X, ctx)
        out = out - anp.max(out, axis=1, keepdims=True)
        lp = out - anp.log(anp.sum(anp.exp(out), axis=1, keepdims=True))
        ce = -anp.mean(anp.sum(Y[tr] * lp[tr], axis=1))
        reg = sum(anp.sum(T ** 2) for p in q for T in p['T'])
        return ce + WD * reg

    g = grad(loss); m = np.zeros_like(flat); v = np.zeros_like(flat); x = flat.copy()
    for t in range(1, EPOCHS + 1):
        gt = g(x); m = .9 * m + .1 * gt; v = .999 * v + .001 * gt ** 2
        x = x - LR * (m / (1 - .9 ** t)) / (np.sqrt(v / (1 - .999 ** t)) + 1e-8)
    return (unflat(x), ps) if return_init else unflat(x)


# ------------------------------------------------------------------ diagnostics
def aniso(P, ctx):
    Pi = np.zeros_like(P)
    for r in range(ctx.diam + 1):
        M = (ctx.Dm == r)
        cnt = M.sum(1, keepdims=True); mean = (P * M).sum(1, keepdims=True) / np.maximum(cnt, 1)
        Pi += mean * M
    return float(np.linalg.norm(P - Pi) / np.linalg.norm(P))


def offspec(P, ctx):
    diag = sum(np.sum((E @ P @ E) ** 2) for E in ctx.Eproj)
    return float(1 - diag / np.sum(P ** 2))


def contraction(kind, P, T, ctx):
    E0 = ctx.E0_gcn if kind == 'gcn' else ctx.E0
    Q = np.eye(ctx.n) - E0
    rho = float(np.linalg.norm(Q @ P @ Q, 2))
    return rho, float(np.linalg.norm(T, 2))


def erank(M):
    s = np.linalg.svd(M, compute_uv=False); p = s / s.sum(); p = p[p > 1e-15]
    return float(np.exp(-np.sum(p * np.log(p))))


def accuracy(kind, ps, X, y, te, ctx):
    return float(np.mean(np.argmax(forward(kind, ps, X, ctx), 1)[te] == y[te]))


def phi_last_hidden(kind, ps, X, ctx):
    _, rec = forward(kind, ps, X, ctx, record=True)
    H = rec[-1]['H_in']                       # input of the output layer = last hidden layer
    E0 = ctx.E0_gcn if kind == 'gcn' else ctx.E0
    Q = np.eye(ctx.n) - E0
    return float(np.sum((Q @ H) ** 2) / np.sum(H ** 2)), rec


def layer_c(kind, rec, ctx):
    cs = []
    for r in rec:
        terms = []
        for P, T in r['ops']:
            rho, s = contraction(kind, P, T, ctx); terms.append((rho * s) ** 2)
        cs.append(float(np.sqrt(sum(terms))))
    return cs


# ------------------------------------------------------------------ main
if __name__ == '__main__':
    rng = np.random.default_rng(SEED)
    G, A, X, y = load(); ctx = Ctx(G, A)
    SPL = splits(y, rng, N_SPLITS)
    kinds = ['gcn', 'dilocal', 'diglobal', 'diglobal4', 'gat1', 'gat4']
    res = dict(config=dict(seed=SEED, n_splits=N_SPLITS, epochs=EPOCHS, lr=LR, wd=WD, hidden=HID,
                           n=int(A.shape[0]), edges=int(A.sum() / 2), diameter=ctx.diam,
                           degree_range=[int(A.sum(1).min()), int(A.sum(1).max())]),
               models={})
    for kind in kinds:
        acc, an, osp, rho, s, c, er = [], [], [], [], [], [], []
        for tr, te in SPL:
            ps = train(kind, 2, X, y, tr, ctx, rng)
            acc.append(accuracy(kind, ps, X, y, te, ctx))
            _, rec = forward(kind, ps, X, ctx, record=True)
            P1 = [P for P, _ in rec[0]['ops']]               # hidden-layer operators
            an.append(np.mean([aniso(P, ctx) for P in P1]))
            osp.append(np.mean([offspec(P, ctx) for P in P1]))
            rs = [contraction(kind, P, T, ctx) for P, T in rec[0]['ops']]
            rho.append(np.mean([a for a, _ in rs])); s.append(np.mean([b for _, b in rs]))
            c.append(layer_c(kind, rec, ctx))
            if kind == 'diglobal4':
                p = ps[0]
                Gam = np.stack([np.exp(w) / np.exp(w).sum() for w in p['w']], 1)     # (D+1) x H
                Tm = np.stack([T.reshape(-1) for T in p['T']])                     # H x (din dout)
                er.append(erank(Gam @ Tm))
        res['models'][kind] = dict(acc_mean=float(np.mean(acc)), acc_std=float(np.std(acc)),
                                   aniso_mean=float(np.mean(an)), aniso_std=float(np.std(an)),
                                   offspec_mean=float(np.mean(osp)), offspec_std=float(np.std(osp)),
                                   rho1_mean=float(np.mean(rho)), s1_mean=float(np.mean(s)),
                                   c_layers_mean=np.mean(np.array(c), 0).tolist(),
                                   erank_mean=(float(np.mean(er)) if er else None),
                                   erank_std=(float(np.std(er)) if er else None))
        print(kind, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res['models'][kind].items()}, flush=True)

    # reference values for untrained / structural operators
    ref = {}
    ref['GCN operator'] = dict(aniso=aniso(ctx.S, ctx), offspec=offspec(ctx.S, ctx))
    P_unif = rowsoftmax_masked(np.zeros((ctx.n, ctx.n)), ctx.mask)
    ref['uniform local attention (D~^-1 A~)'] = dict(aniso=aniso(P_unif, ctx), offspec=offspec(P_unif, ctx))
    Pet = Ctx(nx.petersen_graph(), (adjacency(nx.petersen_graph()) > 0).astype(float))
    P_pet = rowsoftmax_masked(np.zeros((10, 10)), Pet.mask)
    ref['uniform local attention on Petersen (DRG)'] = dict(aniso=aniso(P_pet, Pet), offspec=offspec(P_pet, Pet))
    res['reference'] = ref
    print('reference', ref, flush=True)

    # depth study
    depth = {}
    SPLd = SPL[:N_SPLITS_DEPTH]
    for kind in ('gcn', 'dilocal', 'gat1'):
        for L in DEPTHS:
            acc, phi, logc, phi0, logc0 = [], [], [], [], []
            for tr, te in SPLd:
                ps, ps0 = train(kind, L, X, y, tr, ctx, rng, return_init=True)
                acc.append(accuracy(kind, ps, X, y, te, ctx))
                f, rec = phi_last_hidden(kind, ps, X, ctx); phi.append(f)
                logc.append(float(np.sum(np.log10(layer_c(kind, rec, ctx)[:-1]))))
                f0, rec0 = phi_last_hidden(kind, ps0, X, ctx); phi0.append(f0)
                logc0.append(float(np.sum(np.log10(layer_c(kind, rec0, ctx)[:-1]))))
            depth[f'{kind}_L{L}'] = dict(kind=kind, L=L, acc_mean=float(np.mean(acc)), acc_std=float(np.std(acc)),
                                         phi_median=float(np.median(phi)), phi_q10=float(np.quantile(phi, .1)),
                                         phi_q90=float(np.quantile(phi, .9)),
                                         log10_prod_c_median=float(np.median(logc)),
                                         phi_init_median=float(np.median(phi0)),
                                         log10_prod_c_init_median=float(np.median(logc0)))
            print(kind, L, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in depth[f'{kind}_L{L}'].items()}, flush=True)
    res['depth'] = depth
    json.dump(res, open('results_E6.json', 'w'), indent=1)
