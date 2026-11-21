"""
E6b (OPTIONAL, run on your own machine) - the E6 diagnostics on the Planetoid
citation graphs (Cora, CiteSeer) with the standard public split.

Requires: torch, torch_geometric (only for downloading the data).
    pip install torch torch_geometric
    python exp6b_planetoid_gat_torch.py --dataset Cora --runs 10
Writes results_E6b_<dataset>.json with the same fields as E6 (accuracy,
distance-class anisotropy, off-spectral energy, rho_l, c_l, depth study).

Implementation notes
* Models: GCN, DI-local (distance-invariant local attention), GAT-1h, GAT-8h,
  2 layers, hidden 64 (GAT-8h: 8 x 8), ELU, dropout 0.6 on inputs/attention for
  GATs (as in Velickovic et al.), Adam lr 0.005, weight decay 5e-4, 300 epochs,
  model selected at best validation accuracy.
* Graph = largest connected component is NOT taken: we use the full graph,
  self-loops added.  Dense n x n attention is used for clarity (n = 2708 / 3327
  fits easily in memory).  Off-spectral energy uses the eigendecomposition of A
  (dense eigh; ~ a few seconds for Cora).
* The distance-class anisotropy uses only classes 0 and 1 (support of local
  attention), which is exact for local operators.
"""
import argparse, json
import numpy as np
import torch, torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument('--dataset', default='Cora'); p.add_argument('--runs', type=int, default=10)
p.add_argument('--epochs', type=int, default=300); p.add_argument('--root', default='data')
p.add_argument('--depths', default='2,4,8,16')
args = p.parse_args()
torch.manual_seed(0); np.random.seed(0)


def load(name, root):
    from torch_geometric.datasets import Planetoid
    ds = Planetoid(root, name); d = ds[0]
    n = d.num_nodes
    A = torch.zeros(n, n); A[d.edge_index[0], d.edge_index[1]] = 1.0
    A = ((A + A.T) > 0).float(); A.fill_diagonal_(0)
    X = d.x / d.x.sum(1, keepdim=True).clamp(min=1)          # row-normalised features
    return A, X, d.y, d.train_mask, d.val_mask, d.test_mask, ds.num_classes


A, X, y, trm, vam, tem, C = load(args.dataset, args.root)
n = A.shape[0]; I = torch.eye(n); mask = A + I
deg = mask.sum(1); S = mask / torch.sqrt(deg[:, None] * deg[None, :])
v0 = torch.sqrt(deg) / torch.linalg.norm(torch.sqrt(deg)); E0_gcn = torch.outer(v0, v0)
E0 = torch.full((n, n), 1.0 / n)


def masked_softmax(E):
    E = E.masked_fill(mask == 0, float('-inf'))
    return torch.softmax(E, dim=1)


class Net(torch.nn.Module):
    def __init__(self, kind, L, din, hid, dout, heads=1):
        super().__init__()
        self.kind, self.L = kind, L
        dims = [din] + [hid] * (L - 1) + [dout]
        self.T = torch.nn.ModuleList(); self.a_s = torch.nn.ParameterList(); self.a_d = torch.nn.ParameterList()
        self.w = torch.nn.ParameterList(); self.heads = []
        for l in range(L):
            h = 1 if (kind != 'gat8' or l == L - 1) else heads
            fo = dims[l + 1] if h == 1 else dims[l + 1] // h
            self.heads.append(h)
            self.T.append(torch.nn.ModuleList([torch.nn.Linear(dims[l], fo, bias=False) for _ in range(h)]))
            self.a_s.append(torch.nn.Parameter(0.1 * torch.randn(h, fo)))
            self.a_d.append(torch.nn.Parameter(0.1 * torch.randn(h, fo)))
            self.w.append(torch.nn.Parameter(0.1 * torch.randn(2)))

    def ops(self, l, H):
        out = []
        for h, lin in enumerate(self.T[l]):
            Hp = lin(H)
            if self.kind == 'gcn':
                P = S
            elif self.kind == 'dilocal':
                P = masked_softmax(self.w[l][0] * I + self.w[l][1] * A)
            else:
                e = (Hp @ self.a_s[l][h])[:, None] + (Hp @ self.a_d[l][h])[None, :]
                P = masked_softmax(F.leaky_relu(e, 0.2))
                if self.training: P = F.dropout(P, 0.6)
            out.append((P, lin.weight.T, Hp))
        return out

    def forward(self, X, record=False):
        H = X; rec = []
        for l in range(self.L):
            if self.kind.startswith('gat') and self.training: H = F.dropout(H, 0.6)
            o = self.ops(l, H)
            if record: rec.append(dict(ops=[(P.detach(), T.detach()) for P, T, _ in o], H_in=H.detach()))
            Z = torch.cat([P @ Hp for P, _, Hp in o], 1)
            H = F.elu(Z) if l < self.L - 1 else Z
        return (H, rec) if record else H


def run(kind, L, seed):
    torch.manual_seed(seed)
    m = Net(kind, L, X.shape[1], 64, C, heads=8)
    opt = torch.optim.Adam(m.parameters(), lr=0.005, weight_decay=5e-4)
    best, state = -1, None
    for _ in range(args.epochs):
        m.train(); opt.zero_grad(); F.cross_entropy(m(X)[trm], y[trm]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            va = (m(X).argmax(1)[vam] == y[vam]).float().mean().item()
        if va > best: best, state = va, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state); m.eval()
    return m


# ---- diagnostics (same definitions as exp6_real_graph_gat.py) ----
print('eigendecomposition of A ...', flush=True)
ev, U = torch.linalg.eigh(A.double())
groups, cur = [], [0]
for i in range(1, n):
    if abs(ev[i] - ev[cur[-1]]) < 1e-6: cur.append(i)
    else: groups.append(cur); cur = [i]
groups.append(cur)
Ud = U.float()


def offspec(P):
    Pt = Ud.T @ P @ Ud; diag = 0.0
    for g in groups:
        diag += (Pt[g[0]:g[-1] + 1, g[0]:g[-1] + 1] ** 2).sum().item()
    return 1 - diag / (P ** 2).sum().item()


def aniso(P):
    off = P * A; cnt = A.sum(1, keepdim=True).clamp(min=1)
    mean = off.sum(1, keepdim=True) / cnt
    Pi = torch.diag(torch.diag(P)) + mean * A
    return (torch.linalg.norm(P - Pi) / torch.linalg.norm(P)).item()


def cvals(kind, rec):
    Q = I - (E0_gcn if kind == 'gcn' else E0); cs = []
    for r in rec:
        cs.append(float(np.sqrt(sum((torch.linalg.matrix_norm(Q @ P @ Q, 2).item() *
                                     torch.linalg.matrix_norm(T, 2).item()) ** 2 for P, T in r['ops']))))
    return cs


res = dict(dataset=args.dataset, n=n, edges=int(A.sum().item() / 2), models={}, depth={})
for kind in ('gcn', 'dilocal', 'gat1', 'gat8'):
    acc, an, osp, c = [], [], [], []
    for r in range(args.runs):
        m = run(kind, 2, r)
        with torch.no_grad():
            out, rec = m(X, record=True)
        acc.append((out.argmax(1)[tem] == y[tem]).float().mean().item())
        P1 = [P for P, _ in rec[0]['ops']]
        an.append(np.mean([aniso(P) for P in P1])); osp.append(np.mean([offspec(P) for P in P1]))
        c.append(cvals(kind, rec))
    res['models'][kind] = dict(acc_mean=float(np.mean(acc)), acc_std=float(np.std(acc)),
                               aniso_mean=float(np.mean(an)), offspec_mean=float(np.mean(osp)),
                               c_layers_mean=np.mean(np.array(c), 0).tolist())
    print(kind, res['models'][kind], flush=True)

for kind in ('gcn', 'dilocal', 'gat1'):
    for L in [int(v) for v in args.depths.split(',')]:
        acc, phi = [], []
        for r in range(min(args.runs, 5)):
            m = run(kind, L, 100 + r)
            with torch.no_grad():
                out, rec = m(X, record=True)
            H = rec[-1]['H_in']; Q = I - (E0_gcn if kind == 'gcn' else E0)
            phi.append(((Q @ H) ** 2).sum().item() / (H ** 2).sum().item())
            acc.append((out.argmax(1)[tem] == y[tem]).float().mean().item())
        res['depth'][f'{kind}_L{L}'] = dict(acc_mean=float(np.mean(acc)), phi_median=float(np.median(phi)))
        print(kind, L, res['depth'][f'{kind}_L{L}'], flush=True)

json.dump(res, open(f'results_E6b_{args.dataset}.json', 'w'), indent=1)
