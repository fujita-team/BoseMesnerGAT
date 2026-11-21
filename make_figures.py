"""
Builds the six figures of Section 7 from results_E1..E6.json:
  figs/fig_E1_cycle.pdf, figs/fig_E2_perturbation.pdf, figs/fig_E3_multihead.pdf,
  figs/fig_E4_oversmoothing_drg.pdf, figs/fig_E5_oversmoothing_general.pdf,
  figs/fig_E6_real_graph.pdf
and writes every number quoted in the text to tables_for_text.txt.
Font sizes: titles 14, axis labels 12, legends 10 (fig_style.py).
"""
import json, os
import numpy as np
from fig_style import apply_style, legend, title, labels, COLORS as C
import matplotlib.pyplot as plt

apply_style()
os.makedirs('figs', exist_ok=True)
txt = []
J = lambda f: json.load(open(f))

# ======================================================================= E1
E1 = J('results_E1.json'); A = E1['partA']; B = E1['partB']
fig, axs = plt.subplots(1, 3, figsize=(18, 5.0))
keys = ['K3_d1', 'K3_d2', 'K5_d1', 'K5_d2']; x = np.arange(4); w = 0.26
axs[0].bar(x - w, [0] * 4, w, color=C['cheb'], label='ChebNet')
axs[0].bar(x, [A[k]['shared_real_mean'] for k in keys], w, yerr=[A[k]['shared_real_std'] for k in keys],
           capsize=3, color=C['real'], label=r'shared $\Theta$, real $c_k$ (exact)')
axs[0].bar(x + w, [A[k]['shared_soft_mean'] for k in keys], w, yerr=[A[k]['shared_soft_std'] for k in keys],
           capsize=3, color=C['soft'], label=r'shared $\Theta$, softmax (L-BFGS fit)')
for xi in x:
    axs[0].text(xi - w, 0.008, '0', ha='center', va='bottom', color=C['cheb'], fontsize=10, fontweight='bold')
axs[0].set_xticks(x, ['K=3 trunc.\nd=1', 'K=3 trunc.\nd=2', 'K=D=5 full\nd=1', 'K=D=5 full\nd=2'])
labels(axs[0], 'order $K$, feature dimension $d$', 'normalized operator error')
title(axs[0], '(a) Operator error, 50 targets'); legend(axs[0], loc='upper left')
ax = axs[1]
for K, st in (('K3', '-'), ('K5', '--')):
    R = B[K]; steps = (np.arange(len(R[0]['cheb']['curve'])) + 1) * 50
    for kind, lab in (('cheb', 'ChebNet'), ('real', r'real $c_k$'), ('soft', 'softmax')):
        cur = np.array([r[kind]['curve'] for r in R])
        ax.semilogy(steps, np.maximum(np.median(cur, 0), 1e-33), st, color=C[kind], lw=2,
                    label=f'{lab}, {"K=3" if K == "K3" else "K=D=5"}')
ax.set_ylim(1e-12, 5)
ax.text(0.02, 0.03, 'ChebNet reaches $<10^{-12}$ (numerical zero)', transform=ax.transAxes, fontsize=10, fontweight='bold', color=C['cheb'])
labels(ax, 'Adam step', 'relative training error'); title(ax, '(b) Training (median, 5 targets)')
legend(ax, loc='center right', ncol=2)
ax = axs[2]
xs = []
for gi, K in enumerate(('K3', 'K5')):
    for t, r in enumerate(B[K]):
        xc = gi * 6 + t
        ax.hlines(r['lb_exact'], xc - 0.38, xc + 0.38, color='k', lw=2.2, label='exact rank-1 min. (lower bound)' if xc == 0 else None)
        ax.scatter(xc - 0.18, r['real']['test'], marker='o', s=42, color=C['real'], label=r'Adam test, real $c_k$' if xc == 0 else None, zorder=5)
        ax.scatter(xc + 0.18, r['soft']['test'], marker='s', s=42, color=C['soft'], label='Adam test, softmax' if xc == 0 else None, zorder=5)
        ax.scatter(xc + 0.18, r['fit_soft'], marker='_', s=260, linewidths=2.2, color=C['soft'], label='softmax L-BFGS fit' if xc == 0 else None, zorder=6)
ax.set_xticks([2, 8], ['K=3 (truncated)', 'K=D=5 (full)']); ax.axvline(5, color='grey', lw=0.8)
labels(ax, 'target (5 per order)', 'normalized error'); title(ax, '(c) Per-target test error')
legend(ax, loc='upper left', ncol=1)
lo = min(min(r['lb_exact'] for r in B[K]) for K in B); hi = max(max(r['soft']['test'] for r in B[K]) for K in B)
ax.set_ylim(lo - 0.08, hi + 0.25)
plt.tight_layout(); plt.savefig('figs/fig_E1_cycle.pdf'); plt.close()
txt.append('E1A ' + json.dumps(A))
for K in ('K3', 'K5'):
    for r in B[K]:
        txt.append(f"E1B {K} t{r['target']} LB {r['lb_exact']:.4f} softfit {r['fit_soft']:.4f} | " +
                   ' '.join(f"{k}: tr {r[k]['train']:.3g} te {r[k]['test']:.4f} ps {r[k]['test_persignal']:.4f} epsL {r[k]['eps_learned']:.4f}" for k in ('cheb', 'real', 'soft')))
    R = B[K]
    txt.append(f"E1B {K} max|test-epsL| real {max(abs(r['real']['test']-r['real']['eps_learned']) for r in R):.4f} soft {max(abs(r['soft']['test']-r['soft']['eps_learned']) for r in R):.4f}; "
               f"max(test_real-LB) {max(r['real']['test']-r['lb_exact'] for r in R):.4f} min {min(r['real']['test']-r['lb_exact'] for r in R):.4f}; "
               f"max|test_soft-fit| {max(abs(r['soft']['test']-r['fit_soft']) for r in R):.4f}; max|train-test| {max(max(abs(r[k]['train']-r[k]['test']) for k in ('real','soft')) for r in R):.4f}; "
               f"max|ps-tot| {max(max(abs(r[k]['test_persignal']-r[k]['test']) for k in ('real','soft')) for r in R):.4f}; cheb max {max(max(r['cheb']['train'], r['cheb']['test']) for r in R):.2e}")

# ======================================================================= E2
E2 = J('results_E2.json'); rows = E2['rows']; ps = E2['config']['ps']; xp = np.array(ps)
agg = lambda k: (np.array([np.mean([r[k] for r in rows if r['p'] == p]) for p in ps]),
                 np.array([np.std([r[k] for r in rows if r['p'] == p]) for p in ps]))
fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
series = [('S1_real', r'Cheb. basis, shared $\Theta$, real $c$', C['real'], '-o'),
          ('S1_pos', r'Cheb. basis, shared $\Theta$, positive $c$ (L-BFGS)', C['pos'], '-s'),
          ('S2_real', r'dist. basis, shared $\Theta$, real $c$', C['cheb'], '--o'),
          ('S2_pos', r'dist. basis, shared $\Theta$, positive $c$ (L-BFGS)', C['pos'], '--s'),
          ('S2_attn', 'dist. basis, row-softmax attention (L-BFGS)', C['attn'], '--^'),
          ('S3_real', 'mismatched (orig. protocol), real $c$', C['mis'], ':D'),
          ('S3_attn', 'mismatched, row-softmax attention (L-BFGS)', C['soft'], ':v')]
for k, lab, col, st in series:
    m, s = agg(k); axs[0].errorbar(xp, m, yerr=s, fmt=st, color=col, capsize=2, ms=5, label=lab)
axs[0].set_xscale('symlog', linthresh=0.02); axs[0].set_ylim(0, 0.75)
labels(axs[0], 'rewiring probability $p$', 'normalized operator error'); title(axs[0], '(a) Matched comparisons')
for k, lab, col, mk in (('d_shell', r'$\delta_{\mathrm{shell}}$', C['real'], 'o'),
                        ('d_int', r'$\delta_{\mathrm{int}}$', C['soft'], 's')):
    m, s = agg(k); axs[1].errorbar(xp, m, yerr=s, marker=mk, color=col, capsize=2, label=lab)
axs[1].set_xscale('symlog', linthresh=0.02)
labels(axs[1], 'rewiring probability $p$', 'non-uniformity'); title(axs[1], '(b) Departure from DRG')
legend(axs[1], loc='upper left')
dx = np.array([r['d_int'] for r in rows]); dy = np.array([r['S2_attn'] - r['S2_pos'] for r in rows])
pcol = plt.cm.viridis(np.linspace(0, 0.9, len(ps)))
for i, p in enumerate(ps):
    sel = np.array([r['p'] == p for r in rows])
    axs[2].scatter(dx[sel], dy[sel], color=pcol[i], s=30, label=f'p={p}')
cc = np.corrcoef(dx, dy)[0, 1]
axs[2].axhline(0, color='grey', lw=0.8)
labels(axs[2], r'$\delta_{\mathrm{int}}$ (per graph)', 'S2: row-softmax $-$ positive $c$')
title(axs[2], f'(c) Paired difference, r = {cc:.2f}')
legend(axs[2], loc='upper left', ncol=2)
h, l = axs[0].get_legend_handles_labels()
leg = fig.legend(h, l, loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.15))
for t in leg.get_texts(): t.set_fontweight('bold')
plt.tight_layout(); plt.savefig('figs/fig_E2_perturbation.pdf'); plt.close()
for p in ps:
    sel = [r for r in rows if r['p'] == p]
    txt.append(f"E2 p={p} n={len(sel)} " + ' '.join(f"{k}={np.mean([r[k] for r in sel]):.3f}+-{np.std([r[k] for r in sel]):.3f}"
               for k in ['S1_real', 'S1_pos', 'S2_real', 'S2_pos', 'S2_attn', 'S3_real', 'S3_attn', 'd_shell', 'd_int']))
pp = [r for r in rows if r['p'] > 0]
txt.append('E2 summary: mean(S3_real-S1_real | p>0)=%.3f mean(S2_attn-S2_pos | p>0)=%.3f corr(S1_real,d_int)=%.2f corr(S2_attn-S2_pos,d_int)=%.2f corr(S2_attn,d_int)=%.2f' % (
    np.mean([r['S3_real'] - r['S1_real'] for r in pp]), np.mean([r['S2_attn'] - r['S2_pos'] for r in pp]),
    np.corrcoef([r['d_int'] for r in rows], [r['S1_real'] for r in rows])[0, 1],
    np.corrcoef([r['d_int'] for r in rows], [r['S2_attn'] - r['S2_pos'] for r in rows])[0, 1],
    np.corrcoef([r['d_int'] for r in rows], [r['S2_attn'] for r in rows])[0, 1]))
txt.append('E2 S2_attn<S2_real graphs: %d of %d; S2_attn<S2_pos graphs: %d' % (sum(r['S2_attn'] < r['S2_real'] - 1e-9 for r in rows), len(rows), sum(r['S2_attn'] < r['S2_pos'] - 1e-9 for r in rows)))
txt.append('E2 positivity cost per p: S1 ' + ' '.join('%.3f' % (agg('S1_pos')[0][i] - agg('S1_real')[0][i]) for i in range(len(ps))) +
           ' | S2 ' + ' '.join('%.3f' % (agg('S2_pos')[0][i] - agg('S2_real')[0][i]) for i in range(len(ps))))

# ======================================================================= E3
E3 = J('results_E3.json')
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, dk, t in zip(axs, ('d2', 'd3'), ('(a) $d=2$, rank $\\leq 4$', '(b) $d=3$, rank $=6$')):
    Hs = sorted(int(h) for h in E3[dk] if h.isdigit())
    for kind, lab, col, mk in (('real', 'unconstrained (exact minimum)', C['real'], 'o'),
                               ('soft', 'softmax heads (L-BFGS fit)', C['soft'], 's')):
        ax.errorbar(Hs, [E3[dk][str(h)][f'{kind}_mean'] for h in Hs], yerr=[E3[dk][str(h)][f'{kind}_std'] for h in Hs],
                    marker=mk, color=col, capsize=3, label=lab)
    ax.set_xticks(Hs); labels(ax, 'number of heads $H$', 'normalized operator error'); title(ax, t); legend(ax)
    txt.append(f'E3 {dk} ' + ' '.join(f"H{h}: real {E3[dk][str(h)]['real_mean']:.4f}+-{E3[dk][str(h)]['real_std']:.3f} soft {E3[dk][str(h)]['soft_mean']:.4f}+-{E3[dk][str(h)]['soft_std']:.3f} z{E3[dk][str(h)]['soft_zero_frac']:.2f}" for h in Hs) +
               f" constr {E3[dk]['constructive_max_err']:.1e}")
plt.tight_layout(); plt.savefig('figs/fig_E3_multihead.pdf'); plt.close()

# ======================================================================= E4
E4 = J('results_E4.json')
fig, axs = plt.subplots(2, 2, figsize=(13, 10))
ax = axs[0, 0]; La = np.arange(E4['config']['L_fixed'] + 1)
for kind, col in (('global', '#1f77b4'), ('local', '#ff7f0e')):
    r = E4['E4a'][kind]
    ax.semilogy(La, r['median'], '-o', color=col, ms=4, lw=2, label=f'{kind} (median)')
    ax.fill_between(La, r['q10'], r['q90'], color=col, alpha=.15)
    ax.semilogy(La, r['bound'], '--', color=col, lw=1.6, label=rf'{kind} bound $\rho^{{2L}}$, $\rho$={r["rho"]:.3f}')
labels(ax, 'layer $L$', '$R_L$'); title(ax, '(a) Fixed operators'); legend(ax, loc='lower left')
Lb = np.arange(E4['config']['L_layer'] + 1); cols = {0.6: '#1f77b4', 0.9: '#ff7f0e', 1.1: '#d62728'}
for ax, regime, t in ((axs[0, 1], 'tight', '(b) Near-worst case'), (axs[1, 0], 'generic', '(c) Generic case')):
    for kind, st in (('global', '-'), ('local', '-.')):
        for rs in (0.6, 0.9, 1.1):
            r = E4['E4bc'][f'{regime}_{kind}_{rs}']
            ax.semilogy(Lb, r['median'], st, color=cols[rs], lw=2, label=f'{kind}, $\\rho s$={rs}')
            ax.fill_between(Lb, r['q10'], r['q90'], color=cols[rs], alpha=.1)
    for rs in (0.6, 0.9, 1.1):
        ax.semilogy(Lb, [rs ** (2 * l) for l in Lb], ':', color=cols[rs], lw=1.4)
    ax.plot([], [], 'k:', label=r'bound $(\rho s)^{2L}$')
    labels(ax, 'layer $L$', '$R_L$'); title(ax, t); legend(ax, ncol=2, loc='lower left')
ax = axs[1, 1]
for name, col, lab in (('uniform', '#1f77b4', r'$\rho_\ell=0.9$'), ('harmonic', '#ff7f0e', r'$\rho_\ell=1-0.5/(\ell+1)$'),
                       ('summable', '#d62728', r'$\rho_\ell=1-0.5/(\ell+1)^2$')):
    r = E4['E4d'][name]; ax.loglog(np.arange(1, len(r['curve']) + 1), r['curve'], color=col, lw=2, label=lab)
labels(ax, 'layer $L+1$', '$R_L$'); title(ax, '(d) Non-uniform contraction'); legend(ax, loc='lower left')
plt.tight_layout(); plt.savefig('figs/fig_E4_oversmoothing_drg.pdf'); plt.close()
for k, r in E4['E4a'].items():
    txt.append(f"E4a {k}: coeffs {r['coeffs']} spectrum {r['spectrum']} rho {r['rho']:.4f} R20 {r['median'][-1]:.3e} bound {r['bound'][-1]:.3e} viol {r['violations']}")
for k, r in E4['E4bc'].items():
    txt.append(f"E4bc {k}: rho {r['rho_median']:.3f} [{r['rho_range'][0]:.3f},{r['rho_range'][1]:.3f}] R30 {r['median'][-1]:.3e} bound {r['bound'][-1]:.3e} viol {r['violations']}")
for k, r in E4['E4d'].items():
    txt.append(f"E4d {k}: sup rho {r['sup_rho']:.6f} R30 {r['curve'][30]:.3e} R400 {r['final']:.3e} prod rho^2 {r['prod_rho2']:.3e}")

# ======================================================================= E5
E5 = J('results_E5.json')
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
gc = {'C20': '#1f77b4', 'Petersen': '#ff7f0e', 'Karate club': '#2ca02c', 'Small-world': '#d62728'}
mk = {'C20': 'o', 'Petersen': 's', 'Karate club': 'D', 'Small-world': '^'}
ax = axs[0]
for name, r in E5['E5a'].items():
    Lx = np.arange(len(r['median']))
    ax.semilogy(Lx, r['median'], '-', marker=mk[name], markevery=5, color=gc[name], lw=2, label=f'{name}')
    ax.fill_between(Lx, r['q10'], r['q90'], color=gc[name], alpha=.12)
    ax.semilogy(Lx, r['bound_median'], '--', color=gc[name], lw=1.4)
ax.plot([], [], 'k--', label=r'bound $\rho^{2L}$ (median)')
labels(ax, 'layer $L$', '$R_L$'); title(ax, '(a) Fixed symmetric operators'); legend(ax, loc='lower left')
ax = axs[1]
for name, r in E5['E5b'].items():
    Lx = np.arange(len(r['median']))
    ax.semilogy(Lx, r['median'], color=gc[name], lw=2, label=f'{name} (median)')
    ax.fill_between(Lx, r['q10'], r['q90'], color=gc[name], alpha=.15)
    ax.semilogy(Lx, r['bound_median'], '--', color=gc[name], lw=1.5, label=rf'{name}, $\prod c_\ell^2$ bound')
labels(ax, 'layer $L$', '$R_L$'); title(ax, '(b) Non-commuting operators'); legend(ax, loc='lower left')
plt.tight_layout(); plt.savefig('figs/fig_E5_oversmoothing_general.pdf'); plt.close()
for k, r in E5['E5a'].items():
    txt.append(f"E5a {k}: n {r['n']} rho median {r['rho_median']:.3f} range [{r['rho_range'][0]:.3f},{r['rho_range'][1]:.3f}] orig {r['rho_original_fig4']:.3f} "
               f"lmin {r['lambda_min_median']:.3f} score {r['score_mean']:.3f}+-{r['score_std']:.3f} R40 {r['median'][-1]:.3e} bound {r['bound_median'][-1]:.3e} viol {r['violations']}")
for k, r in E5['E5b'].items():
    txt.append(f"E5b {k}: R30 {r['median'][-1]:.3e} bound {r['bound_median'][-1]:.3e} viol {r['violations']} comm {r['rel_commutator_median']:.3f}")

# ======================================================================= E6
if os.path.exists('results_E6.json'):
    E6 = J('results_E6.json'); M = E6['models']
    names = {'gcn': 'GCN', 'dilocal': 'DI-local', 'diglobal': 'DI-global', 'diglobal4': 'DI-global-4h',
             'gat1': 'GAT-1h', 'gat4': 'GAT-4h'}
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
    ks = list(names); x = np.arange(len(ks)); w = 0.38
    ax = axs[0]
    ax.bar(x - w / 2, [M[k]['aniso_mean'] for k in ks], w, yerr=[M[k]['aniso_std'] for k in ks], capsize=3,
           color='#1f77b4', label='distance-class anisotropy')
    ax.bar(x + w / 2, [M[k]['offspec_mean'] for k in ks], w, yerr=[M[k]['offspec_std'] for k in ks], capsize=3,
           color='#d62728', label='off-spectral energy')
    ax.set_xticks(x, [names[k] for k in ks], rotation=25)
    labels(ax, 'model (layer 1)', 'fraction'); title(ax, '(a) Structural diagnostics'); legend(ax, loc='upper left')
    D = E6['depth']; Ls = sorted({v['L'] for v in D.values()})
    dc = {'gcn': '#2ca02c', 'dilocal': '#1f77b4', 'gat1': '#d62728'}
    ax = axs[1]
    for k in ('gcn', 'dilocal', 'gat1'):
        ax.semilogy(Ls, [D[f'{k}_L{L}']['phi_median'] for L in Ls], '-o', color=dc[k], lw=2, label=f'{names[k]}, trained')
        ax.semilogy(Ls, [D[f'{k}_L{L}']['phi_init_median'] for L in Ls], '--s', color=dc[k], lw=1.5, label=f'{names[k]}, init.')
    ax.set_xscale('log', base=2); ax.set_xticks(Ls, [str(L) for L in Ls])
    labels(ax, 'depth $L$', r'$\phi$ (last hidden layer)'); title(ax, '(b) Non-principal energy'); legend(ax, loc='lower left', ncol=1)
    ax = axs[2]
    for k in ('gcn', 'dilocal', 'gat1'):
        ax.errorbar(Ls, [D[f'{k}_L{L}']['acc_mean'] for L in Ls], yerr=[D[f'{k}_L{L}']['acc_std'] for L in Ls],
                    marker='o', color=dc[k], capsize=3, lw=2, label=names[k])
    ax.set_xscale('log', base=2); ax.set_xticks(Ls, [str(L) for L in Ls])
    labels(ax, 'depth $L$', 'test accuracy'); title(ax, '(c) Accuracy vs depth'); legend(ax, loc='lower left')
    plt.tight_layout(); plt.savefig('figs/fig_E6_real_graph.pdf'); plt.close()
    txt.append('E6 config ' + json.dumps(E6['config']))
    for k in ks:
        r = M[k]
        txt.append(f"E6 {names[k]}: acc {r['acc_mean']:.3f}+-{r['acc_std']:.3f} aniso {r['aniso_mean']:.3f}+-{r['aniso_std']:.3f} "
                   f"offspec {r['offspec_mean']:.3f}+-{r['offspec_std']:.3f} rho1 {r['rho1_mean']:.3f} s1 {r['s1_mean']:.2f} "
                   f"agg {r.get('aggregation')} c_med {['%.2f' % v for v in r['c_layers_median']]} c_min {['%.2f' % v for v in r['c_layers_min']]} "
                   f"g_med {['%.2f' % v for v in r['g_layers_median']]} g_min {['%.2f' % v for v in r['g_layers_min']]} "
                   f"c<1 {r['n_c_below_1']}/{r['n_layer_evals']} g<1 {r['n_g_below_1']}/{r['n_layer_evals']} "
                   f"erank {r['erank_mean']} +- {r['erank_std']} vertex range [{r['erank_vertex_min']},{r['erank_vertex_max']}] surrogate {r['erank_surrogate_mean']}")
    txt.append('E6 reference ' + json.dumps(E6['reference']))
    for k, r in D.items():
        txt.append(f"E6 depth {k}: acc {r['acc_mean']:.3f}+-{r['acc_std']:.3f} phi {r['phi_median']:.3e} [{r['phi_q10']:.2e},{r['phi_q90']:.2e}] "
                   f"log10prodc {r['log10_prod_c_median']:.2f} | init phi {r['phi_init_median']:.3e} log10prodc {r['log10_prod_c_init_median']:.2f} "
                   f"| c<1 trained {r['n_c_below_1_trained']}/{r['n_hidden_layer_evals']} init {r['n_c_below_1_init']}/{r['n_hidden_layer_evals']}")

# ======================================================================= E7
if os.path.exists('results_E7.json'):
    E7 = J('results_E7.json'); P7 = E7['partA']
    fig, axs = plt.subplots(1, 5, figsize=(18, 4.6), gridspec_kw=dict(wspace=0.35))
    tgt = np.array(P7['target'])
    panels = [('H1_real', '(b) H=1, real', 'eps'), ('H1_soft', '(c) H=1, softmax', 'eps_fit'),
              ('H2_soft', '(d) H=2, softmax', 'eps_fit'), ('H3_soft', '(e) H=3, softmax', 'eps_fit')]
    vmax = max(np.abs(np.array(P7[k]['residual'])).max() for k, _, _ in panels)
    tv = np.abs(tgt).max()
    cols = [r'$\theta_{11}$', r'$\theta_{21}$', r'$\theta_{12}$', r'$\theta_{22}$']
    im0 = axs[0].imshow(tgt, cmap='RdBu_r', vmin=-tv, vmax=tv, aspect='auto')
    title(axs[0], '(a) Target'); labels(axs[0], r'entry of vec$(\Theta^\star_k)$', 'relation $k$')
    cb = fig.colorbar(im0, ax=axs[0], fraction=0.05); cb.ax.tick_params(labelsize=10)
    for ax, (k, t, key) in zip(axs[1:], panels):
        im = ax.imshow(np.array(P7[k]['residual']), cmap='PuOr_r', vmin=-vmax, vmax=vmax, aspect='auto')
        title(ax, t)
        if k == 'H1_real':
            xl = rf'$\varepsilon$={P7[k]["eps"]:.3f} (exact)'
        else:
            xl = rf'fit {P7[k]["eps_fit"]:.3f}; bound {P7[k]["lower_bound_exact"]:.3f}'
        labels(ax, xl, '')
    for ax in axs:
        ax.set_xticks(range(4), cols); ax.set_yticks(range(tgt.shape[0])); ax.grid(False)
    cb = fig.colorbar(im, ax=list(axs[1:]), fraction=0.02, pad=0.02); cb.ax.tick_params(labelsize=10)
    cb.set_label('residual', fontsize=12, fontweight='bold')
    plt.savefig('figs/fig_E7_residuals.pdf'); plt.close()
    txt.append('E7A gram diag ' + str(np.round(P7['gram_diag'], 4)) + ' offdiag max %.2e' % P7['gram_offdiag_max'])
    txt.append('E7A H1 real exact %.4f' % P7['H1_real']['eps'] + ' | ' + ' | '.join(
        f"{k}: fit {P7[k]['eps_fit']:.4f} check {P7[k]['eps_check']:.4f} LB {P7[k]['lower_bound_exact']:.4f}" for k in ('H1_soft', 'H2_soft', 'H3_soft')))
    # ---- supplementary figure S1
    Q7 = E7['partB']
    import networkx as nx
    fig, axs = plt.subplots(2, 4, figsize=(18, 9), gridspec_kw=dict(width_ratios=[1.1, 0.55, 1, 1]))
    for row, (name, lab) in enumerate((('C20', '$C_{20}$'), ('rewired', 'rewired, $p=0.2$'))):
        r = Q7[name]; Gx = nx.Graph(); Gx.add_nodes_from(range(20)); Gx.add_edges_from(r['edges'])
        dist = nx.single_source_shortest_path_length(Gx, 0)
        colr = [min(dist[v], 4) for v in range(20)]
        ax = axs[row, 0]; pos = nx.circular_layout(Gx)
        nx.draw_networkx_edges(Gx, pos, ax=ax, edge_color='grey')
        sc = ax.scatter([pos[v][0] for v in range(20)], [pos[v][1] for v in range(20)], c=colr, cmap='viridis', vmin=0, vmax=4, s=140, zorder=3, edgecolors='k')
        ax.set_aspect('equal'); ax.axis('off'); title(ax, f'({"ae"[row]}) {lab}: shells of 0')
        cb = fig.colorbar(sc, ax=ax, fraction=0.04, ticks=range(5)); cb.ax.set_yticklabels(['0', '1', '2', '3', '>3']); cb.ax.tick_params(labelsize=10)
        ax = axs[row, 1]; im = ax.imshow(np.array(r['shells']), cmap='magma_r', aspect='auto', vmin=0, vmax=6)
        ax.set_xticks(range(4)); title(ax, f'({"bf"[row]}) $n_r(u)$'); labels(ax, 'distance $r$', 'vertex $u$'); ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.08); cb.ax.tick_params(labelsize=10)
        ax = axs[row, 2]; im = ax.imshow(np.array(r['P']), cmap='Blues', aspect='equal', vmin=0)
        title(ax, f'({"cg"[row]}) attention $\\mathcal{{A}}$'); labels(ax, 'vertex $v$', 'vertex $u$'); ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046); cb.ax.tick_params(labelsize=10)
        ax = axs[row, 3]; M = np.array(r['coupling'])
        from matplotlib.colors import LogNorm
        im = ax.imshow(np.maximum(M, 1e-8), cmap='inferno', norm=LogNorm(vmin=1e-6, vmax=1), aspect='equal')
        title(ax, f'({"dh"[row]}) coupling, off = {r["offspectral"]:.3f}'); labels(ax, 'eigenspace $k$', 'eigenspace $j$'); ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046); cb.ax.tick_params(labelsize=10)
    from matplotlib.ticker import MaxNLocator
    for a in axs[:, 1:].ravel():
        a.xaxis.set_major_locator(MaxNLocator(integer=True)); a.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout(); plt.savefig('figs/fig_S1_rewiring.pdf'); plt.close()
    for name in ('C20', 'rewired'):
        r = Q7[name]
        txt.append(f"E7B {name}: offspectral {r['offspectral']:.4f} normalizer range {r['row_sum_range']} symmetric {r['symmetric']} n_eigenspaces {len(r['eigenvalues'])} seed {r['seed']}")

open('tables_for_text.txt', 'w').write('\n'.join(txt))
print('\n'.join(txt))
