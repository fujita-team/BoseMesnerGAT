"""
Builds the six figures of Section 7 from results_E1..E6.json:
  figs/fig_E1_cycle.pdf, figs/fig_E2_perturbation.pdf, figs/fig_E3_multihead.pdf,
  figs/fig_E4_oversmoothing_drg.pdf, figs/fig_E5_oversmoothing_general.pdf,
  figs/fig_E6_real_graph.pdf
and writes every number quoted in the text to tables_for_text.txt.
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
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
keys = ['K3_d1', 'K3_d2', 'K5_d1', 'K5_d2']; x = np.arange(4); w = 0.26
axs[0].bar(x - w, [0] * 4, w, color=C['cheb'], label='ChebNet')
axs[0].bar(x, [A[k]['shared_real_mean'] for k in keys], w, yerr=[A[k]['shared_real_std'] for k in keys],
           capsize=3, color=C['real'], label=r'shared $\Theta$, real $c_k$')
axs[0].bar(x + w, [A[k]['shared_soft_mean'] for k in keys], w, yerr=[A[k]['shared_soft_std'] for k in keys],
           capsize=3, color=C['soft'], label=r'shared $\Theta$, softmax (GAT)')
for xi in x:
    axs[0].text(xi - w, 0.008, '0', ha='center', va='bottom', color=C['cheb'], fontsize=10, fontweight='bold')
axs[0].set_xticks(x, [f'K={k[1]}, d={k[-1]}' for k in keys])
labels(axs[0], 'order $K$, feature dimension $d$', 'min. normalised error')
title(axs[0], '(a) Exact minimum error'); legend(axs[0], loc='upper left')
steps = (np.arange(len(B[0]['cheb']['curve'])) + 1) * 50
for kind, lab in (('cheb', 'ChebNet'), ('real', r'shared, real $c_k$'), ('soft', 'shared, softmax')):
    cur = np.array([r[kind]['curve'] for r in B])
    axs[1].semilogy(steps, np.maximum(np.median(cur, 0), 1e-33), color=C[kind], lw=2, label=f'{lab} (train)')
    te = np.maximum([r[kind]['test'] for r in B], 1e-33)
    axs[1].scatter([steps[-1] * 1.04] * len(te), te, color=C[kind], marker='x', s=30, zorder=5)
for kind in ('real', 'soft'):
    axs[1].axhline(np.median([r[f'floor_shared_{kind}'] for r in B]), color=C[kind], ls=':', lw=1.6)
axs[1].scatter([], [], color='k', marker='x', label='test, 2000 new signals')
axs[1].plot([], [], 'k:', label='exact minimum, panel (a)')
labels(axs[1], 'Adam step', 'relative MSE'); title(axs[1], '(b) Training, K=3, d=2')
legend(axs[1], loc='center right')
plt.tight_layout(); plt.savefig('figs/fig_E1_cycle.pdf'); plt.close()
txt.append('E1A ' + json.dumps(A))
for r in B:
    txt.append(f"E1B t{r['target']} floor real {r['floor_shared_real']:.4f} soft {r['floor_shared_soft']:.4f} | " +
               ' '.join(f"{k}: tr {r[k]['train']:.3g} te {r[k]['test']:.3g}" for k in ('cheb', 'real', 'soft')))

# ======================================================================= E2
E2 = J('results_E2.json'); rows = E2['rows']; ps = E2['config']['ps']; xp = np.array(ps)
agg = lambda k: (np.array([np.mean([r[k] for r in rows if r['p'] == p]) for p in ps]),
                 np.array([np.std([r[k] for r in rows if r['p'] == p]) for p in ps]))
fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
series = [('S1_real', r'Cheb. basis, shared $\Theta$, real $c$', C['real'], '-o'),
          ('S1_pos', r'Cheb. basis, shared $\Theta$, positive $c$', C['pos'], '-s'),
          ('S2_real', r'dist. basis, shared $\Theta$, real $c$', C['cheb'], '--o'),
          ('S2_attn', 'dist. basis, row-softmax attention', C['attn'], '--^'),
          ('S3_real', 'mismatched (orig. protocol), real $c$', C['mis'], ':D'),
          ('S3_attn', 'mismatched, row-softmax attention', C['soft'], ':v')]
for k, lab, col, st in series:
    m, s = agg(k); axs[0].errorbar(xp, m, yerr=s, fmt=st, color=col, capsize=2, ms=5, label=lab)
axs[0].set_xscale('symlog', linthresh=0.02); axs[0].set_ylim(0, 0.75)
labels(axs[0], 'rewiring probability $p$', 'min. normalised error'); title(axs[0], '(a) Error decomposition')
for k, lab, col, mk in (('d_shell', r'$\delta_{\mathrm{shell}}$', C['real'], 'o'),
                        ('d_int', r'$\delta_{\mathrm{int}}$', C['soft'], 's')):
    m, s = agg(k); axs[1].errorbar(xp, m, yerr=s, marker=mk, color=col, capsize=2, label=lab)
axs[1].set_xscale('symlog', linthresh=0.02)
labels(axs[1], 'rewiring probability $p$', 'nonuniformity'); title(axs[1], '(b) Departure from DRG')
legend(axs[1], loc='upper left')
for k, lab, col, mk in (('S1_real', r'Cheb. basis, real $c$', C['real'], 'o'),
                        ('S2_attn', 'row-softmax attention', C['attn'], '^')):
    axs[2].scatter([r['d_int'] for r in rows], [r[k] for r in rows], color=col, marker=mk, s=22, alpha=.75, label=lab)
axs[2].set_ylim(0, 0.75)
labels(axs[2], r'$\delta_{\mathrm{int}}$ (per graph)', 'min. normalised error'); title(axs[2], '(c) Per-graph errors')
legend(axs[2], loc='lower right')
h, l = axs[0].get_legend_handles_labels()
leg = fig.legend(h, l, loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.12))
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
txt.append('E2 positivity cost per p: S1 ' + ' '.join('%.3f' % (agg('S1_pos')[0][i] - agg('S1_real')[0][i]) for i in range(len(ps))) +
           ' | S2 ' + ' '.join('%.3f' % (agg('S2_pos')[0][i] - agg('S2_real')[0][i]) for i in range(len(ps))))

# ======================================================================= E3
E3 = J('results_E3.json')
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, dk, t in zip(axs, ('d2', 'd3'), ('(a) $d=2$, rank $\\leq 4$', '(b) $d=3$, rank $=6$')):
    Hs = sorted(int(h) for h in E3[dk] if h.isdigit())
    for kind, lab, col, mk in (('real', 'unconstrained (exact minimum)', C['real'], 'o'),
                               ('soft', 'softmax heads (GAT)', C['soft'], 's')):
        ax.errorbar(Hs, [E3[dk][str(h)][f'{kind}_mean'] for h in Hs], yerr=[E3[dk][str(h)][f'{kind}_std'] for h in Hs],
                    marker=mk, color=col, capsize=3, label=lab)
    ax.set_xticks(Hs); labels(ax, 'number of heads $H$', 'min. normalised error'); title(ax, t); legend(ax)
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
                   f"c {['%.2f' % v for v in r['c_layers_mean']]} erank {r['erank_mean']} +- {r['erank_std']}")
    txt.append('E6 reference ' + json.dumps(E6['reference']))
    for k, r in D.items():
        txt.append(f"E6 depth {k}: acc {r['acc_mean']:.3f}+-{r['acc_std']:.3f} phi {r['phi_median']:.3e} [{r['phi_q10']:.2e},{r['phi_q90']:.2e}] "
                   f"log10prodc {r['log10_prod_c_median']:.2f} | init phi {r['phi_init_median']:.3e} log10prodc {r['log10_prod_c_init_median']:.2f}")

open('tables_for_text.txt', 'w').write('\n'.join(txt))
print('\n'.join(txt))
