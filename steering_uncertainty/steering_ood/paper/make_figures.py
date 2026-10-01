"""Experiment figures for the paper (all from saved runs; see RUNBOOK.md for the layout).

Run from steering_ood/:  python paper/make_figures.py [runs/modal]
Writes paper/figures/fig_*.pdf:
  fig_teaser       static AUROC vs rejection on radial paths (RQ1)
  fig_failure_map  matched-horizon rejection, path family x detector (RQ2)
  fig_theory       Proposition 3 predictions vs observations (RQ3)
  fig_support      vision principal-axis geodesics vs ID support (RQ4)
  fig_mechanism    energy and kNN scores along real radial paths
  fig_curves_seeds R(alpha) with seed bands (appendix)
  fig_real_ood     feature-norm ratio vs energy-over-distance AUROC gap (appendix)
  fig_gauge        energy decisions under a prediction-invariant row shift (Prop. 2)
  fig_realise      natural transformations: radial/angular motion and detector response (needs runs/modal/realise)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import logsumexp, softmax

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG)); sys.path.insert(0, str(PKG / 'scripts'))
from steering_ood.core import threshold  # noqa: E402
from steering_ood.detectors import SKKNN, ShrinkageMahalanobis  # noqa: E402
from steering_ood.experiment import path_points  # noqa: E402
import seed_stats as ss  # noqa: E402

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else PKG / 'runs' / 'modal').resolve()
ss.ROOT = ROOT
OUT = HERE / 'figures'
OUT.mkdir(exist_ok=True)

plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8,
                     'axes.titlesize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6,
                     'pdf.fonttype': 42})
TEXT, VISION = ss.MODALITIES['text'], ss.MODALITIES['vision']
NAMES = {'mpnet': 'MPNet', 'minilm': 'MiniLM', 'bge_base': 'BGE-base', 'bge_large': 'BGE-large',
         'resnet18': 'ResNet-18', 'resnet50': 'ResNet-50', 'vit_b16': 'ViT-B/16', 'dinov2_s': 'DINOv2-S'}
DETS = {'knn': 'kNN', 'mahalanobis_shrinkage': 'Mahalanobis', 'energy_T1': 'Energy', 'msp_T1': 'MSP'}
# distance detectors in blues, logit detectors in warm colours (Okabe-Ito); distinct line styles and markers
DCOL = {'knn': '#0072B2', 'mahalanobis_shrinkage': '#56B4E9', 'energy_T1': '#D55E00', 'msp_T1': '#E69F00'}
DLS = {'knn': '-', 'mahalanobis_shrinkage': '--', 'energy_T1': '-.', 'msp_T1': ':'}
DMK = {'knn': 'o', 'mahalanobis_shrinkage': 's', 'energy_T1': '^', 'msp_T1': 'D'}
PATH_ORDER = ['radial', 'pca', 'random', 'sphere_pca', 'sphere_random', 'origin']
PATH_LABEL = {'radial': 'radial growth', 'pca': 'straight, PC', 'random': 'straight, random',
              'sphere_pca': 'geodesic, PC', 'sphere_random': 'geodesic, random', 'origin': 'toward origin'}
SEEDS = ss.SEEDS


def modality(m):
    return 'text' if m in TEXT else 'vision'


def cell(seed, model):
    return ROOT / 'seeds' / f'seed{seed}' / model / 'full'


def cache(seed, model):
    return np.load(ROOT / 'caches' / (f'seed{seed}' if seed else '') / f'{model}.npz')


def head(seed, model):
    st = torch.load(cell(seed, model) / 'head' / 'head.pt')
    return st['weight'].double().numpy(), st['bias'].double().numpy()


def energy(z, W, b):
    return -logsumexp(z @ W.T + b, axis=-1)


def radius(seed, model, kind='radial'):
    return json.loads((cell(seed, model) / f'crossed_{kind}' / 'steering_checks.json').read_text())['id_reference_radius']


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('wrote', OUT / f'{name}.pdf')


A_M = {mod: ss.matched_alpha(mod) for mod in ss.MODALITIES}


def matched_R(seed, model, path, det):
    return ss.indicators(seed, model, path, det, A_M[modality(model)]).mean()


# ---------------------------------------------------------------- fig 2: teaser (RQ1)
def fig_teaser():
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.5), sharey=True)
    for ax, path in zip(axes, ('radial', 'origin')):
        for m in TEXT + VISION:
            for det in DETS:
                au = np.mean([json.loads((cell(s, m) / 'static' / 'metrics.json').read_text())[det]['auroc'] for s in SEEDS])
                r = np.mean([matched_R(s, m, path, det) for s in SEEDS])
                ax.scatter(au, r, s=22, marker='o' if m in TEXT else 's', facecolor=DCOL[det],
                           edgecolor='k', linewidth=.3, zorder=3)
        ax.axhline(.05, c='.6', lw=.5, ls=(0, (1, 2)))
        ax.set_xlabel('static AUROC'); ax.set_title(f'rejection on {PATH_LABEL[path]}')
        ax.set_xlim(.35, 1.0); ax.set_ylim(-.04, 1.04); ax.grid(alpha=.25, lw=.4)
    axes[0].set_ylabel('fraction of steered probes rejected', fontsize=7)
    h = [plt.Line2D([], [], ls='', marker='o', mfc=DCOL[d], mec='k', mew=.3, label=DETS[d]) for d in DETS]
    h += [plt.Line2D([], [], ls='', marker='o', mfc='w', mec='k', label='text encoder'),
          plt.Line2D([], [], ls='', marker='s', mfc='w', mec='k', label='vision encoder')]
    fig.tight_layout(rect=(0, .1, 1, 1))
    fig.legend(handles=h, loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(.5, 0))
    save(fig, 'fig_teaser')


# ---------------------------------------------------------------- fig 3: failure map (RQ2)
def fig_failure_map():
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.6))
    for ax, mod in zip(axes, ss.MODALITIES):
        M = np.array([[np.mean([matched_R(s, m, p, d) for s in SEEDS for m in ss.MODALITIES[mod]])
                       for d in DETS] for p in PATH_ORDER])
        im = ax.imshow(M, cmap='cividis', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(DETS))); ax.set_xticklabels(DETS.values())
        ax.set_yticks(range(len(PATH_ORDER))); ax.set_yticklabels([PATH_LABEL[p] for p in PATH_ORDER])
        ax.set_title(f'{mod} (matched horizon)')
        ax.axvline(1.5, c='w', lw=1.2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        for xpos, lab in ((.5, 'distance-based'), (2.5, 'logit-based')):
            ax.text(xpos, len(PATH_ORDER) - .5 + .55, lab, ha='center', va='top', fontsize=7, style='italic',
                    transform=ax.transData)
    axes[1].set_yticklabels([])
    cb = fig.colorbar(im, ax=axes, fraction=.025, pad=.02)
    cb.set_label('fraction rejected')
    save(fig, 'fig_failure_map')


# ---------------------------------------------------------------- fig 4: theory vs observation (RQ3)
def fig_theory():
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.7))
    ax = axes[0]
    for s in SEEDS:
        for m in TEXT + VISION:
            c, (W, b) = cache(s, m), head(s, m)
            run = cell(s, m) / 'crossed_origin'
            d = np.load(run / 'design.npz')
            x = c['probe_x'][d['probe_indices']].astype(float)
            n = np.linalg.norm(x, axis=1); s_end = 1 - d['alphas'][-1] / n; near = s_end < .1
            if not near.any():
                continue
            te = np.load(run / 'energy_T1_traces.npz')
            t = te['thresholds'][0]
            ax.scatter((-logsumexp(b) - t) / abs(t), te['rejected'][..., -1][..., near].mean(), marker='^',
                       s=18, facecolor=DCOL['energy_T1'], edgecolor='k', lw=.3, zorder=3)
            tk = np.load(run / 'knn_traces.npz'); tm = np.load(run / 'mahalanobis_shrinkage_traces.npz')
            for bi, idx in enumerate(d['reference_indices']):
                ref, y = c['reference_x'][idx].astype(float), c['reference_y'][idx]
                rk = np.sort(np.linalg.norm(ref, axis=1))[ss_k - 1]
                dec = near & (abs(rk - tk['thresholds'][bi]) > s_end * n)
                if dec.any():
                    ax.scatter((rk - tk['thresholds'][bi]) / tk['thresholds'][bi],
                               tk['rejected'][bi, ..., -1][..., dec].mean(), marker='o', s=12,
                               facecolor=DCOL['knn'], edgecolor='k', lw=.3, zorder=3)
                if bi == 0:
                    m0 = ShrinkageMahalanobis().fit(ref, y).score(np.zeros((1, ref.shape[1])))[0]
                    tmb = tm['thresholds'][bi]
                    ax.scatter((m0 - tmb) / tmb, tm['rejected'][bi, ..., -1][..., near].mean(), marker='s', s=14,
                               facecolor=DCOL['mahalanobis_shrinkage'], edgecolor='k', lw=.3, zorder=3)
    ax.axvline(0, c='.4', lw=.6); ax.axhline(.5, c='.4', lw=.6, ls=':')
    ax.set_xscale('symlog', linthresh=.05)
    ax.fill_betweenx([-.05, .5], -1e3, 0, color='.92', zorder=0); ax.fill_betweenx([.5, 1.05], 0, 1e3, color='.92', zorder=0)
    ax.set_xlim(-30, 30); ax.set_ylim(-.04, 1.04)
    ax.set_xticks([-10, -1, -.1, 0, .1, 1, 10]); ax.set_xticklabels(['$-10$', '$-1$', '$-0.1$', '0', '0.1', '1', '10'])
    ax.set_xlabel('predicted margin at the origin (relative to threshold)')
    ax.set_ylabel('observed rejection near the origin')
    ax.set_title('(a) origin limits: shaded = prediction agrees')
    ax.legend(handles=[plt.Line2D([], [], ls='', marker=DMK[d], mfc=DCOL[d], mec='k', mew=.3, label=DETS[d])
                       for d in ('knn', 'mahalanobis_shrinkage', 'energy_T1')], loc='center left', frameon=False)
    ax = axes[1]
    for m in TEXT + VISION:
        W, b = head(0, m)
        run = cell(0, m) / 'crossed_radial'
        d = np.load(run / 'design.npz'); c = cache(0, m)
        x = c['probe_x'][d['probe_indices']].astype(float)
        te = np.load(run / 'energy_T1_traces.npz'); t = te['thresholds'][0]
        traced = te['scores'][0, 0, 0].max(-1)
        s_max = 1 + d['alphas'][-1] / np.linalg.norm(x, axis=1)
        pred = []
        for xi, sm in zip(x, s_max):
            grid = np.linspace(1, sm, 2001)
            pred.append(energy(grid[:, None] * xi[None], W, b).max())
        ax.scatter((np.array(pred) - t) / abs(t), (traced - t) / abs(t), s=6, marker='o' if m in TEXT else 's',
                   facecolor=DCOL['energy_T1'], edgecolor='none', alpha=.6)
    lim = ax.get_xlim(); ax.plot(lim, lim, c='.3', lw=.6); ax.axvline(0, c='.4', lw=.5, ls=':'); ax.axhline(0, c='.4', lw=.5, ls=':')
    ax.set_xlabel('closed-form max energy on growth (rel. to threshold)')
    ax.set_ylabel('traced max energy (rel. to threshold)')
    ax.set_title('(b) radial growth: closed form vs traces')
    fig.tight_layout()
    save(fig, 'fig_theory')


ss_k = 5


# ---------------------------------------------------------------- fig 5: ID support (RQ4)
def fig_support():
    bins = np.linspace(0, 6, 25)
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    curves = {d: [] for d in DETS}
    edges = []
    for m in VISION:
        c = cache(0, m); idt = c['id_test_x'].astype(float)
        run = cell(0, m) / 'crossed_sphere_pca'
        d = np.load(run / 'design.npz'); x = c['probe_x'][d['probe_indices']].astype(float)
        Z = []
        for v in d['vectors']:
            p = idt @ v; mu, sd = p.mean(), p.std()
            edges.append(np.quantile(np.abs((p - mu) / sd), .995))
            Z.append([np.abs((path_points('sphere_pca', x, sg * v, d['alphas']) @ v - mu) / sd) for sg in d['signs']])
        Z = np.array(Z)                                           # C, S, P, G
        which = np.digitize(Z, bins) - 1
        for det in DETS:
            rej = np.load(run / f'{det}_traces.npz')['rejected'].mean(0)   # C, S, P, G (avg over reference fits)
            curves[det].append([rej[which == i].mean() if (which == i).sum() > 50 else np.nan for i in range(len(bins) - 1)])
    mid = (bins[1:] + bins[:-1]) / 2
    ax.axvspan(0, np.mean(edges), facecolor='.88', edgecolor='.6', lw=.5, zorder=0, label='ID range (99.5%)')
    for det in DETS:
        ax.plot(mid, np.nanmean(curves[det], 0), c=DCOL[det], ls=DLS[det], marker=DMK[det], ms=3, markevery=3,
                lw=1.1, label=DETS[det])
    ax.set_xlabel('|position along principal axis| (ID s.d.)'); ax.set_ylabel('fraction rejected')
    ax.set_ylim(-.03, 1.03); ax.set_xlim(0, 6); ax.grid(alpha=.25, lw=.4)
    ax.set_title('vision geodesics along the top principal axis')
    ax.legend(frameon=False, loc='upper left', fontsize=6.5)
    save(fig, 'fig_support')


# ---------------------------------------------------------------- fig 6: mechanism on real radial paths
def fig_mechanism():
    fig, axes = plt.subplots(2, 2, figsize=(6.75, 3.6), sharex='col')
    for col, m in enumerate(('mpnet', 'vit_b16')):
        c, (W, b) = cache(0, m), head(0, m)
        rng = np.random.default_rng(0)
        x = c['probe_x'][rng.choice(len(c['probe_x']), 12, replace=False)].astype(float)
        cal = c['calibration_x'].astype(float)
        knn = SKKNN(5).fit(c['reference_x'].astype(float))
        smax = 1 + 3 * radius(0, m) / np.linalg.norm(x, axis=1).min()
        s = np.linspace(0, smax, 200)
        for row, (name, score, t, lim) in enumerate((
                ('energy', lambda z: energy(z, W, b), threshold(energy(cal, W, b), .05), -logsumexp(b)),
                ('kNN', knn.score, threshold(knn.score(cal), .05),
                 np.sort(np.linalg.norm(c['reference_x'], axis=1))[4]))):
            ax = axes[row, col]
            sd = np.std(score(cal))
            for xi in x:
                ax.plot(s, (score(s[:, None] * xi[None]) - t) / sd, c=DCOL['energy_T1' if row == 0 else 'knn'],
                        lw=.6, alpha=.6)
            ax.axhline(0, c='k', lw=.7, ls='--', label='threshold')
            ax.plot([0], [(lim - t) / sd], marker='*', ms=9, c='k', ls='', label='origin limit (Prop. 3)')
            ax.axvline(1, c='.6', lw=.5, ls=':')
            ax.set_ylabel(f'{name} score\n(threshold = 0)')
            if row == 0:
                ax.set_title(NAMES[m])
            if row == 1:
                ax.set_xlabel(r'scale $s$ along $z\mapsto sz$ (probe at $s=1$)')
            ax.grid(alpha=.25, lw=.4)
    axes[0, 0].legend(frameon=False, fontsize=6.5, loc='lower right')
    fig.tight_layout()
    save(fig, 'fig_mechanism')


# ---------------------------------------------------------------- fig 7: R(alpha) with seed bands (appendix)
def fig_curves_seeds():
    grid = np.linspace(0, 3, 301)
    pstyle = {'radial': ('#CC79A7', '-.'), 'origin': ('#009E73', ':'), 'pca': ('#0072B2', '-'),
              'random': ('#E69F00', '-'), 'sphere_pca': ('#0072B2', '--'), 'sphere_random': ('#E69F00', '--')}
    fig, axes = plt.subplots(2, 4, figsize=(6.75, 3.2), sharex=True, sharey=True)
    for r, mod in enumerate(ss.MODALITIES):
        for ci, det in enumerate(DETS):
            ax = axes[r, ci]
            for p in PATH_ORDER:
                per_seed = []
                for s in SEEDS:
                    ys = []
                    for m in ss.MODALITIES[mod]:
                        dd = cell(s, m) / f'crossed_{p}'
                        a = np.load(dd / 'design.npz')['alphas'] / radius(s, m, p)
                        pw = np.load(dd / f'{det}_traces.npz')['power']
                        y = pw.reshape(-1, len(a)).mean(0)
                        ys.append(np.where(grid <= a[-1] + 1e-9, np.interp(grid, a, y), np.nan))
                    per_seed.append(np.mean(ys, 0))
                per_seed = np.array(per_seed); ok = np.isfinite(per_seed).all(0)
                col, ls = pstyle[p]
                ax.fill_between(grid[ok], per_seed.min(0)[ok], per_seed.max(0)[ok], color=col, alpha=.18, lw=0)
                ax.plot(grid[ok], per_seed.mean(0)[ok], c=col, ls=ls, lw=1, label=PATH_LABEL[p])
            ax.axvline(A_M[mod], c='.6', lw=.5); ax.grid(alpha=.25, lw=.4)
            if r == 0:
                ax.set_title(DETS[det])
            if ci == 0:
                ax.set_ylabel(f'{mod}\nfraction rejected')
    fig.supxlabel(r'steering distance $\alpha$ (ID radii)', fontsize=8, y=.06)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(.5, -.04))
    fig.tight_layout(rect=(0, .08, 1, 1))
    save(fig, 'fig_curves_seeds')


# ---------------------------------------------------------------- fig 8: real OOD norm link (appendix)
def fig_real_ood():
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for m in TEXT + VISION:
        c = cache(0, m); met = json.loads((cell(0, m) / 'static' / 'metrics.json').read_text())
        idn = np.median(np.linalg.norm(c['id_test_x'], axis=1))
        groups = c['ood_test_groups'] if 'ood_test_groups' in c.files else np.full(len(c['ood_test_x']), 'oos')
        for g in np.unique(groups):
            by = lambda d: met[d]['ood_by_dataset'][g]['auroc'] if 'ood_by_dataset' in met[d] else met[d]['auroc']
            ratio = np.median(np.linalg.norm(c['ood_test_x'][groups == g], axis=1)) / idn
            gap = by('energy_T1') - max(by('knn'), by('mahalanobis_shrinkage'))
            ax.scatter(ratio, gap, marker='o' if m in TEXT else 's', s=20, facecolor='.75' if g != 'svhn' else '#CC79A7',
                       edgecolor='k', lw=.3, zorder=3)
            if m == 'resnet18' and g == 'svhn':
                ax.annotate('ResNet-18 / SVHN', (ratio, gap), xytext=(8, -4), textcoords='offset points', fontsize=6.5)
    ax.axvline(1, c='.6', lw=.5, ls=':'); ax.axhline(0, c='.6', lw=.5, ls=':')
    ax.set_xlabel('median OOD / ID feature norm'); ax.set_ylabel('AUROC gap\n(energy − best distance)')
    ax.grid(alpha=.25, lw=.4)
    ax.legend(handles=[plt.Line2D([], [], ls='', marker='o', mfc='.75', mec='k', mew=.3, label='text (out-of-scope)'),
                       plt.Line2D([], [], ls='', marker='s', mfc='.75', mec='k', mew=.3, label='vision, CIFAR-100'),
                       plt.Line2D([], [], ls='', marker='s', mfc='#CC79A7', mec='k', mew=.3, label='vision, SVHN')],
              frameon=False, fontsize=6.5, loc='upper right')
    save(fig, 'fig_real_ood')


# ---------------------------------------------------------------- fig 10: gauge sweep (Prop. 2)
def fig_gauge():
    lam = np.linspace(-1, 3, 81)
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.5), sharey=True)
    for ax, path in zip(axes, ('radial', 'origin')):
        for m in TEXT + VISION:
            c, (W, b) = cache(0, m), head(0, m)
            wbar = W.mean(0); cal = c['calibration_x'].astype(float)
            d = np.load(cell(0, m) / f'crossed_{path}' / 'design.npz')
            x = c['probe_x'][d['probe_indices']].astype(float)
            a = A_M[modality(m)] * radius(0, m, path)
            if path == 'origin':
                a = min(a, d['alphas'][-1])
            z = path_points(path, x, np.zeros(x.shape[1]), np.array([a]))[:, 0]
            acc0 = (c['id_test_x'] @ W.T + b).argmax(1)
            R = []
            for l in lam:
                Wl = W - l * wbar
                assert np.array_equal((c['id_test_x'] @ Wl.T + b).argmax(1), acc0)   # predictions unchanged
                t = threshold(energy(cal, Wl, b), .05)
                R.append(np.mean(energy(z, Wl, b) > t))
            ax.plot(lam, R, c=DCOL['energy_T1'], ls='-' if m in TEXT else '--', lw=.9, alpha=.85,
                    marker='o' if m in TEXT else 's', ms=2.5, markevery=10)
        ax.axvline(0, c='.4', lw=.6, ls=':'); ax.axvline(1, c='.4', lw=.6, ls=':')
        ax.text(0, 1.06, 'trained', ha='center', fontsize=6.5); ax.text(1, 1.06, r'$\sum_c w_c=0$', ha='center', fontsize=6.5)
        ax.set_xlabel(r'row shift $\lambda$ ($w_c \mapsto w_c - \lambda \bar w$)')
        ax.set_title(f'({"ab"[path == "origin"]}) {PATH_LABEL[path]}', pad=12)
        ax.set_ylim(-.04, 1.04); ax.grid(alpha=.25, lw=.4)
    axes[0].set_ylabel('energy rejection\nat matched horizon')
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.legend(handles=[plt.Line2D([], [], c=DCOL['energy_T1'], ls='-', marker='o', ms=3, label='text encoders'),
                        plt.Line2D([], [], c=DCOL['energy_T1'], ls='--', marker='s', ms=3, label='vision encoders')],
               frameon=False, loc='lower center', ncol=2, bbox_to_anchor=(.5, 0))
    save(fig, 'fig_gauge')


# ---------------------------------------------------------------- fig 9: realisability (needs realise/*.npz)
def fig_realise():
    files = {m: ROOT / 'realise' / f'{m}.npz' for m in VISION}
    if not all(f.exists() for f in files.values()):
        print('skip fig_realise: runs/modal/realise/*.npz missing'); return
    T = list(np.load(files['resnet18'])['transforms'])
    rad, ang, rej = {}, {}, {}                     # [model] -> arrays over (transform, level)
    for m in VISION:
        F = np.load(files[m])['features'].astype(float)
        c, (W, b) = cache(0, m), head(0, m)
        ref, yref, cal = c['reference_x'].astype(float), c['reference_y'], c['calibration_x'].astype(float)
        knn, maha = SKKNN(5).fit(ref), ShrinkageMahalanobis().fit(ref, yref)
        scores = {'knn': knn.score, 'mahalanobis_shrinkage': maha.score, 'energy_T1': lambda z: energy(z, W, b),
                  'msp_T1': lambda z: -softmax(z @ W.T + b, axis=1).max(1)}
        th = {d: threshold(f(cal), .05) for d, f in scores.items()}
        z0n = np.median(np.linalg.norm(F[0, 0], axis=1))
        rad[m] = np.zeros(F.shape[:2]); ang[m] = np.zeros(F.shape[:2])
        rej[m] = {d: np.zeros(F.shape[:2]) for d in scores}
        for ti in range(F.shape[0]):
            z0 = F[ti, 0]; u0 = z0 / np.linalg.norm(z0, axis=1, keepdims=True)
            for li in range(F.shape[1]):
                dz = F[ti, li] - z0; rc = (dz * u0).sum(1)
                rad[m][ti, li] = np.median(rc) / z0n
                ang[m][ti, li] = np.median(np.linalg.norm(dz - rc[:, None] * u0, axis=1)) / z0n
                for d, f in scores.items():
                    rej[m][d][ti, li] = np.mean(f(F[ti, li]) > th[d])
    x = np.arange(np.load(files['resnet18'])['features'].shape[1])
    fig, axes = plt.subplots(2, len(T), figsize=(6.75, 3.3), sharey='row', sharex=True)

    def band(ax, arr, color, ls, label=None, marker=None):
        arr = np.array(arr)
        ax.fill_between(x, arr.min(0), arr.max(0), color=color, alpha=.15, lw=0)
        ax.plot(x, arr.mean(0), c=color, ls=ls, lw=1.1, marker=marker, ms=2.5, label=label)

    for ti, tname in enumerate(T):
        ax = axes[0, ti]
        band(ax, [rad[m][ti] for m in VISION], '#CC79A7', '-', 'radial (norm change)', 'o')
        band(ax, [ang[m][ti] for m in VISION], '.35', '--', 'angular (rotation)', 's')
        ax.axhline(0, c='.6', lw=.5); ax.set_title(tname); ax.grid(alpha=.25, lw=.4)
        ax = axes[1, ti]
        for d in DETS:
            band(ax, [rej[m][d][ti] for m in VISION], DCOL[d], DLS[d], DETS[d], DMK[d])
        ax.axhline(.05, c='.6', lw=.5, ls=(0, (1, 2))); ax.grid(alpha=.25, lw=.4)
        ax.set_xlabel('strength')
    axes[0, 0].set_ylabel('feature motion\n(rel. to ID norm)')
    axes[1, 0].set_ylabel('fraction rejected')
    h0, l0 = axes[0, 0].get_legend_handles_labels(); h1, l1 = axes[1, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.legend(h0 + h1, l0 + l1, loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(.5, 0))
    save(fig, 'fig_realise')


if __name__ == '__main__':
    only = sys.argv[2:]
    for f in (fig_teaser, fig_failure_map, fig_theory, fig_support, fig_mechanism, fig_curves_seeds,
              fig_real_ood, fig_gauge, fig_realise):
        if not only or f.__name__ in only:
            f()
