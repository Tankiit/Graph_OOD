"""Rejection curves R(alpha) per detector and path type, plus matched-horizon tables.

R(alpha): fraction of steered ID probes rejected at arc length/distance alpha (full dimension),
averaged over reference fits, directions and signs, then over the encoders of a modality.
alpha is in ID radii (median ||reference - mean||) so encoders are comparable.

Sources: straight kNN/energy/MSP from runs/modal/main; Mahalanobis on straight rays and all
curved paths from runs/modal/v2 (float64 Ledoit-Wolf class-conditional Mahalanobis).
Usage: python scripts/path_curves.py [runs/modal]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else 'runs/modal')
MODALITIES = {'text': ['mpnet', 'minilm', 'bge_base', 'bge_large'],
              'vision': ['resnet18', 'resnet50', 'vit_b16', 'dinov2_s']}
DETECTORS = {'knn': 'kNN', 'mahalanobis_shrinkage': 'Mahalanobis', 'energy_T1': 'Energy', 'msp_T1': 'MSP'}
PATHS = {'pca': 'straight, PC', 'random': 'straight, random', 'sphere_pca': 'geodesic, PC',
         'sphere_random': 'geodesic, random', 'origin': 'toward origin', 'radial': 'radial growth'}
GRID = np.linspace(0, 3, 301)


def source(model, det, path):
    main = det != 'mahalanobis_shrinkage' and path in ('pca', 'random')
    return ROOT / ('main' if main else 'v2') / model / 'full' / f'crossed_{path}'


def curves(model, det, path):
    """(R(alpha), ever-rejected-by-alpha) on GRID in radii; NaN beyond the path's horizon."""
    run = source(model, det, path)
    if not (run / f'{det}_traces.npz').exists():
        return None
    radius = json.loads((run / 'steering_checks.json').read_text())['id_reference_radius']
    alphas = np.load(run / 'design.npz')['alphas'] / radius
    tr = np.load(run / f'{det}_traces.npz')
    out = []
    for key in ('power', 'first_rejection_cdf'):
        y = tr[key].reshape(-1, len(alphas)).mean(0)
        out.append(np.where(GRID <= alphas[-1] + 1e-9, np.interp(GRID, alphas, y), np.nan))
    return out


def collect():
    data = {}
    for mod, models in MODALITIES.items():
        for det in DETECTORS:
            for path in PATHS:
                rows = [curves(m, det, path) for m in models]
                rows = [r for r in rows if r is not None]
                if rows:
                    data[mod, det, path] = (np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), len(rows))
    return data


def reach(data, mod, paths):
    """Largest alpha (radii) reached by every encoder on every listed path in a modality."""
    return min(GRID[np.isfinite(v[0]).all(0)].max() for (m, _, p), v in data.items() if m == mod and p in paths)


def tables(data):
    lines = []
    for mod in MODALITIES:
        a_match = reach(data, mod, PATHS)
        a_long = reach(data, mod, ('pca', 'random', 'sphere_pca', 'sphere_random', 'radial'))
        for title, alpha, paths in ((f'matched horizon alpha = {a_match:.2f} radii (all paths)', a_match, PATHS),
                                    (f'long horizon alpha = {a_long:.2f} radii (straight, geodesic, radial growth)', a_long,
                                     ('pca', 'random', 'sphere_pca', 'sphere_random', 'radial'))):
            i = int(np.searchsorted(GRID, alpha - 1e-9))
            lines += [f'\n### {mod}: {title}', '',
                      'R = rejected at alpha / ever rejected by alpha; mean over encoders (n shown)', '',
                      '| path | ' + ' | '.join(DETECTORS.values()) + ' |', '|---' * (len(DETECTORS) + 1) + '|']
            for path in paths:
                cells = []
                for det in DETECTORS:
                    v = data.get((mod, det, path))
                    cells.append('—' if v is None else f'{np.nanmean(v[0][:, i]):.2f} / {np.nanmean(v[1][:, i]):.2f} (n={v[2]})')
                lines.append(f'| {PATHS[path]} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def figure(data, output):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6})
    # Okabe-Ito colorblind-safe colors; line style encodes path geometry, marker encodes direction.
    style = {'pca': ('#0072B2', '-', 'o'), 'random': ('#E69F00', '-', 's'),
             'sphere_pca': ('#0072B2', '--', 'o'), 'sphere_random': ('#E69F00', '--', 's'),
             'origin': ('#009E73', ':', '^'), 'radial': ('#CC79A7', '-.', 'v')}
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.3), sharex=True, sharey=True)
    for r, mod in enumerate(MODALITIES):
        a_match = reach(data, mod, PATHS)
        for c, (det, name) in enumerate(DETECTORS.items()):
            ax = axes[r, c]
            for path, label in PATHS.items():
                v = data.get((mod, det, path))
                if v is None:
                    continue
                color, ls, marker = style[path]
                ok = np.isfinite(v[0]).all(0)
                mean = v[0].mean(0)
                ax.fill_between(GRID[ok], v[0].min(0)[ok], v[0].max(0)[ok], color=color, alpha=.12, lw=0)
                ax.plot(GRID[ok], mean[ok], color=color, ls=ls, lw=1.1, marker=marker, markersize=2.5,
                        markevery=40, label=label)
            ax.axhline(.05, color='0.5', lw=.5, ls=(0, (1, 2)))
            ax.axvline(a_match, color='0.6', lw=.5)
            ax.grid(alpha=.25, lw=.4)
            ax.set_ylim(-.02, 1.02); ax.set_xlim(0, 3)
            if r == 0:
                ax.set_title(name)
            if c == 0:
                ax.set_ylabel(f'{mod}\nfraction rejected')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.supxlabel(r'steering distance $\alpha$ (ID radii)', fontsize=8, y=.075)
    fig.legend(handles, labels, loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(.5, -.01))
    fig.tight_layout(rect=(0, .08, 1, 1), h_pad=.6, w_pad=.4)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight')
    return output


def main():
    data = collect()
    out = ROOT / 'figures'
    text = tables(data)
    (out / 'matched_horizon.md').parent.mkdir(parents=True, exist_ok=True)
    (out / 'matched_horizon.md').write_text(text)
    print(text)
    print('saved', figure(data, out / 'path_curves.pdf'))


if __name__ == '__main__':
    main()
