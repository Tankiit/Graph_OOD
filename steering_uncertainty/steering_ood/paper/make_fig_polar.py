"""Figure 1: schematic geometry of detector rejection regions (2-D cartoon).

Run from paper/: python make_fig_polar.py  ->  figures/fig_polar.pdf (+ .png preview).
Schematic only: the kNN threshold is set by hand to t > rho, as happens in high dimension.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import logsumexp

rng = np.random.default_rng(0)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "mathtext.fontset": "stix"})

# ID features on a shell of radius 1
n = 240
th = rng.uniform(0, 2*np.pi, n)
ref = np.c_[np.cos(th), np.sin(th)] * (1 + 0.05*rng.standard_normal(n))[:, None]
thc = rng.uniform(0, 2*np.pi, 400)
cal = np.c_[np.cos(thc), np.sin(thc)] * (1 + 0.05*rng.standard_normal(400))[:, None]

L = 2.5
g = np.linspace(-L, L, 501); X, Y = np.meshgrid(g, g); G = np.c_[X.ravel(), Y.ravel()]

# distance detector: kNN; threshold set as in high dimension (t > rho), schematic
k = 5
def knn(P):
    d = np.linalg.norm(P[:, None, :] - ref[None], axis=2)
    return np.sort(d, axis=1)[:, k-1]
t_knn = 1.15

# linear head, 3 classes, rows summing to zero (origin in their hull)
ang = np.deg2rad([90, 210, 330]); W = 3.5*np.c_[np.cos(ang), np.sin(ang)]; b = np.zeros(3)
def energy(P, Wm): return -logsumexp(P @ Wm.T + b, axis=1)
t_E = np.quantile(energy(cal, W), 0.95)

# gauge: shift every row by r = -lam * v (softmax and predictions unchanged)
u_dir = np.array([np.cos(np.deg2rad(75)), np.sin(np.deg2rad(75))])
v = np.array([np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))])
lam = 2.6*np.max(W @ v); Wr = W - lam*v[None]
t_Er = np.quantile(energy(cal, Wr), 0.95)

# probe and paths; Okabe-Ito colors and line styles match the R(alpha) figure
x0 = u_dir.copy()
s_out = np.linspace(1, 2.45, 200)[:, None]*x0[None]            # radial growth
s_in = np.linspace(1, 0.03, 200)[:, None]*x0[None]             # toward origin
phi = np.deg2rad(np.linspace(75, 170, 200)); geo = np.c_[np.cos(phi), np.sin(phi)]
straight = x0[None] + np.linspace(0, 2.3, 200)[:, None]*v[None]
paths = [(s_out, "#CC79A7", "-.", "radial growth"), (s_in, "#009E73", ":", "toward origin"),
         (geo, "#0072B2", "--", "geodesic"), (straight, "0.25", "-", "straight")]

def first_reject(P, score, t):
    r = score(P) > t
    return P[np.argmax(r)] if r.any() and not r[0] else None

panels = [("(a) distance score (kNN)", lambda P: knn(P), t_knn),
          ("(b) energy", lambda P: energy(P, W), t_E),
          ("(c) energy, shifted head", lambda P: energy(P, Wr), t_Er)]

fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.6))
for ax, (title, score, t) in zip(axes, panels):
    Z = (score(G) > t).reshape(X.shape)
    ax.contourf(X, Y, Z, levels=[0.5, 1.5], colors=["#f5d0c5"])
    ax.contour(X, Y, Z, levels=[0.5], colors=["#b03a2e"], linewidths=0.6)
    if "energy" in title:   # argmax boundaries: identical in (b) and (c)
        lab = np.argmax(G @ W.T, axis=1).reshape(X.shape)
        ax.contour(X, Y, lab, levels=[0.5, 1.5], colors=["0.6"], linewidths=0.4, linestyles=":")
    ax.scatter(ref[:, 0], ref[:, 1], s=1.2, c="0.2", lw=0)
    for P, c, ls, name in paths:
        ax.plot(P[:, 0], P[:, 1], c=c, ls=ls, lw=1.1, label=name)
        ax.annotate("", xy=P[-1], xytext=P[-8], arrowprops=dict(arrowstyle="-|>", color=c, lw=1.1))
        fr = first_reject(P, score, t)
        if fr is not None: ax.plot(*fr, marker="x", c="k", ms=5, mew=1.2)
    ax.plot(*x0, "o", c="k", ms=3); ax.plot(0, 0, "+", c="k", ms=5)
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=8)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, frameon=False, handlelength=2.2)
fig.tight_layout(pad=0.3, rect=(0, 0.07, 1, 0.93))
fig.savefig("figures/fig_polar.pdf"); fig.savefig("figures/fig_polar.png", dpi=200)
print("t_knn", t_knn, "t_E", round(t_E, 3), "E(0)", round(-np.log(3), 3), "t_Er", round(t_Er, 3))
