"""Paired static evaluation and independent-factor crossed steering experiments."""
from dataclasses import asdict, dataclass
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from .core import (bootstrap_indices, file_hash, load_cache, new_output,
                   threshold, variance_components, versions, write_json, source_hashes)
from .detectors import make_detector
from .paths import trace_path


STRAIGHT_KINDS = ('pca', 'random', 'oracle')
# Polar controls: great circles through the probe on the sphere of radius ||x|| about the
# feature origin (angular only, arc length alpha), shrinkage toward the origin and pure radial
# growth (radial only). With straight rays (radial and angular) these span the polar design.
CURVED_KINDS = ('sphere_pca', 'sphere_random', 'origin', 'radial')
RADIAL_KINDS = ('origin', 'radial')


def path_points(kind, x, u, alphas):
    """Points on the steering path from probe(s) x[..., D] with signed unit direction u[D].

    Straight kinds: x + alpha u. sphere_*: great circle through x whose tangent at x is
    the component of u orthogonal to x, parametrized by arc length, so ||z(alpha)|| = ||x||.
    origin: x - alpha x/||x||; radial: x + alpha x/||x|| (u ignored). Returns [..., G, D].
    """
    x, alphas = np.asarray(x, float), np.asarray(alphas, float)
    if kind in STRAIGHT_KINDS:
        return x[..., None, :] + alphas[:, None] * u
    r = np.linalg.norm(x, axis=-1, keepdims=True)
    xhat = x / r
    if kind in RADIAL_KINDS:
        sign = -1. if kind == 'origin' else 1.
        return x[..., None, :] + sign * alphas[:, None] * xhat[..., None, :]
    w = u - (xhat @ u)[..., None] * xhat
    w /= np.linalg.norm(w, axis=-1, keepdims=True)
    theta = alphas[:, None] / r[..., None, :]
    return r[..., None, :] * (np.cos(theta) * xhat[..., None, :] + np.sin(theta) * w[..., None, :])


def effective_horizon(kind, x, horizon):
    """Cap curved paths: stay short of the antipode (sphere) or the origin (origin)."""
    rmin = float(np.linalg.norm(x, axis=1).min())
    if kind.startswith('sphere_'):
        return min(horizon, .95 * np.pi * rmin)
    if kind == 'origin':
        return min(horizon, .99 * rmin)
    return horizon


@dataclass
class Config:
    detectors: tuple = ('knn', 'mahalanobis', 'energy', 'msp')
    backend: str = 'pytorch'
    seed: int = 7
    tau: float = .05
    k: int = 5
    references: int = 10
    directions: int = 10
    probes: int = 128
    horizon: float = 5.
    steps: int = 101
    refine: int = 15
    direction_kind: str = 'pca'
    calibration_draws: int = 0
    temperatures: tuple = (1.,)

    def validate(self):
        if min(self.references, self.directions, self.probes, self.k) < 1:
            raise ValueError('Counts must be positive')
        if self.steps < 2 or self.horizon <= 0 or self.refine < 0 or not 0 < self.tau < 1:
            raise ValueError('Invalid path/calibration configuration')
        if not self.detectors or len(set(self.detectors)) != len(self.detectors):
            raise ValueError('Specify unique detectors')
        if self.direction_kind not in STRAIGHT_KINDS + CURVED_KINDS:
            raise ValueError('Unknown direction kind')
        if not self.temperatures or any(t <= 0 for t in self.temperatures):
            raise ValueError('Positive temperatures required')


def estimate_direction(x):
    cov = np.atleast_2d(np.cov(x, rowvar=False))
    values, vectors = eigh(cov)
    v = vectors[:, -1]
    if v[np.argmax(abs(v))] < 0:
        v = -v
    return v, dict(top_eigenvalue=float(values[-1]),
                   covariance_eigengap=float(values[-1]-values[-2]) if len(values)>1 else None)


def variants(config):
    for name in config.detectors:
        for temp in (config.temperatures if name in ('energy', 'msp') else (1.,)):
            yield name, temp, (f'{name}_T{temp:g}' if name in ('energy','msp') else name)


def fpr_at_tpr(labels, scores, tpr=.95):
    """ID false-positive rate at the first threshold detecting >= tpr of OOD."""
    f, t, _ = roc_curve(labels, scores)
    return float(f[np.searchsorted(t, tpr, side='left')])


def metrics(det, t, data):
    si, so = det.score(data['id_test_x']), det.score(data['ood_test_x'])
    labels = np.r_[np.zeros(len(si)), np.ones(len(so))]
    scores = np.r_[si, so]
    result = dict(id_rejection=float(np.mean(si > t)), ood_detection=float(np.mean(so > t)),
                  auroc=float(roc_auc_score(labels, scores)),
                  ood_aupr=float(average_precision_score(labels, scores)),
                  fpr_at_95_tpr=fpr_at_tpr(labels, scores),
                  threshold=t, threshold_infinite=bool(np.isinf(t)))
    if 'ood_test_groups' in data:
        result['ood_by_dataset'] = {}
        for group in np.unique(data['ood_test_groups']):
            sg=so[data['ood_test_groups']==group]
            yg=np.r_[np.zeros(len(si)),np.ones(len(sg))]
            result['ood_by_dataset'][str(group)] = dict(n_ood=len(sg),
                ood_detection=float(np.mean(sg>t)),
                auroc=float(roc_auc_score(yg,np.r_[si,sg])),
                ood_aupr=float(average_precision_score(yg,np.r_[si,sg])),
                fpr_at_95_tpr=fpr_at_tpr(yg,np.r_[si,sg]))
    return result, si, so


def evaluate(cache, output, config, head=None):
    config.validate()
    data, out = load_cache(cache), new_output(output)
    start, result = time.perf_counter(), {}
    try:
        for name, temp, label in variants(config):
            det = make_detector(name, config.backend, config.k, head, temp)
            det.fit(data['reference_x'], data['reference_y'])
            cs = det.score(data['calibration_x']); t = threshold(cs, config.tau)
            result[label], si, so = metrics(det, t, data)
            np.savez_compressed(out/f'{label}_scores.npz', calibration=cs, id_test=si, ood_test=so,
                                id_ids=data['id_test_ids'], ood_ids=data['ood_test_ids'],
                                ood_groups=data.get('ood_test_groups',np.full(len(so),'ood')))
        if head is not None:
            import torch
            with torch.inference_mode():
                logits = head(torch.tensor(data['id_test_x'], dtype=torch.float32))
            result['id_classification_accuracy'] = float(np.mean(logits.argmax(1).numpy()==data['id_test_y']))
            if data['metadata'].get('modality')=='text':
                result['intent_accuracy'] = result['id_classification_accuracy']
        write_json(out/'metrics.json', result)
        write_json(out/'manifest.json', dict(config=asdict(config), cache_sha256=file_hash(cache),
            cache_metadata=data['metadata'], head_sha256=getattr(head,'_steering_state_hash',None), versions=versions(), source_hashes=source_hashes(), seconds=time.perf_counter()-start,
            positive_class='OOD', target='static held-out evaluation', status='completed'))
    except Exception as exc:
        write_json(out/'failure.json', dict(error=repr(exc), completed=result, status='failed'))
        raise
    return result


def crossed(cache, output, config, head=None):
    """Independent stratified bootstraps from disjoint D and V pools.

    Conditional empirical-data variability; not independent population samples.
    Both direction signs are evaluated; decomposition is performed per sign.
    """
    config.validate()
    if config.direction_kind == 'oracle':
        raise ValueError('Oracle direction is only defined for the synthetic world')
    data = load_cache(cache)
    seeds = np.random.SeedSequence(config.seed).spawn(3)
    rr, rv, rp = [np.random.default_rng(s) for s in seeds]
    refs = [bootstrap_indices(data['reference_y'], rr) for _ in range(config.references)]
    dirs = [bootstrap_indices(data['direction_y'], rv) for _ in range(config.directions)]
    n = min(config.probes, len(data['probe_x']))
    pi = rp.choice(len(data['probe_x']), n, replace=False)
    vectors, diagnostics = [], []
    for idx in dirs:
        if config.direction_kind in ('pca', 'sphere_pca', 'origin', 'radial'):
            v, diag = estimate_direction(data['direction_x'][idx])
        else:
            v = rv.normal(size=data['probe_x'].shape[1]); v /= np.linalg.norm(v)
            diag = {'kind': 'random'}
        diag['path'] = config.direction_kind
        vectors.append(v); diagnostics.append(diag)
    return run_crossed(data, output, config,
        [(data['reference_x'][idx], data['reference_y'][idx]) for idx in refs],
        np.array(vectors), pi, head,
        metadata=dict(cache_sha256=file_hash(cache), cache_metadata=data['metadata'],
                      sampling='independent stratified bootstrap of disjoint fixed pools',
                      direction_diagnostics=diagnostics),
        indices=dict(reference_indices=np.array(refs), direction_indices=np.array(dirs)))


def synthetic_crossed(output, config, n_reference=64, n_direction=64, dim=6):
    """E2 Gaussian population experiment with independent fresh draws, no OOD fit."""
    config.validate()
    if config.direction_kind in CURVED_KINDS:
        raise ValueError('Curved path kinds are only implemented for cache-based crossed runs')
    if dim < 2 or min(n_reference, n_direction) < 2:
        raise ValueError('Gaussian PCA experiment needs dimension and sample sizes >=2')
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(config.seed).spawn(7)]
    scale = np.ones(dim); scale[0] = 2.
    draw = lambda r,n: r.normal(size=(n, dim))*scale
    refs = [(draw(rngs[0], n_reference), np.zeros(n_reference, dtype=int)) for _ in range(config.references)]
    vs, diagnostics = [], []
    oracle = np.eye(dim)[0]
    for _ in range(config.directions):
        if config.direction_kind == 'oracle':
            v, diag = oracle, {'kind':'oracle'}
        elif config.direction_kind == 'random':
            v = rngs[1].normal(size=dim); v /= np.linalg.norm(v); diag={'kind':'random'}
        else:
            v, diag = estimate_direction(draw(rngs[1], n_direction))
            diag['absolute_oracle_cosine'] = float(abs(v@oracle))
        vs.append(v); diagnostics.append(diag)
    data = dict(calibration_x=draw(rngs[2], 399), probe_x=draw(rngs[3],config.probes),
                id_test_x=draw(rngs[4],1000), ood_test_x=draw(rngs[5],1000)+3*oracle)
    data['probe_ids'] = np.array([f'gaussian:probe:{i}' for i in range(config.probes)])
    return run_crossed(data, output, config, refs, np.array(vs), np.arange(config.probes),
        metadata=dict(sampling='independent fresh Gaussian draws', covariance_diagonal=scale**2,
                      n_reference=n_reference,n_direction=n_direction,
                      direction_diagnostics=diagnostics, fixture=True))


def trace(det, t, kind, z, u, alphas, refine, grid_scores=None):
    """Straight kinds use the original tracer unchanged; curved kinds trace the 1-D arc parameter."""
    if kind in STRAIGHT_KINDS:
        return trace_path(det.score, t, z, u, alphas, refine_steps=refine, grid_scores=grid_scores)
    arc = lambda a: det.score(path_points(kind, z, u, a[:, 0]))
    return trace_path(arc, t, np.zeros(1), np.ones(1), alphas, refine_steps=refine, grid_scores=grid_scores)


def run_crossed(data, output, config, references, vectors, probe_indices,
                head=None, metadata=None, indices=None):
    out, start = new_output(output), time.perf_counter()
    x = data['probe_x'][probe_indices]
    kind = config.direction_kind
    horizon = effective_horizon(kind, x, config.horizon)
    alphas = np.linspace(0, horizon, config.steps)
    # Radial paths ignore the direction, so only one sign is meaningful.
    signs = (1.,) if kind in RADIAL_KINDS else (1., -1.)
    np.savez_compressed(out/'design.npz', vectors=vectors, signs=np.array(signs),
                        alphas=alphas, probe_indices=probe_indices, probe_ids=data['probe_ids'][probe_indices],
                        **(indices or {}))
    manifest = dict(config=asdict(config), versions=versions(), source_hashes=source_hashes(), metadata=metadata,
                    head_sha256=getattr(head,'_steering_state_hash',None),
                    score_orientation='higher is anomalous', calibration='fixed points; recalibrated for each D',
                    path=path_points.__doc__.strip().splitlines()[0] + f' kind={kind}',
                    direction_kind=kind, requested_horizon=config.horizon, effective_horizon=horizon,
                    decomposition='empirical balanced ANOVA per probe and sign, then averaged; no confidence intervals',
                    status='running')
    write_json(out/'manifest.json', manifest)
    completed = {}
    try:
        for name, temp, label in variants(config):
            # B,C,S,P,G arrays; the cap is always accompanied by event/status.
            shape = (len(references), len(vectors), len(signs), len(x))
            scores = np.empty(shape+(len(alphas),), dtype='float64')
            observed = np.empty(shape)
            events = np.zeros(shape, dtype=bool)
            initial = np.zeros(shape, dtype=bool)
            recross = np.zeros(shape, dtype=int)
            status = np.empty(shape, dtype='<U32')
            brackets = np.full(shape+(2,), np.nan)
            thresholds, static = [], []
            with open(out/f'{label}_paths.jsonl','w') as records:
                for b,(d,y) in enumerate(references):
                    det = make_detector(name, config.backend, config.k, head, temp).fit(d,y)
                    t = threshold(det.score(data['calibration_x']), config.tau)
                    thresholds.append(t)
                    static.append(metrics(det,t,data)[0])
                    for c,v in enumerate(vectors):
                        for s,sign in enumerate(signs):
                            points = path_points(kind, x, sign*v, alphas)
                            grid_scores = det.score(points.reshape(-1,x.shape[1])).reshape(len(x),len(alphas))
                            for p,z in enumerate(x):
                                tr = trace(det, t, kind, z, sign*v, alphas, config.refine, grid_scores[p])
                                ix=(b,c,s,p)
                                scores[ix]=tr.scores
                                observed[ix]=horizon if tr.censored else tr.alpha_star
                                events[ix]=not tr.censored
                                initial[ix]=tr.status=='already_rejected'
                                recross[ix]=tr.n_recross
                                status[ix]=tr.status
                                if tr.first_bracket is not None:
                                    brackets[ix]=tr.first_bracket
                                records.write(json.dumps(dict(reference=b,direction=c,sign=sign,
                                    probe_id=str(data['probe_ids'][probe_indices[p]]),status=tr.status,
                                    alpha_star=tr.alpha_star,censored=tr.censored,
                                    first_bracket=tr.first_bracket,return_brackets=tr.return_brackets))+'\n')
            # Compare full rejection with cumulative first rejection, especially with returns.
            rejected = scores > np.array(thresholds)[:,None,None,None,None]
            np.savez_compressed(out/f'{label}_traces.npz', scores=scores, rejected=rejected,
                observed=observed,event=events,initially_rejected=initial,recrossings=recross,
                status=status,first_brackets=brackets,thresholds=thresholds,
                power=rejected.mean(axis=3), first_rejection_cdf=np.maximum.accumulate(rejected,axis=-1).mean(axis=3))
            per_sign = {str(int(sign)):variance_components(observed[:,:,s,:]) for s,sign in enumerate(signs)}
            completed[label] = dict(variance=per_sign,crossing_probability=float(events.mean()),
                initial_rejection_probability=float(initial.mean()),restricted_crossing_mean=float(observed.mean()),
                return_or_recross_probability=float(np.mean(recross>0)), static=static,
                reference_dependence=name not in ('energy','msp'))
            if config.calibration_draws:
                # Isolate calibration resampling: reference 0 and direction 0 fixed.
                det = make_detector(name,config.backend,config.k,head,temp).fit(*references[0])
                c_scores=det.score(data['calibration_x'])
                rng=np.random.default_rng(config.seed+991)
                ca, ce=[],[]
                for _ in range(config.calibration_draws):
                    t=threshold(rng.choice(c_scores,len(c_scores),replace=True),config.tau)
                    traces=[trace(det,t,kind,z,vectors[0],alphas,config.refine) for z in x]
                    ca.append([horizon if tr.censored else tr.alpha_star for tr in traces])
                    ce.append([not tr.censored for tr in traces])
                np.savez_compressed(out/f'{label}_calibration_only.npz',observed=ca,event=ce)
            write_json(out/'summary.json',completed)
        manifest.update(status='completed',seconds=time.perf_counter()-start)
        write_json(out/'manifest.json',manifest)
    except Exception as exc:
        manifest.update(status='failed',error=repr(exc),seconds=time.perf_counter()-start)
        write_json(out/'manifest.json',manifest)
        raise
    return completed
