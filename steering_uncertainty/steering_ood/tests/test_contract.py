import numpy as np
import pytest
from steering_nlp.audit import run_e0
from steering_nlp.core import threshold,variance_components,load_cache
from steering_nlp.paths import trace_path
from steering_nlp.data import synthetic_cache
from steering_nlp.detectors import make_detector
from steering_nlp.experiment import Config, crossed, synthetic_crossed


def test_e0():
    assert run_e0()['passed']


def test_censoring_initial_and_reentry():
    g=np.linspace(0,6,121)
    score=lambda z: np.minimum(z[:,0]**2,(z[:,0]-5)**2)
    tr=trace_path(score,1.,np.array([0.]),np.array([1.]),g)
    assert tr.alpha_star==pytest.approx(1.,abs=1e-6)
    assert tr.return_brackets and not tr.rejected[-1]
    tr=trace_path(score,1.,np.array([2.]),np.array([1.]),g)
    assert tr.alpha_star==0 and tr.status=='already_rejected'
    tr=trace_path(score,float('inf'),np.array([0.]),np.array([1.]),g)
    assert tr.censored and tr.alpha_star is None and tr.status=='uninformative_threshold'


def test_order_statistic_and_ties():
    assert threshold(np.arange(19.),.05)==18
    assert np.isinf(threshold(np.arange(18.),.05))
    assert not np.any(np.ones(20)>threshold(np.ones(20)))


def test_variance_sources():
    ref=np.array([-1.,0,1])[:,None,None]
    direction=np.array([-2.,0,2])[None,:,None]
    a=ref+direction+ref*direction
    v=variance_components(a)
    assert v['reference']==pytest.approx(2/3)
    assert v['direction']==pytest.approx(8/3)
    assert v['interaction']==pytest.approx(16/9)
    assert sum(v[k] for k in ('reference','direction','interaction'))==pytest.approx(v['total'])


def test_cache_rejects_leakage(tmp_path):
    p=tmp_path/'cache.npz'; synthetic_cache(p)
    with np.load(p,allow_pickle=False) as f: data=dict(f)
    data['reference_ids'][0]=data['head_ids'][0]
    np.savez(p,**data)
    with pytest.raises(ValueError,match='overlap'): load_cache(p)


def test_crossed_reproducibility_and_pairing(tmp_path):
    p=tmp_path/'cache.npz'; synthetic_cache(p)
    cfg=Config(detectors=('gaussian','gaussian_nll','knn'),backend='sklearn',
        references=2,directions=2,probes=3,steps=31,refine=8)
    crossed(p,tmp_path/'a',cfg); crossed(p,tmp_path/'b',cfg)
    with (np.load(tmp_path/'a/gaussian_traces.npz') as a, np.load(tmp_path/'a/gaussian_nll_traces.npz') as n,
          np.load(tmp_path/'b/gaussian_traces.npz') as b):
        assert np.array_equal(a['rejected'],n['rejected'])
        assert np.array_equal(a['observed'],b['observed'])
        assert a['scores'].shape==(2,2,2,3,31)
        assert np.all(a['first_rejection_cdf']>=a['power'])


def test_synthetic_oracle_has_no_direction_variance(tmp_path):
    cfg=Config(detectors=('gaussian',),backend='sklearn',references=3,directions=3,
               probes=4,steps=31,refine=8,direction_kind='oracle')
    result=synthetic_crossed(tmp_path/'run',cfg)
    for v in result['gaussian']['variance'].values():
        assert v['direction']<1e-20
        assert v['interaction']<1e-20


def test_library_detectors_and_skorch(tmp_path):
    pytest.importorskip('pytorch_ood'); pytest.importorskip('skorch')
    from steering_nlp.head import train_head,load_head
    import torch
    p=tmp_path/'cache.npz'; synthetic_cache(p)
    data=load_cache(p)
    train_head(p,tmp_path/'head',epochs=3)
    head=load_head(tmp_path/'head',p)
    z=data['probe_x']
    for name in ('knn','mahalanobis','energy','msp'):
        det=make_detector(name,k=3,head=head).fit(data['reference_x'],data['reference_y'])
        assert det.score(z).shape==(len(z),)
    energy=make_detector('energy',head=head,temperature=2.)
    with torch.inference_mode():
        expected=(-2*torch.logsumexp(head(torch.tensor(z))/2,dim=1)).numpy()
    assert np.allclose(energy.score(z),expected)
    knn=make_detector('knn',k=3).fit(data['reference_x'],data['reference_y'])
    exact=make_detector('knn',backend='sklearn',k=3).fit(data['reference_x'])
    assert np.allclose(knn.score(z),exact.score(z),rtol=1e-5)
    cfg=Config(detectors=('energy','msp'),references=2,directions=2,probes=2,steps=15,refine=3)
    result=crossed(p,tmp_path/'crossed',cfg,head)
    for method in result.values():
        for v in method['variance'].values():
            assert v['reference']<1e-20
            assert v['interaction']<1e-20


def test_spectral_matches_legacy_definition():
    from steering_nlp.detectors import Spectral
    from scipy.linalg import eigvalsh
    from scipy.spatial.distance import cdist
    rng=np.random.default_rng(4); x=rng.normal(size=(8,3)); z=rng.normal(size=(2,3))
    m=Spectral().fit(x)
    def eig(a):
        w=np.exp(-cdist(a,a,'sqeuclidean')/(2*m.bandwidth**2)); np.fill_diagonal(w,0)
        return eigvalsh(np.diag(w.sum(1))-w)[1]
    expected=[abs(eig(np.vstack((x,p)))-eig(x)) for p in z]
    assert np.allclose(m.score(z),expected)


def test_sentence_transformer_encode_stage(tmp_path):
    pytest.importorskip('sentence_transformers')
    import json
    from sentence_transformers import SentenceTransformer, models
    from steering_nlp.data import encode
    from steering_nlp.core import ROLES
    # Local, deterministic tiny encoder validates actual ST serialization/encoding.
    # It is not a pretrained NLP benchmark and requires no model download.
    local=tmp_path/'encoder'
    SentenceTransformer(modules=[models.BoW(vocab=['hello','world','bank','card'])]).save(str(local))
    rows=[dict(id=f'{role}:{i}',role=role,text='hello bank' if i==0 else 'world card',
               label=-1 if role=='ood_test' else i) for role in ROLES for i in range(2)]
    split=tmp_path/'split.json'; split.write_text(json.dumps(dict(rows=rows,metadata={})))
    encode(split,tmp_path/'encoded.npz',model=str(local),batch_size=2)
    data=load_cache(tmp_path/'encoded.npz')
    assert data['head_x'].shape==(2,4)
    assert np.allclose(data['head_x'],data['probe_x'])


def test_projection_ignores_test_data(tmp_path):
    from steering_nlp.data import project_cache
    a=tmp_path/'a.npz'; synthetic_cache(a)
    with np.load(a,allow_pickle=False) as f: arrays=dict(f)
    arrays['ood_test_x']*=1000
    np.savez(tmp_path/'b.npz',**arrays)
    project_cache(a,tmp_path/'pa.npz',3)
    project_cache(tmp_path/'b.npz',tmp_path/'pb.npz',3)
    pa,pb=load_cache(tmp_path/'pa.npz'),load_cache(tmp_path/'pb.npz')
    assert np.array_equal(pa['projection_components'],pb['projection_components'])
    assert np.array_equal(pa['calibration_x'],pb['calibration_x'])


def test_curved_paths_geometry():
    from steering_ood.experiment import path_points
    rng=np.random.default_rng(0)
    x=rng.normal(size=(5,7)); u=rng.normal(size=7); u/=np.linalg.norm(u)
    a=np.linspace(0,2,21)
    sph=path_points('sphere_pca',x,u,a)
    assert np.allclose(np.linalg.norm(sph,axis=-1),np.linalg.norm(x,axis=1)[:,None])
    assert np.allclose(sph[:,0],x)
    # arc length: chord between consecutive points matches 2 r sin(d/(2r))
    r=np.linalg.norm(x,axis=1)[:,None]; d=a[1]-a[0]
    assert np.allclose(np.linalg.norm(np.diff(sph,axis=1),axis=-1),2*r*np.sin(d/(2*r)))
    ori=path_points('origin',x,u,a[:5]*.1)
    assert np.allclose(np.linalg.norm(ori,axis=-1),np.linalg.norm(x,axis=1)[:,None]-a[:5]*.1)
    assert np.allclose(path_points('pca',x,u,a),x[:,None]+a[:,None]*u)
    rad=path_points('radial',x,u,a)
    assert np.allclose(np.linalg.norm(rad,axis=-1),np.linalg.norm(x,axis=1)[:,None]+a)
    assert np.allclose(rad/np.linalg.norm(rad,axis=-1,keepdims=True),(x/np.linalg.norm(x,axis=1,keepdims=True))[:,None])


def test_curved_crossed_runs_and_crossing_is_on_arc(tmp_path):
    from steering_ood.experiment import path_points
    p=tmp_path/'cache.npz'; synthetic_cache(p)
    for kind in ('sphere_random','origin','radial'):
        cfg=Config(detectors=('gaussian',),backend='sklearn',references=2,directions=2,
                   probes=3,steps=31,refine=20,horizon=4.,direction_kind=kind)
        crossed(p,tmp_path/kind,cfg)
        design=np.load(tmp_path/kind/'design.npz'); tr=np.load(tmp_path/kind/'gaussian_traces.npz')
        data=load_cache(p); x=data['probe_x'][design['probe_indices']]
        assert len(design['signs'])==(1 if kind in ('origin','radial') else 2)
        assert design['alphas'][-1]<=4.
        det=make_detector('gaussian','sklearn').fit(*[data['reference_x'][design['reference_indices'][0]],None])
        pts=path_points(kind,x,design['signs'][0]*design['vectors'][0],design['alphas'])
        assert np.allclose(det.score(pts.reshape(-1,x.shape[1])).reshape(len(x),-1),tr['scores'][0,0,0])


def test_shrinkage_mahalanobis_whitening_matches_direct():
    from sklearn.covariance import LedoitWolf
    rng=np.random.default_rng(1)
    y=np.repeat(np.arange(4),30); x=rng.normal(size=(120,12))+y[:,None]
    det=make_detector('mahalanobis_shrinkage').fit(x,y)
    q=rng.normal(size=(50,12))*3
    direct=np.min([np.einsum('ni,ij,nj->n',q-m,det.precision,q-m) for m in det.means],axis=0)
    assert np.allclose(det.score(q),direct,rtol=1e-9,atol=1e-9)
