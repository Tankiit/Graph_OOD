"""E0: analytical controls and numerical invariance, not a research result."""
import numpy as np
from .analytic import alpha_star_maha
from .core import threshold, write_json, versions
from .detectors import Gaussian, make_detector
from .paths import trace_path


def run_e0(output=None, include_pytorch=False):
    rng = np.random.default_rng(101)
    d,cal,probe = rng.normal(size=(256,3)),rng.normal(size=(999,3)),rng.normal(size=(30,3))
    det=Gaussian().fit(d); nll=Gaussian(nll=True).fit(d)
    t=threshold(det.score(cal)); tn=threshold(nll.score(cal))
    grid=np.linspace(0,12,241)
    errors=[]; invariance=True; grid_difference=0.
    transforms=(lambda s:s,lambda s:3*s+11,lambda s:np.log1p(s))
    for z in probe:
        v=rng.normal(size=3); v/=np.linalg.norm(v)
        exact=alpha_star_maha(z,v,det.mu,det.Sigma,t)
        trace=trace_path(det.score,t,z,v,grid)
        errors.append(abs(trace.alpha_star-exact))
        finer=trace_path(det.score,t,z,v,np.linspace(0,12,481))
        grid_difference=max(grid_difference,abs(trace.alpha_star-finer.alpha_star))
        for g in transforms:
            transformed=trace_path(lambda x:g(det.score(x)),threshold(g(det.score(cal))),z,v,grid)
            invariance &= np.array_equal(trace.rejected,transformed.rejected)
            invariance &= abs(trace.alpha_star-transformed.alpha_star)<1e-7
        nt=trace_path(nll.score,tn,z,v,grid)
        invariance &= np.array_equal(trace.rejected,nt.rejected)
    # An independent ID check is reported, not equated with exact 5% conditional coverage.
    false_alarm=float(np.mean(det.score(rng.normal(size=(10000,3)))>t))
    result=dict(analytic_max_error=max(errors),doubled_grid_max_difference=grid_difference,
                monotone_and_gaussian_equivalence=bool(invariance),independent_id_rejection=false_alarm,
                insufficient_calibration_is_infinite=bool(np.isinf(threshold(np.arange(5.)))),
                versions=versions())
    if include_pytorch:
        checks={}
        x=np.array([[-1.,0.],[-.8,.1],[1.,0.],[.8,.1]],dtype='float32'); y=np.array([0,0,1,1])
        for name in ('knn','mahalanobis'):
            model=make_detector(name,k=1).fit(x,y)
            checks[name]=bool(model.score([[20.,20.]])[0]>model.score([[-1.,0.]])[0])
        result['pytorch_score_orientation']=checks
    result['passed']=bool(max(errors)<1e-7 and invariance and grid_difference<1e-7
                          and result['insufficient_calibration_is_infinite']
                          and all(result.get('pytorch_score_orientation',{'core':True}).values()))
    if output:
        write_json(output,result)
    if not result['passed']:
        raise AssertionError(result)
    return result
