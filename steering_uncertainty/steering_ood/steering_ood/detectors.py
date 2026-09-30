"""Adapters retaining the legacy fit(D)/score(X) contract; labels are optional."""
import inspect
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from .core import matrix


class Gaussian:
    def __init__(self, nll=False, ridge=1e-6):
        self.nll, self.ridge = nll, ridge

    def fit(self, x, y=None):
        x = matrix(x)
        if len(x) < 2:
            raise ValueError("Gaussian needs at least two references")
        self.mu = x.mean(0)
        self.Sigma = np.atleast_2d(np.cov(x, rowvar=False)) + self.ridge*np.eye(x.shape[1])
        self.precision = np.linalg.inv(self.Sigma)
        self.constant = .5*(x.shape[1]*np.log(2*np.pi)+np.linalg.slogdet(self.Sigma)[1])
        return self

    def score(self, x):
        d = matrix(x)-self.mu
        s = np.einsum('ni,ij,nj->n', d, self.precision, d)
        return .5*s+self.constant if self.nll else s


class SKKNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, x, y=None):
        if len(x) < self.k:
            raise ValueError("k exceeds reference size")
        self.model = NearestNeighbors(n_neighbors=self.k, metric="euclidean").fit(matrix(x))
        return self

    def score(self, x):
        return self.model.kneighbors(matrix(x))[0][:, -1]


class ShrinkageMahalanobis:
    """Explicit custom baseline: class means, pooled Ledoit-Wolf residual covariance."""
    def fit(self, x, y):
        x = matrix(x)
        classes = np.unique(y)
        self.means = np.array([x[y == c].mean(0) for c in classes])
        centered = np.concatenate([x[y == c]-m for c, m in zip(classes, self.means)])
        lw = LedoitWolf(assume_centered=True).fit(centered)
        self.precision, self.shrinkage = lw.precision_, lw.shrinkage_
        # float64 whitening: precision = L L^T, so d_c(x)^2 = ||x L - m_c L||^2.
        values, vectors = np.linalg.eigh(lw.covariance_)
        self.whiten = vectors / np.sqrt(values)
        self.white_means = self.means @ self.whiten
        return self

    def score(self, x):
        z = matrix(x) @ self.whiten
        d2 = (z**2).sum(1)[:, None] - 2*z @ self.white_means.T + (self.white_means**2).sum(1)[None]
        return np.maximum(d2.min(1), 0.)


class TorchOOD:
    """CPU feature adapter; no implicit normalization or input-gradient perturbation.

    Constructor signature detection supports model/encoder renaming. An old kNN
    version lacking configurable k fails explicitly instead of silently using 1NN.
    """
    def __init__(self, name, k=5, head=None, temperature=1., batch_size=256):
        import torch
        from pytorch_ood.detector import KNN, Mahalanobis, EnergyBased, MaxSoftmax
        self.torch, self.head, self.name = torch, head, name
        self.batch_size, self.temperature = batch_size, temperature
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        if name == "knn":
            if "k" not in inspect.signature(KNN).parameters:
                if k != 1:
                    raise RuntimeError("This pytorch-ood KNN only supports 1NN; install >=0.3.3 or explicitly use k=1")
                self.model = KNN(None)
            else:
                self.model = KNN(None, k=k)
        elif name == "mahalanobis":
            kw = {"eps": 0.0} if "eps" in inspect.signature(Mahalanobis).parameters else {}
            self.model = Mahalanobis(None, **kw)
        else:
            if head is None:
                raise ValueError(f"{name} needs a trained head")
            self.head = head.cpu().eval()
            self.model = EnergyBased(None, t=temperature) if name == "energy" else MaxSoftmax(None)

    def fit(self, x, y=None):
        if self.name in ("knn", "mahalanobis"):
            x = matrix(x)
            if y is None:
                y = np.zeros(len(x), dtype="int64")
            y = np.asarray(y, dtype="int64")
            if np.any(y < 0):
                raise ValueError("OOD cannot enter detector fitting")
            _, y = np.unique(y, return_inverse=True)
            self.model.fit_features(self.torch.tensor(x, dtype=self.torch.float32),
                                    self.torch.tensor(y, dtype=self.torch.long))
        return self

    def score(self, x):
        torch = self.torch
        result = []
        with torch.inference_mode():
            for chunk in np.array_split(matrix(x), max(1, int(np.ceil(len(x)/self.batch_size)))):
                z = torch.tensor(chunk, dtype=torch.float32)
                if self.name in ("energy", "msp"):
                    logits = self.head(z)
                    if self.name == "msp":
                        logits = logits/self.temperature
                    predict = getattr(self.model, "predict_logits", None)
                    if predict is None:
                        predict = self.model.predict_features  # older logits API
                    out = predict(logits)
                else:
                    out = self.model.predict_features(z)
                result.append(out.detach().cpu().numpy().reshape(-1))
        scores = np.concatenate(result).astype(float)
        if not np.isfinite(scores).all():
            raise FloatingPointError(f"Nonfinite scores from {self.name}")
        return scores


class Spectral:
    """Exact unnormalized RBF insertion |delta lambda2|, matching the user prototype.

    Dense eigensolves: use only small matched-reference panels. No Cheeger claims.
    """
    def fit(self, x, y=None):
        from scipy.spatial.distance import pdist
        self.x = matrix(x).copy()
        if len(x) < 2 or len(x) > 256:
            raise ValueError("Exact spectral adapter requires 2..256 references")
        d = pdist(self.x); d = d[d > 0]
        self.bandwidth = float(np.median(d)) if len(d) else 1.
        self.base = self._lambda(self.x)
        return self

    def _lambda(self, x):
        from scipy.linalg import eigh
        w = np.exp(-cdist(x, x, 'sqeuclidean')/(2*self.bandwidth**2))
        np.fill_diagonal(w, 0)
        return float(eigh(np.diag(w.sum(1))-w, eigvals_only=True, subset_by_index=[1, 1])[0])

    def score(self, x):
        return np.array([abs(self._lambda(np.vstack((self.x, row)))-self.base) for row in matrix(x)])


def make_detector(name, backend="pytorch", k=5, head=None, temperature=1.):
    if name == "gaussian":
        return Gaussian()
    if name == "gaussian_nll":
        return Gaussian(nll=True)
    if name == "mahalanobis_shrinkage":
        return ShrinkageMahalanobis()
    if name == "spectral":
        return Spectral()
    if backend == "sklearn":
        if name != "knn":
            raise ValueError("sklearn backend supports knn; use gaussian or mahalanobis_shrinkage explicitly")
        return SKKNN(k)
    if name not in ("knn", "mahalanobis", "energy", "msp"):
        raise ValueError(f"Unknown detector {name}")
    return TorchOOD(name, k=k, head=head, temperature=temperature)
