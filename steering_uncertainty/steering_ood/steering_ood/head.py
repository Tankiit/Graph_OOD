"""Skorch training; portable state_dict plus architecture/provenance metadata."""
import json
from pathlib import Path
import numpy as np
from .core import file_hash, load_cache, new_output, versions, write_json


def train_head(cache, output, seed=7, epochs=30, lr=.01, batch_size=128):
    import torch
    from skorch import NeuralNetClassifier
    data = load_cache(cache)
    x, y = data['head_x'].astype('float32'), data['head_y'].astype('int64')
    out = new_output(output)
    torch.manual_seed(seed); np.random.seed(seed)
    net = NeuralNetClassifier(torch.nn.Linear,
        module__in_features=x.shape[1], module__out_features=len(np.unique(y)),
        criterion=torch.nn.CrossEntropyLoss, optimizer=torch.optim.AdamW,
        optimizer__weight_decay=1e-4, lr=lr, max_epochs=epochs, batch_size=batch_size,
        train_split=None, iterator_train__shuffle=True, device='cpu', verbose=0)
    net.fit(x, y)
    torch.save(net.module_.cpu().state_dict(), out/'head.pt')
    net.history.to_file(str(out/'history.json'))
    write_json(out/'metadata.json', dict(in_features=x.shape[1], out_features=len(np.unique(y)),
        cache_sha256=file_hash(cache), seed=seed, epochs=epochs, lr=lr, batch_size=batch_size,
        validation='none; epochs fixed before calibration', versions=versions()))
    return out


def load_head(directory, cache=None):
    import torch
    p = Path(directory)
    meta = json.loads((p/'metadata.json').read_text())
    if cache and meta['cache_sha256'] != file_hash(cache):
        raise ValueError('Head was trained on a different cache')
    head = torch.nn.Linear(meta['in_features'], meta['out_features'])
    head.load_state_dict(torch.load(p/'head.pt', map_location='cpu', weights_only=True))
    head._steering_state_hash = file_hash(p/'head.pt')
    return head.eval()
