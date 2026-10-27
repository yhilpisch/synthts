"""Portable inference bundles for the supported generator families."""

from pathlib import Path

import numpy as np
import torch

from synthts.data import ReturnScaler
from synthts.sequential import SeqDiscriminator, SeqGenerator
from synthts.timegan import (
    TimeGANDiscriminator, TimeGANEmbedder, TimeGANGenerator,
    TimeGANObsDiscriminator, TimeGANRecovery, TimeGANSupervisor,
)
from synthts.vanilla import Discriminator, Generator
from synthts.wgangp import Critic, WGANGenerator


MODEL_TYPES = {cls.__name__: cls for cls in (
    Generator, Discriminator, WGANGenerator, Critic, SeqGenerator,
    SeqDiscriminator, TimeGANEmbedder, TimeGANRecovery, TimeGANGenerator,
    TimeGANSupervisor, TimeGANDiscriminator, TimeGANObsDiscriminator,
)}


def save_bundle(path, models, scaler, metadata):
    """Save component weights, constructors, preprocessing, and provenance.

    This is an inference contract, not an optimiser/resume checkpoint.
    Store bundles outside Git. Metadata must contain plain Python values.
    """
    components = {}
    for role, model in models.items():
        name = type(model).__name__
        if MODEL_TYPES.get(name) is not type(model):
            raise ValueError(f'unsupported model type: {name}')
        components[role] = {
            'class': name, 'config': model.config,
            'state': {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()},
        }
    payload = {
        'schema_version': 1, 'models': components, 'metadata': metadata,
        'scaler': {key: np.asarray(getattr(scaler.scaler, key)).tolist()
                   for key in ('mean_', 'scale_', 'var_',
                               'n_features_in_', 'n_samples_seen_')},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_bundle(path, device: str='cpu'):
    """Load a supported inference bundle with restricted Torch loading.

    Returns a dictionary containing models, scaler, and metadata. Components
    are in evaluation mode; training can be restarted explicitly if needed.
    """
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload['schema_version'] != 1:
        raise ValueError('unsupported inference bundle schema')
    models = {}
    for role, component in payload['models'].items():
        cls = MODEL_TYPES[component['class']]
        model = cls(**component['config']).to(device)
        model.load_state_dict(component['state'])
        models[role] = model.eval()
    scaler = ReturnScaler()
    for key, value in payload['scaler'].items():
        setattr(scaler.scaler, key, np.asarray(value))
    return {'models': models, 'scaler': scaler,
            'metadata': payload['metadata']}
