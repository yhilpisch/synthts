"""Non-interactive, deterministic resource settings for local and CI tests."""

import matplotlib
import torch

matplotlib.use('Agg')
torch.set_num_threads(1)
