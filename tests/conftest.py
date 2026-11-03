import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def example_graphs():
    return torch.load(os.path.join(ROOT, "examples", "example_graphs.pt"), weights_only=False)


@pytest.fixture(scope="session")
def weights():
    from guesser.inference import released_weights
    return released_weights(os.path.join(ROOT, "weights"))
