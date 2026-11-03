"""Datasets over cached graph pairs, the forward/reverse augmentation and the pair collate."""
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class GraphPairDataset(Dataset):
    """List of {'graph_pair': (g_wt, g_mt), ...} items (as saved by build_graphs) held in memory."""

    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        if isinstance(item, dict):
            return item['graph_pair']
        return item


class ReverseAugmentedDataset(Dataset):
    """
    With augment=True every mutation appears twice: index 2i is WT -> MT with target ddg,
    index 2i+1 is MT -> WT with target -ddg. The target is attached to both graphs as .y.
    """

    def __init__(self, dataset, augment=True):
        self.dataset = dataset
        self.augment = augment

    def __len__(self):
        return len(self.dataset) * 2 if self.augment else len(self.dataset)

    def __getitem__(self, idx):
        if self.augment:
            original_idx, is_reverse = idx // 2, (idx % 2 == 1)
        else:
            original_idx, is_reverse = idx, False

        item = self.dataset[original_idx]
        g_wt, g_mt = item['graph_pair'] if isinstance(item, dict) else item
        g_1, g_2 = g_wt.clone(), g_mt.clone()

        base_y = g_mt.y if getattr(g_mt, 'y', None) is not None else torch.tensor([0.0])
        if is_reverse:
            target, first, second = -1.0 * base_y, g_2, g_1
        else:
            target, first, second = base_y, g_1, g_2
        first.y = target
        second.y = target
        return first, second


class CoordinateNoiseDataset(Dataset):
    """
    Clones each pair and, if use_noise, adds Gaussian noise (sigma, in A) to the CA
    coordinates and recomputes the CA-CA distance edge feature (edge_attr[:, 1]).
    With use_noise=False it only clones (used for validation and inference).
    """

    def __init__(self, dataset, sigma=0.1, use_noise=True):
        self.dataset = dataset
        self.sigma = sigma
        self.use_noise = use_noise

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        g_wt, g_mt = self.dataset[idx]
        g_wt, g_mt = g_wt.clone(), g_mt.clone()
        if self.use_noise:
            self._apply_noise(g_wt)
            self._apply_noise(g_mt)
        return g_wt, g_mt

    def _apply_noise(self, graph):
        if getattr(graph, 'ca_coords', None) is not None:
            noisy_ca = graph.ca_coords + torch.randn_like(graph.ca_coords) * self.sigma
            src, tgt = graph.edge_index
            if graph.edge_attr.shape[1] >= 2:
                graph.edge_attr[:, 1] = (noisy_ca[src] - noisy_ca[tgt]).norm(dim=1)


def collate_pairs(batch):
    """List of (g1, g2) -> (Batch of g1, Batch of g2). Targets travel as .y."""
    return Batch.from_data_list([b[0] for b in batch]), Batch.from_data_list([b[1] for b in batch])
