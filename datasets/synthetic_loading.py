"""
DD source-anchor loading for DO-ALL.

Loads a dataset-distilled (or coreset) source-anchor set stored as an ImageFolder.
Each item is returned as a 5-tuple ``(image, label, path, soft_label, index)``; DO-ALL
uses indices 0 (image), 1 (hard label) and 3 (source soft label / logits).

A per-image soft label is loaded from ``<image>.pt`` next to the image if present,
otherwise it falls back to a one-hot of the hard label.
"""

import logging

import torch
import torchvision
import torchvision.transforms as T

logger = logging.getLogger(__name__)


class ImageFolderAnchors(torchvision.datasets.ImageFolder):
    """ImageFolder that also yields a stored per-image soft label."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # DD anchors are already at the target resolution; just cast to float tensor.
        self.transform = T.Compose([T.ToTensor(), T.ConvertImageDtype(torch.float)])

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.transform(self.loader(path))
        try:
            soft_label = torch.load(path.replace("jpg", "pt"))
        except Exception:
            soft_label = torch.nn.functional.one_hot(
                torch.tensor(target), num_classes=len(self.classes)
            ).unsqueeze(0).to(torch.float32)
        return sample, target, path, soft_label, index


def get_synthetic_loader(data_root_dir: str, batch_size: int, workers: int = 4):
    """Build a loader over the DD source-anchor ImageFolder at ``data_root_dir``."""
    dataset = ImageFolderAnchors(root=data_root_dir)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        drop_last=False,
    )
    logger.info(f"[DO-ALL] anchor set: #img={len(dataset)} #batches={len(loader)} root={data_root_dir}")
    return dataset, loader
