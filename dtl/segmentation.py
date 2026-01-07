"""Segmentation functions."""

from typing import Any

import numpy.typing as npt
import torch
from cellpose import models

from .utils import compact_mask


def segment(
    img: npt.NDArray[Any],
    model_type: str,
    diameter: float,
    device: str | None | torch.device = None,
    stitch_threshold: float = 0.0,
) -> npt.NDArray[Any]:
    """Run greyscale segmentation.

    Args:
        img: Image to segment (ZCYX).
        model_type: Name of the model.
        diameter: Expected diameter of objects.
        device: Name of the torch device.
        stitch_threshold: if stitch_threshold>0.0, masks are stitched in 3D to return volume segmentation.
    """
    device = (
        device if isinstance(device, torch.device) else _get_device(device)
    )
    model = models.CellposeModel(
        device=device,
        model_type=model_type,
    )

    # 0 = greyscale; [[2, 1]] = cells in green (2), nuclei in red (1)
    channels = [[0, 0]]
    do_3D = img.ndim == 4 and stitch_threshold == 0

    array, _flows, _styles = model.eval(
        img,
        channels=channels,
        do_3D=do_3D,
        z_axis=0,
        channel_axis=1,
        normalize=True,
        diameter=diameter,
        anisotropy=1.0,
        stitch_threshold=stitch_threshold,
    )

    return compact_mask(array)


def _get_device(device: str | None = None) -> torch.device:
    """Get a torch device given the available backends."""
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
