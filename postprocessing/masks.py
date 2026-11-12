"""Fixed Figure 5 RI tissue masks, independent of motion and onset."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def largest_filled_component(binary):
    labels, count = ndi.label(binary)
    if not count:
        raise ValueError("No tissue component: check the RI threshold and ROI")
    largest = labels == 1 + np.argmax(np.bincount(labels.ravel())[1:])
    return ndi.binary_fill_holes(largest)


def generate_masks(mean_ri, *, threshold=.007, sigma=1., regions=None):
    """Gaussian(mean RI) > threshold, largest component, then hole filling.

    A region ROI is [y_start,y_stop,x_start,x_stop], with exclusive stops.
    Smoothing precedes ROI extraction, as in the Figure 5 analysis. Multiple
    regions use this same function independently and must not overlap.
    """
    mean_ri = np.asarray(mean_ri)
    if mean_ri.ndim != 2 or not np.isfinite(mean_ri).all():
        raise ValueError("mean_ri must be a finite 2D section")
    if not np.isfinite(threshold) or sigma < 0 or not np.isfinite(sigma):
        raise ValueError("Invalid RI mask threshold or Gaussian sigma")
    if regions is None:
        regions = [dict(name="tissue", roi=[0, mean_ri.shape[0], 0, mean_ri.shape[1]])]
    smooth = ndi.gaussian_filter(mean_ri, sigma, mode="reflect")
    masks, names = [], []
    for region in regions:
        y0, y1, x0, x1 = map(int, region["roi"])
        if not (0 <= y0 < y1 <= mean_ri.shape[0] and 0 <= x0 < x1 <= mean_ri.shape[1]):
            raise ValueError(f"ROI outside section: {region}")
        mask = np.zeros(mean_ri.shape, dtype=bool)
        mask[y0:y1, x0:x1] = largest_filled_component(smooth[y0:y1, x0:x1] > threshold)
        masks.append(mask)
        names.append(region["name"])
    masks = np.array(masks)
    if len(set(names)) != len(names) or (masks.sum(axis=0) > 1).any():
        raise ValueError("Mask regions must have unique names and disjoint support")
    return masks, names


def validate_masks(masks, shape):
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None]
    if masks.dtype != np.bool_ or masks.shape[1:] != tuple(shape) or not masks.any(axis=(1, 2)).all():
        raise ValueError("Provide nonempty boolean masks with matching section dimensions")
    if (masks.sum(axis=0) > 1).any():
        raise ValueError("Mask regions overlap")
    return masks
