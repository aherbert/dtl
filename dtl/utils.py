"""Utility functions."""

import csv
import math
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.spatial
import skimage.filters
import skimage.measure
import tifffile
from scipy import ndimage as ndi
from skimage.segmentation import clear_border
from skimage.util import map_array

_OFFSET = 0.0


def find_images(
    files: list[str],
) -> list[str]:
    """Find the images in the input file or directory paths.

    Adds any file with extension ICS, or any TIFF with 4 dimensions (CZYX).

    Args:
        files: Image file or directory paths.

    Returns:
        list of image files

    Raises:
        RuntimeError: if a path is not a file or directory
    """
    images = []
    for fn in files:
        if os.path.isfile(fn):
            if _is_image(fn):
                images.append(fn)
        elif os.path.isdir(fn):
            for file in os.listdir(fn):
                file = os.path.join(fn, file)
                if _is_image(file):
                    images.append(file)
        else:
            raise RuntimeError("Not a file or directory: " + fn)
    return images


def _is_image(fn: str) -> bool:
    """Check if the file is a target image."""
    base, suffix = os.path.splitext(fn)
    suffix = suffix.lower()
    if suffix == ".ics" or suffix == ".czi":
        return True
    if suffix == ".tiff":
        # Require CZYX. Ignores result image masks (YX) from a directory.
        with tifffile.TiffFile(fn) as tif:
            # image shape
            shape = tif.series[0].shape
            return len(shape) == 4
    return False


def find_objects(
    label_image: npt.NDArray[Any],
) -> list[tuple[int, int, tuple[slice, ...]]]:
    """Find the objects in the labeled image.

    Identifies the size and bounding box of objects. The bounding box (bb) is
    a tuple of slices of [min, max) for each dimension
    suitable for extracting the region using im[bb].

    This method combines numpy.bincount with scipy.ndimage.find_objects.

    Args:
        label_image: Label image.

    Returns:
        list of (ID, size, tuple(slice(min, max), ...))
    """
    data = []
    h = np.bincount(label_image.ravel())
    objects = ndi.find_objects(label_image)
    for i, bb in enumerate(objects):
        if bb is None:
            continue
        label = i + 1
        data.append((label, int(h[label]), bb))
    return data


def _find_border(
    label_image: npt.NDArray[Any], labels: list[int]
) -> npt.NDArray[Any]:
    """Find border pixels for all labelled objects."""
    mask = np.zeros(label_image.shape, dtype=bool)
    eroded = np.zeros(label_image.shape, dtype=bool)
    strel = ndi.generate_binary_structure(label_image.ndim, 1)
    for label in labels:
        target = label_image == label
        mask = mask | target
        # erosion must not erode the object face at the border
        eroded = eroded | ndi.binary_erosion(target, strel, border_value=1)
    border = label_image * mask - label_image * eroded
    return border.astype(label_image.dtype)


def object_threshold(
    im: npt.NDArray[Any],
    label_image: npt.NDArray[Any],
    fun: Callable[[npt.NDArray[Any]], int],
    objects: list[tuple[int, int, tuple[slice, ...]]] | None = None,
    fill_holes: int = 0,
    min_size: int = 0,
) -> npt.NDArray[Any]:
    """Threshold the pixels in each masked object.

    The thresholding function accepts a histogram of pixel
    value counts and returns the threshold level above
    which pixels are foreground.

    Args:
        im: Image pixels.
        label_image: Label image.
        objects: Objects of interest (computed using find_objects).
        fun: Thresholding method.
        fill_holes: Remove contiguous holes smaller than the specified size.
        min_size: Minimum size of thresholded regions.

    Returns:
        mask of thresholded objects
    """
    if objects is None:
        objects = find_objects(label_image)
    final_mask = np.zeros(im.shape, dtype=int)
    total = 0
    for label, _area, bbox in objects:
        # crop for efficiency
        crop_i = im[bbox]
        crop_m = label_image[bbox]
        # threshold the object
        target = crop_m == label
        values = crop_i[target]
        h = np.bincount(values.ravel())
        t = fun(h)
        # create labels
        target = target & (crop_i > t)
        if fill_holes:
            target = skimage.morphology.remove_small_holes(
                target, fill_holes, out=target
            )
        labels, n = skimage.measure.label(target, return_num=True)
        # TODO: Watershed to split touching foci

        if min_size > 0:
            labels, n = filter_segmentation(
                labels, border=-1, min_size=min_size
            )
        if total:
            labels[labels != 0] += total
        total += n

        final_mask[bbox] += labels

    return compact_mask(final_mask, m=total)


def compact_mask(mask: npt.NDArray[Any], m: int = 0) -> npt.NDArray[Any]:
    """Compact the int datatype to the smallest required to store all mask IDs.

    Args:
        mask: Segmentation mask.
        m: Maximum value in the mask.

    Returns:
        Compact segmentation mask.
    """
    if m == 0:
        m = mask.max()
    if m < 2**8:
        return mask.astype(np.uint8)
    if m < 2**16:
        return mask.astype(np.uint16)
    return mask


def threshold_method(
    name: str, std: float = 4, q: float = 0.5
) -> Callable[[npt.NDArray[Any]], int]:
    """Create a threshold function.

    Supported functions:

    mean_plus_std: Threshold using (mean + n * std).
    mean_plus_std_q: Threshold using (mean + n * std) using lowest quantile (q) of values.
    otsu: Otsu thresholding.
    yen: Yen's method.
    minimum: Smoth the histogram until only two maxima and return the mid-point between them.

    Args:
        name: Method name.
        std: Factor n for (mean + n * std) mean_plus_std method.
        q: Quantile for lowest set of values.

    Returns:
        Callable threshold method.
    """
    if name == "mean_plus_std":

        def mean_plus_std(h: npt.NDArray[Any]) -> int:
            values = np.arange(len(h))
            probs = h / np.sum(h)
            mean = np.sum(probs * values)
            sd = np.sqrt(np.sum(probs * (values - mean) ** 2))
            t = np.clip(math.floor(mean + std * sd), 0, values[-1])
            return int(t)

        return mean_plus_std

    if name == "mean_plus_std_q":

        def mean_plus_std_q(h: npt.NDArray[Any]) -> int:
            # Find lowest n values to achieve quantile (or next above)
            cumul = np.cumsum(h)
            n = np.searchsorted(cumul / cumul[-1], q) + 1
            values = np.arange(n)
            probs = h[:n] / cumul[n - 1]
            mean = np.sum(probs * values)
            sd = np.sqrt(np.sum(probs * (values - mean) ** 2))
            t = np.clip(math.floor(mean + std * sd), 0, len(h) - 1)
            return int(t)

        return mean_plus_std_q

    if name == "otsu":

        def otsu(h: npt.NDArray[Any]) -> int:
            return int(skimage.filters.threshold_otsu(hist=h))

        return otsu

    if name == "yen":

        def yen(h: npt.NDArray[Any]) -> int:
            return int(skimage.filters.threshold_yen(hist=h))

        return yen

    if name == "minimum":

        def minimum(h: npt.NDArray[Any]) -> int:
            try:
                return int(skimage.filters.threshold_minimum(hist=h))
            except RuntimeError as e:
                print(e)
                return -1

        return minimum

    raise Exception(f"Unknown method: {name}")


def filter_method(
    sigma1: float = 0, sigma2: float = 0
) -> Callable[[npt.NDArray[Any]], npt.NDArray[Any]]:
    """Create a filter function.

    The filter will be a Gaussian smoothing filter or a difference of Gaussians
    if the second radius is larger than the first.

    Args:
        sigma1: First Gaussian filter standard deviation.
        sigma2: Second Gaussian filter standard deviation.

    Returns:
        Callable filter method.
    """
    if sigma1 > 0 or sigma2 > sigma1:

        def f(im: npt.NDArray[Any]) -> npt.NDArray[Any]:
            # 2D filtering of YX
            axes = (im.ndim - 2, im.ndim - 1)

            # Foreground smoothing
            im1 = (
                ndi.gaussian_filter(im, sigma1, mode="mirror", axes=axes)
                if sigma1 > 0
                else im
            )

            if sigma2 > sigma1:
                # Background subtraction
                background = ndi.gaussian_filter(
                    im, sigma2, mode="mirror", axes=axes
                )
                # Do not allow negative values but return as same datatype
                # to support unsigned int images.
                result = im1.astype(np.float64) - background
                im1 = (result - np.min(result)).astype(im.dtype)

            return im1

        return f

    def identity(im: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return im

    return identity


def spot_analysis(
    label_image: npt.NDArray[Any],
    objects: list[tuple[int, int, tuple[slice, ...]]],
    im1: npt.NDArray[Any],
    label1: npt.NDArray[Any],
    im2: npt.NDArray[Any],
    label2: npt.NDArray[Any],
    anisotropy: float = 1.0,
) -> list[tuple[int | float, ...]]:
    """Analyse spots in image 1 within parent objects.

    The distance between the spot and the edge of the parent is computed.
    The distance between the spot and internal objects in image 2 is computed.
    The results contain the closest neighbour distance and type.

    label: Label.
    parent: Label of parent.
    size: Size of label.
    mean intensity: Mean intensity of label image.
    cx: Centroid x.
    cy: Centroid y.
    cz: Centroid z.
    ox: Centroid x of neighbour.
    oy: Centroid y of neighbour.
    oz: Centroid z of neighbour.
    type: Type of neighbour.
    distance: Distance to nearest neighbour.

    Args:
        label_image: Label iamge.
        objects: Objects of interest (computed using find_objects).
        im1: First image.
        label1: First image object labels (spots).
        im2: Second image.
        label2: Second image object labels (internal objects).
        anisotropy: Anisotropy scaling applied to z dimension.

    Returns:
        analysis results
    """
    results: list[tuple[int | float, ...]] = []
    for parent, _, bbox in objects:
        # Simplify the group object by cropping and relabel
        mask = label_image[bbox] == parent
        c_label1, id1 = relabel(label1[bbox] * mask)
        if len(id1) == 0:
            # Nothing to analyse
            continue
        c_label_, id_ = relabel(label_image[bbox] * mask)
        c_label2, id2 = relabel(label2[bbox] * mask)
        c_im1 = im1[bbox]
        oz, oy, ox = bbox[0].start, bbox[1].start, bbox[2].start
        # Objects
        objects1 = find_objects(c_label1)
        data1 = analyse_objects(c_im1, c_label1, objects1, (oz, oy, ox))
        # Borders as KD-tree
        z, y, x = np.nonzero(
            _find_border(c_label_, [x + 1 for x in range(len(id_))])
        )
        tree1 = scipy.spatial.KDTree(_to_coords(x, y, z, anisotropy))
        # TODO: Add option to find distance to border or centroid of internal objects
        z, y, x = np.nonzero(
            _find_border(c_label2, [x + 1 for x in range(len(id2))])
        )
        tree2 = scipy.spatial.KDTree(_to_coords(x, y, z, anisotropy))
        for i, d in enumerate(data1):
            # Compute distance to borders. Note the centroids are offset.
            offset = np.array([ox, oy, oz]) + _OFFSET
            coords = np.array(d[-3:]) - offset
            coords[2] *= anisotropy
            r1 = tree1.query(coords)
            r2 = tree2.query(coords)
            # Save closest
            if r1[0] < r2[0]:
                r = r1
                c = tree1.data[r1[1]]
            else:
                r = r2
                c = tree2.data[r2[1]]
            c[2] = round(c[2] / anisotropy)
            c = c + offset
            results.append(
                (
                    int(id1[i]),
                    parent,
                    objects1[i][1],
                    d[0] / objects1[i][1],
                    d[1],
                    d[2],
                    d[3],
                    float(c[0]),
                    float(c[1]),
                    float(c[2]),
                    0 if r is r1 else 1,
                    r[0],
                )
            )

    return results


def _to_coords(
    x: npt.NDArray[np.int_],
    y: npt.NDArray[np.int_],
    z: npt.NDArray[np.int_],
    anisotropy: float,
) -> npt.NDArray[np.float64]:
    """Convert indices to coordinates."""
    xx = x.astype(np.float64)
    yy = y.astype(np.float64)
    zz = z.astype(np.float64) * anisotropy
    return np.column_stack([xx, yy, zz])


def filter_segmentation(
    mask: npt.NDArray[Any],
    border: int = 5,
    relabel: bool = True,
    min_size: int = 10,
) -> tuple[npt.NDArray[Any], int]:
    """Removes border objects and filters small objects from segmentation mask.

    The size of the border can be specified. Use a negative size to skip removal of
    border objects. Only the border of XY planes is considered (i.e. objects in the
    first and last Z plane of the stack are not filtered).

    Objects are optionally relabeled to be continuous from 1.
    Note: Removal of border objects can result in missing labels from the
    original [0, N] label IDs.

    Args:
        mask: unfiltered segmentation mask
        border: width of the border examined (negative to disable)
        relabel: Set to True to relabel objects in [0, N]
        min_size: Minimum size of objects.

    Returns:
        filtered segmentation mask, number of objects (N)
    """
    cleared: npt.NDArray[Any]
    if border < 0:
        cleared = mask
    else:
        # Input can be [Z]YX
        if mask.ndim == 2:
            cleared = clear_border(mask, buffer_size=border)  # type: ignore[no-untyped-call]
        else:
            assert mask.ndim == 3
            clear = np.ones(mask.shape[-2:], dtype=bool)
            n = border + 1
            clear[:n, :] = False
            clear[-n:, :] = False
            clear[:, :n] = False
            clear[:, -n:] = False
            clear = np.repeat([clear], mask.shape[0], axis=0)
            cleared = clear_border(mask, mask=clear)  # type: ignore[no-untyped-call]

    sizes = np.bincount(cleared.ravel())
    mask_sizes = sizes > min_size
    mask_sizes[0] = 0
    cells_cleaned = mask_sizes[cleared]
    new_mask = cells_cleaned * mask
    n = np.sum(mask_sizes)
    if relabel:
        old_id = np.arange(len(sizes))[mask_sizes]
        new_mask = map_array(
            new_mask, old_id, np.arange(1, 1 + n), out=np.zeros_like(mask)
        )
    return new_mask, n  # type: ignore[no-any-return]


def relabel(
    mask: npt.NDArray[Any],
) -> tuple[npt.NDArray[Any], list[int]]:
    """Relabels the mask to be continuous from 1.

    The old id array contains the original ID for each new label:

    original id = old_id[label - 1]

    Args:
        mask: unfiltered segmentation mask

    Returns:
        relabeled mask, old_id
    """
    sizes = np.bincount(mask.ravel())
    sizes[0] = 0
    mask_sizes = sizes != 0
    old_id = np.arange(len(sizes))[mask_sizes]
    n = len(old_id)
    new_mask = map_array(
        mask, old_id, np.arange(1, 1 + n), out=np.zeros_like(mask)
    )
    return new_mask, old_id  # type: ignore[no-any-return]


def analyse_objects(
    im: npt.NDArray[Any],
    label_image: npt.NDArray[Any],
    objects: list[tuple[int, int, tuple[slice, ...]]],
    offset: tuple[int, int, int] = (0, 0, 0),
) -> list[tuple[float, float, float, float]]:
    """Extract the intensity and centroids for all labelled objects.

    Centroids use _OFFSET as the centre of voxels.

    Args:
        im: Image.
        label_image: Label image.
        objects: List of (ID, size, (slice(min, max), ...)).
        offset: Offset to add to the centre (Z,Y,X).

    Returns:
        list of (intensity, cx, cy, cz)
    """
    data = []
    for label, _, bbox in objects:
        crop_image = im[bbox]
        crop_label = label_image[bbox]
        mask = crop_label == label
        intensity = float((crop_image * mask).sum())
        z, y, x = np.nonzero(mask)
        weights = crop_image[mask].ravel()
        weights = weights / weights.sum()
        cz, cy, cx = (
            float(np.sum(z * weights) + bbox[0].start + offset[0] + _OFFSET),
            float(np.sum(y * weights) + bbox[1].start + offset[1] + _OFFSET),
            float(np.sum(x * weights) + bbox[2].start + offset[2] + _OFFSET),
        )
        data.append((intensity, cx, cy, cz))

    return data


def spot_summary(
    results: list[tuple[int | float, ...]],
    n: int,
) -> list[tuple[int, ...]]:
    """Summarise the spots results by parent.

    Parents may not contain any spots so the total number of parents
    must be provided.

    label: Parent label.
    count1: Number of spots closest to edge.
    count2: Number of spots closest to internal.

    Args:
        results: Spot analysis results.
        n: Number of parent objects.

    Returns:
        summary
    """
    # Count of spots in each parent label
    count1: dict[int, int] = {}
    count2: dict[int, int] = {}
    for _label, parent, *other in results:
        parent = int(parent)
        cls = other[-2]
        d = count1 if cls == 0 else count2
        d[parent] = d.get(parent, 0) + 1

    summary: list[tuple[int, ...]] = []
    for label in range(1, n + 1):
        summary.append((label, count1.get(label, 0), count2.get(label, 0)))
    return summary


def format_spot_results(
    results: list[tuple[int | float, ...]],
    scale: float = 0,
) -> list[tuple[int | float | str, ...]]:
    """Format the spot results.

    If a scale is provided an additional scaled distance column is returned.
    Units are assumed to be micrometers (μm).

    Args:
        results: Spot analysis results.
        scale: Optional distance scale (micrometers/pixel).

    Returns:
        Formatted results
    """
    out: list[tuple[int | float | str, ...]] = []
    out.append(
        (
            "label",
            "parent",
            "size",
            "mean",
            "cx",
            "cy",
            "cz",
            "ox",
            "oy",
            "oz",
            "type",
            "distance",
        )
    )
    if scale:
        out[0] = out[0] + ("distance (μm)",)
    for data in results:
        # Change type
        t = "edge" if data[-2] == 0 else "internal"
        formatted = data[:-2] + (t,) + data[-1:]
        if scale:
            formatted = formatted + (data[-1] * scale,)
        out.append(formatted)

    return out


def format_summary_results(
    summary: list[tuple[int, ...]],
    object_data: dict[int, tuple[int, float, float, float, float]]
    | None = None,
) -> list[tuple[int | float | str, ...]]:
    """Format the spot summary results.

    Args:
        summary: Spot analysis summary results.
        object_data: Optional data for each object (size, intensity, cx, cy, cz)

    Returns:
        Formatted results
    """
    out: list[tuple[int | float | str, ...]] = []
    out.append(
        (
            "label",
            "size",
            "intensity",
            "cx",
            "cy",
            "cz",
            "edge",
            "internal",
            "total",
        )
    )
    for data in summary:
        parent = int(data[0])
        parent_data = (
            object_data.get(parent, (0, 0, 0, 0, 0))
            if object_data
            else (0, 0, 0, 0, 0)
        )
        formatted = data[:1] + parent_data + data[1:] + (data[-2] + data[-1],)
        out.append(formatted)

    return out


def save_csv(
    fn: str,
    data: list[Any],
) -> None:
    """Save the data to a CSV file.

    Args:
        fn: File name.
        data: Data.
    """
    with open(fn, "w", newline="") as f:
        csv.writer(f).writerows(data)
