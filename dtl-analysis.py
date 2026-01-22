#!/usr/bin/env python3
"""Program to analyse the distance to the lamina within nuclei objects."""

import argparse


def main() -> None:
    """Program to analyse the distance to the lamina within nuclei objects."""
    parser = argparse.ArgumentParser(
        description="""Program to analyse the distance to the lamina within nuclei objects."""
    )
    _ = parser.add_argument(
        "image",
        nargs="+",
        help="Image, or image directory",
        metavar="image/dir",
    )
    _ = parser.add_argument(
        "--object-ch",
        default=2,
        type=int,
        help="Object channel (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--spot-ch",
        default=1,
        type=int,
        help="Spot channel (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--lamina-ch",
        default=0,
        type=int,
        help="Lamina channel (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--scale",
        default=0.065,
        type=float,
        help="Scale (um/px) (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--anisotropy",
        default=2 / 0.65,
        type=float,
        help="Anisotropy (scale of z-dimension) (default: %(default)s)",
    )

    group = parser.add_argument_group("Object Options")
    _ = group.add_argument(
        "--model-type",
        default="cyto3",
        help="Name of default model (default: %(default)s)",
    )
    _ = group.add_argument(
        "--diameter",
        type=float,
        default=150,
        help="Expected nuclei diameter (pixels) (default: %(default)s)",
    )
    _ = group.add_argument(
        "--device",
        type=str,
        help="Torch device name (default: auto-detect)",
    )
    _ = group.add_argument(
        "--stitch-threshold",
        type=float,
        default=0.5,
        help="If above zero, masks are stitched in 3D to return volume segmentation (default: %(default)s)",
    )
    _ = group.add_argument(
        "--border",
        default=20,
        type=int,
        help="Border to exclude objects (pixels; negative to disable) (default: %(default)s)",
    )
    _ = group.add_argument(
        "--min-size",
        default=10000,
        type=int,
        help="Minimum object size (pixels) (default: %(default)s)",
    )

    group = parser.add_argument_group("Spot Threshold Options")
    _ = group.add_argument(
        "--spot-sigma",
        default=1.5,
        type=float,
        help="Gaussian smoothing filter standard deviation (default: %(default)s)",
    )
    _ = group.add_argument(
        "--spot-sigma2",
        default=0,
        type=float,
        help="Background Gaussian smoothing filter standard deviation (default: %(default)s)",
    )
    _ = group.add_argument(
        "--spot-method",
        default="otsu",
        choices=["mean_plus_std", "mean_plus_std_q", "otsu", "yen", "minimum"],
        help="Thresholding method (default: %(default)s)",
    )
    _ = group.add_argument(
        "--spot-threshold",
        default=0,
        type=int,
        help="Manual threshold (overrides spot method) (default: %(default)s)",
    )
    _ = group.add_argument(
        "--spot-std",
        default=7,
        type=float,
        help="Std.dev above the mean (default: %(default)s)",
    )
    _ = group.add_argument(
        "--spot-quantile",
        default=0.75,
        type=float,
        help="Quantile for lowest value used in mean_plus_std_q (default: %(default)s)",
    )
    _ = group.add_argument(
        "--spot-global",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use threshold across all objects; else threshold within each object (default: %(default)s)",
    )
    _ = group.add_argument(
        "--spot-split",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Split using watershed: 1=Distance transform; 2=Image (default: %(default)s)",
    )

    group = parser.add_argument_group("Lamina Threshold Options")
    _ = group.add_argument(
        "--lamina-sigma",
        default=1.5,
        type=float,
        help="Gaussian smoothing filter standard deviation (default: %(default)s)",
    )
    _ = group.add_argument(
        "--lamina-sigma2",
        default=0,
        type=float,
        help="Background Gaussian smoothing filter standard deviation (default: %(default)s)",
    )
    _ = group.add_argument(
        "--lamina-method",
        default="mean_plus_std_q",
        choices=["mean_plus_std", "mean_plus_std_q", "otsu", "yen", "minimum"],
        help="Thresholding method (default: %(default)s)",
    )
    _ = group.add_argument(
        "--lamina-threshold",
        default=0,
        type=int,
        help="Manual threshold (overrides lamina method) (default: %(default)s)",
    )
    _ = group.add_argument(
        "--lamina-std",
        default=10,
        type=float,
        help="Std.dev above the mean (default: %(default)s)",
    )
    _ = group.add_argument(
        "--lamina-quantile",
        default=0.75,
        type=float,
        help="Quantile for lowest value used in mean_plus_std_q (default: %(default)s)",
    )
    _ = group.add_argument(
        "--lamina-global",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use threshold across all objects; else threshold within each object (default: %(default)s)",
    )

    group = parser.add_argument_group("Threshold Options")
    _ = group.add_argument(
        "--fill-holes",
        default=2,
        type=int,
        help="Remove contiguous holes smaller than the specified size (pixels) (default: %(default)s)",
    )
    _ = group.add_argument(
        "--min-spot-size",
        default=6,
        type=int,
        help="Minimum spot size (pixels) (default: %(default)s)",
    )

    group = parser.add_argument_group("Distance Options")
    _ = group.add_argument(
        "--lamina-edge",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Include distances to lamina objects that touch the edge; else ignore as edge (default: %(default)s)",
    )
    _ = group.add_argument(
        "--face-border",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Include image face voxels as object border; else extend objects beyond the image bounds (default: %(default)s)",
    )

    group = parser.add_argument_group("View Options")
    _ = group.add_argument(
        "--view",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Show results in graphical viewer (default: %(default)s)",
    )
    _ = group.add_argument(
        "--channel-names", nargs="+", default=[], help="Channel names"
    )
    _ = group.add_argument(
        "--visible-channels",
        nargs="+",
        type=int,
        default=[],
        help="Visible channels (default is the spot channels)",
    )
    _ = group.add_argument(
        "--upper-limit",
        default=99.999,
        type=float,
        help="Upper contrast limit (percentile) (default: %(default)s)",
    )
    _ = group.add_argument(
        "--aniso",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use anisotropy in graphical viewer (default: %(default)s)",
    )

    group = parser.add_argument_group("Other Options")
    _ = group.add_argument(
        "--repeat",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Repeat from stage: 1=object segmentation; 2=spot identification; 3=analysis",
    )

    args = parser.parse_args()

    # Delay imports until argument parsing succeeds
    import logging

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger.info("Initialising")

    import json
    import os
    from typing import Any

    # TODO: try pylibics as BioIO is complex to install
    import bioio_bioformats
    import numpy as np
    import numpy.typing as npt
    from bioio import BioImage
    from tifffile import imread, imwrite

    from dtl.utils import (
        analyse_objects,
        filter_method,
        find_images,
        find_objects,
        format_spot_results,
        format_summary_results,
        object_threshold,
        save_csv,
        spot_analysis,
        spot_summary,
        threshold_method,
    )

    spot_std = args.spot_std
    lamina_std = args.lamina_std

    images = find_images(args.image)
    logger.info("Identified %d images", len(images))
    for image_fn in images:
        logger.info("Processing %s", image_fn)
        base, suffix = os.path.splitext(image_fn)

        img = BioImage(image_fn, reader=bioio_bioformats.Reader)
        if img.dims.T != 1:
            raise RuntimeError("Expected CZYX image: " + str(img.dims))
        # CZYX
        image = img.data.squeeze(axis=0)

        stage = args.repeat if args.repeat else 10

        label_fn = f"{base}.objects.tiff"
        if stage <= 1 or not os.path.exists(label_fn):
            from dtl.segmentation import segment
            from dtl.utils import filter_segmentation

            image_ch = image[args.object_ch]
            shape = image_ch.shape

            label_image = segment(
                # Reshape to ZCYX with C=1
                image_ch.reshape((shape[0],) + (1,) + shape[-2:]),
                args.model_type,
                args.diameter,
                device=args.device,
                stitch_threshold=args.stitch_threshold,
            )
            label_image, n_objects = filter_segmentation(
                label_image,
                border=args.border,
                min_size=args.min_size,
            )

            imwrite(label_fn, label_image, compression="zlib")
        else:
            logger.info("Loading %s", label_fn)
            label_image = imread(label_fn)
            n_objects = np.max(label_image)
        logger.info("Identified %d objects: %s", n_objects, label_fn)

        # Spot identification
        spots_fn = f"{base}.spots.tiff"
        im1 = image[args.spot_ch]
        if stage <= 2 or not os.path.exists(spots_fn):
            if (
                args.spot_method == "mean_plus_std_q"
                and spot_std == args.spot_std
            ):
                # make std shift equivalent to get the same thresholding level if a normal distribution is truncated
                import scipy.stats

                m, v = scipy.stats.truncnorm(
                    -10, scipy.stats.norm.ppf(args.spot_quantile)
                ).stats("mv")
                spot_std = (spot_std - m) / np.sqrt(v)
                logger.info(
                    "Adjusted spot threshold %s to %.3f using normal distribution truncated at cdf=%s",
                    args.spot_std,
                    spot_std,
                    args.spot_quantile,
                )
            spot_fun = threshold_method(
                args.spot_method,
                std=spot_std,
                q=args.spot_quantile,
                threshold=args.spot_threshold,
            )
            spot_filter_fun = filter_method(args.spot_sigma, args.spot_sigma2)

            # thresholding requires an integer image.
            # convert to uint16
            logger.info("Filtering spot channel %d", args.spot_ch)
            filtered = spot_filter_fun(im1)
            filtered = (
                (filtered - np.min(filtered)) * (2**16 / np.ptp(filtered))
            ).astype(np.int_)
            logger.info("Identifying spots")
            label1 = object_threshold(
                filtered,
                label_image,
                spot_fun,
                fill_holes=args.fill_holes,
                min_size=args.min_spot_size,
                split_objects=args.spot_split,
                global_threshold=args.spot_global,
            )
            imwrite(spots_fn, label1, compression="zlib")
        else:
            logger.info("Loading %s", spots_fn)
            label1 = imread(spots_fn)
        logger.info(
            "Identified %d spots in channel %d: %s",
            np.max(label1),
            args.spot_ch,
            spots_fn,
        )

        # Lamina identification
        # Lamina may have invaginations (spots) inside the nucleus
        # or a peripheral ring when the plane intersects the membrane.
        lamina_fn = f"{base}.lamina.tiff"
        im2 = image[args.lamina_ch]
        if stage <= 2 or not os.path.exists(lamina_fn):
            if (
                args.lamina_method == "mean_plus_std_q"
                and lamina_std == args.lamina_std
            ):
                # make std shift equivalent to get the same thresholding level if a normal distribution is truncated
                import scipy.stats

                m, v = scipy.stats.truncnorm(
                    -10, scipy.stats.norm.ppf(args.lamina_quantile)
                ).stats("mv")
                lamina_std = (lamina_std - m) / np.sqrt(v)
                logger.info(
                    "Adjusted lamina threshold %s to %.3f using normal distribution truncated at cdf=%s",
                    args.lamina_std,
                    lamina_std,
                    args.lamina_quantile,
                )
            lamina_fun = threshold_method(
                args.lamina_method,
                std=lamina_std,
                q=args.lamina_quantile,
                threshold=args.lamina_threshold,
            )
            lamina_filter_fun = filter_method(
                args.lamina_sigma, args.lamina_sigma2
            )

            # thresholding requires an integer image.
            # convert to uint16
            logger.info("Filtering lamina channel %d", args.lamina_ch)
            filtered = lamina_filter_fun(im2)
            filtered = (
                (filtered - np.min(filtered)) * (2**16 / np.ptp(filtered))
            ).astype(np.int_)
            logger.info("Identifying lamina")
            label2 = object_threshold(
                filtered,
                label_image,
                lamina_fun,
                fill_holes=args.fill_holes,
                min_size=args.min_spot_size,
                global_threshold=args.lamina_global,
            )
            imwrite(lamina_fn, label2, compression="zlib")
        else:
            logger.info("Loading %s", lamina_fn)
            label2 = imread(lamina_fn)
        logger.info(
            "Identified %d lamina spots in channel %d: %s",
            np.max(label2),
            args.lamina_ch,
            lamina_fn,
        )

        # # Analysis cannot be loaded from previous results, just skip
        spot_fn = f"{base}.spots.csv"
        summary_fn = f"{base}.summary.csv"

        # Create an anlysis function to allow this to be repeated.
        # Settings used: distance, size, min_size, neighbour_distance.
        # These could be added as input for the function and updated through the GUI.
        def analysis_fun(
            label_image: npt.NDArray[Any],
            label1: npt.NDArray[Any],
            label2: npt.NDArray[Any],
            # fix values for loop variables
            image: npt.NDArray[Any] = image,
            im1: npt.NDArray[Any] = im1,
            im2: npt.NDArray[Any] = im2,
            spot_fn: str = spot_fn,
            summary_fn: str = summary_fn,
        ) -> None:
            # find objects
            objects = find_objects(label_image)

            results = spot_analysis(
                label_image,
                objects,
                im1,
                label1,
                im2,
                label2,
                anisotropy=args.anisotropy,
                remove_internal_edge=not args.lamina_edge,
                remove_face_border=not args.face_border,
            )

            formatted = format_spot_results(results, scale=args.scale)
            logger.info("Saving spot results: %s", spot_fn)
            save_csv(spot_fn, formatted)

            o_data = analyse_objects(
                image[args.object_ch], label_image, objects
            )
            # Collate to {ID: (size, intensity, cx, cy, cz)}
            object_data = {
                x[0]: (x[1],) + y for x, y in zip(objects, o_data, strict=True)
            }

            summary = spot_summary(results, len(objects))
            formatted2 = format_summary_results(
                summary, object_data=object_data
            )
            logger.info("Saving summary results: %s", summary_fn)
            save_csv(summary_fn, formatted2)

        if stage <= 3 or not (
            os.path.exists(spot_fn) and os.path.exists(summary_fn)
        ):
            analysis_fun(label_image, label1, label2)
        else:
            logger.info("Existing spot results: %s", spot_fn)
            logger.info("Existing summary results: %s", summary_fn)

        # Save settings if all OK
        fn = f"{base}.settings.json"
        logger.info("Saving settings: %s", fn)
        with open(fn, "w") as f:
            d = vars(args)
            d["image"] = image_fn
            json.dump(d, f, indent=2)

        if args.view:
            logger.info("Launching viewer")
            import pandas as pd
            from skimage.measure import label

            from dtl.gui import (
                add_analysis_function,
                create_viewer,
                show_viewer,
            )
            from dtl.utils import compact_mask

            label_df = pd.read_csv(summary_fn)
            spot_df = pd.read_csv(spot_fn)

            visible_channels = (
                args.visible_channels
                if args.visible_channels
                else [args.spot_ch, args.lamina_ch]
            )
            viewer = create_viewer(
                image,
                label_image,
                label1,
                label2,
                channel_names=args.channel_names,
                visible_channels=visible_channels,
                label_df=label_df,
                spot_df=spot_df,
                upper_limit=args.upper_limit,
                anisotropy=args.anisotropy if args.aniso else 1.0,
            )

            # Allow recomputation of features
            def redo_analysis_fun(
                label_image: npt.NDArray[Any],
                label1: npt.NDArray[Any],
                label2: npt.NDArray[Any],
                # fix values for loop variables
                label_fn: str = label_fn,
                spots_fn: str = spots_fn,
                lamina_fn: str = lamina_fn,
                summary_fn: str = summary_fn,
                spot_fn: str = spot_fn,
            ) -> tuple[
                npt.NDArray[Any] | None,
                npt.NDArray[Any] | None,
                npt.NDArray[Any] | None,
                pd.DataFrame | None,
                pd.DataFrame | None,
            ]:
                logger.info("Repeating analysis")
                # Manual editing may duplicate label IDs
                label_image, m = label(label_image, return_num=True)
                label1, m1 = label(label1, return_num=True)
                label2, m2 = label(label2, return_num=True)
                label_image = compact_mask(label_image, m=m)
                label1 = compact_mask(label1, m=m1)
                label2 = compact_mask(label2, m=m2)

                # Save new labels
                imwrite(label_fn, label_image, compression="zlib")
                imwrite(spots_fn, label1, compression="zlib")
                imwrite(lamina_fn, label2, compression="zlib")

                analysis_fun(label_image, label1, label2)

                # Reload analysis
                label_df = pd.read_csv(summary_fn)
                spot_df = pd.read_csv(spot_fn)

                return label_image, label1, label2, label_df, spot_df

            add_analysis_function(viewer, redo_analysis_fun)

            show_viewer(viewer)

    logger.info("Done")


if __name__ == "__main__":
    main()
