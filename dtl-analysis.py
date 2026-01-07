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

    # group = parser.add_argument_group("Threshold Options")
    # _ = group.add_argument(
    #     "--sigma",
    #     default=1.5,
    #     type=float,
    #     help="Gaussian smoothing filter standard deviation (default: %(default)s)",
    # )
    # _ = group.add_argument(
    #     "--sigma2",
    #     default=30,
    #     type=float,
    #     help="Background Gaussian smoothing filter standard deviation (default: %(default)s)",
    # )
    # _ = group.add_argument(
    #     "--method",
    #     default="mean_plus_std_q",
    #     choices=["mean_plus_std", "mean_plus_std_q", "otsu", "yen", "minimum"],
    #     help="Thresholding method (default: %(default)s)",
    # )
    # _ = group.add_argument(
    #     "--std",
    #     default=7,
    #     type=float,
    #     help="Std.dev above the mean (default: %(default)s)",
    # )
    # _ = group.add_argument(
    #     "--quantile",
    #     default=0.75,
    #     type=float,
    #     help="Quantile for lowest value used in mean_plus_std_q (default: %(default)s)",
    # )
    # _ = group.add_argument(
    #     "--fill-holes",
    #     default=2,
    #     type=int,
    #     help="Remove contiguous holes smaller than the specified size (pixels) (default: %(default)s)",
    # )
    # _ = group.add_argument(
    #     "--min-spot-size",
    #     default=4,
    #     type=int,
    #     help="Minimum spot size (pixels) (default: %(default)s)",
    # )

    group = parser.add_argument_group("View Options")
    _ = group.add_argument(
        "--view",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Show results in graphical viewer",
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

    import os

    # TODO: try pylibics as BioIO is complex to install
    import bioio_bioformats
    import numpy as np
    from bioio import BioImage
    from tifffile import imread, imwrite

    from dtl.utils import (
        find_images,
    )

    # std = args.std

    images = find_images(args.image)
    for image_fn in images:
        logger.info("Processing %s", image_fn)
        base, suffix = os.path.splitext(image_fn)

        img = BioImage(image_fn, reader=bioio_bioformats.Reader)
        if img.dims.T != 1:
            raise RuntimeError("Expected CZYX image: " + str(img.dims))
        # CZYX
        image = img.data.squeeze(axis=0)

        stage = args.repeat if args.repeat else 10

        label_fn = f"{base}.objects{suffix}"
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
            label_image = imread(label_fn)
            n_objects = np.max(label_image)
        logger.info("Identified %d objects: %s", n_objects, label_fn)

        # # Spot identification
        # if args.method == "mean_plus_std_q" and std == args.std:
        #     # make std shift equivalent to get the same thresholding level if a normal distribution is truncated
        #     import scipy.stats

        #     m, v = scipy.stats.truncnorm(
        #         -10, scipy.stats.norm.ppf(args.quantile)
        #     ).stats("mv")
        #     std = (std - m) / np.sqrt(v)
        #     logger.info(
        #         "Adjusted threshold %s to %.3f using normal distribution truncated at cdf=%s",
        #         args.std,
        #         std,
        #         args.quantile,
        #     )
        # fun = threshold_method(args.method, std=std, q=args.quantile)
        # filter_fun = filter_method(args.sigma, args.sigma2)

        # spot1_fn = f"{base}.spot1{suffix}"
        # im1 = image[args.spot_ch1]
        # if stage <= 2 or not os.path.exists(spot1_fn):
        #     label1 = object_threshold(
        #         filter_fun(im1),
        #         label_image,
        #         fun,
        #         fill_holes=args.fill_holes,
        #         min_size=args.min_spot_size,
        #     )
        #     imwrite(spot1_fn, label1, compression="zlib")
        # else:
        #     label1 = imread(spot1_fn)
        # logger.info(
        #     "Identified %d spots in channel %d: %s",
        #     np.max(label1),
        #     args.spot_ch1,
        #     spot1_fn,
        # )

        # spot2_fn = f"{base}.spot2{suffix}"
        # im2 = image[args.spot_ch2]
        # if stage <= 2 or not os.path.exists(spot2_fn):
        #     label2 = object_threshold(
        #         filter_fun(im2),
        #         label_image,
        #         fun,
        #         fill_holes=args.fill_holes,
        #         min_size=args.min_spot_size,
        #     )
        #     imwrite(spot2_fn, label2, compression="zlib")
        # else:
        #     label2 = imread(spot2_fn)
        # logger.info(
        #     "Identified %d spots in channel %d: %s",
        #     np.max(label2),
        #     args.spot_ch2,
        #     spot2_fn,
        # )

        # # Analysis cannot be loaded from previous results, just skip
        # spot_fn = f"{base}.spots.csv"
        # summary_fn = f"{base}.summary.csv"

        # # Create an anlysis function to allow this to be repeated.
        # # Settings used: distance, size, min_size, neighbour_distance.
        # # These could be added as input for the function and updated through the GUI.
        # def analysis_fun(
        #     label_image: npt.NDArray[Any],
        #     label1: npt.NDArray[Any],
        #     label2: npt.NDArray[Any],
        #     # fix values for loop variables
        #     image: npt.NDArray[Any] = image,
        #     im1: npt.NDArray[Any] = im1,
        #     im2: npt.NDArray[Any] = im2,
        #     spot_fn: str = spot_fn,
        #     summary_fn: str = summary_fn,
        # ) -> None:
        #     # find micro-nuclei and bleb parents
        #     objects = find_objects(label_image)
        #     data = find_micronuclei(
        #         label_image,
        #         objects=objects,
        #         distance=args.distance,
        #         size=args.size,
        #         min_size=args.min_size,
        #     )

        #     # Create groups. This collates blebs with their parent.
        #     groups = collate_groups(data)
        #     class_names = classify_objects(data, args.size, args.distance)
        #     logger.info(
        #         "Classified objects: %s", dict(Counter(class_names.values()))
        #     )

        #     results = spot_analysis(
        #         label_image,
        #         objects,
        #         groups,
        #         im1,
        #         label1,
        #         im2,
        #         label2,
        #         neighbour_distance=args.neighbour_distance,
        #     )

        #     formatted = format_spot_results(
        #         results, class_names=class_names, scale=args.scale
        #     )
        #     logger.info("Saving spot results: %s", spot_fn)
        #     save_csv(spot_fn, formatted)

        #     o_data = analyse_objects(
        #         image[args.object_ch], label_image, objects
        #     )
        #     # Collate to ID: (size, intensity, cx, cy)
        #     object_data = {
        #         x[0]: (x[1],) + y for x, y in zip(objects, o_data, strict=True)
        #     }

        #     summary = spot_summary(results, groups)
        #     formatted2 = format_summary_results(
        #         summary, class_names=class_names, object_data=object_data
        #     )
        #     logger.info("Saving summary results: %s", summary_fn)
        #     save_csv(summary_fn, formatted2)

        # if stage <= 3 or not (
        #     os.path.exists(spot_fn) and os.path.exists(summary_fn)
        # ):
        #     analysis_fun(label_image, label1, label2)
        # else:
        #     logger.info("Existing spot results: %s", spot_fn)
        #     logger.info("Existing summary results: %s", summary_fn)

        # # Save settings if all OK
        # fn = f"{base}.settings.json"
        # logger.info("Saving settings: %s", fn)
        # with open(fn, "w") as f:
        #     d = vars(args)
        #     d["image"] = image_fn
        #     json.dump(d, f, indent=2)

        if args.view:
            logger.info("Launching viewer")

            from dtl.gui import (
                create_viewer,
                show_viewer,
            )

            # label_df = pd.read_csv(summary_fn)
            # spot_df = pd.read_csv(spot_fn)

            visible_channels = (
                args.visible_channels
                if args.visible_channels
                else [args.spot_ch, args.lamina_ch]
            )
            # label_image = np.zeros(image[0].shape, dtype=np.uint8)
            label1 = np.zeros(label_image.shape, dtype=np.uint8)
            label2 = np.zeros(label_image.shape, dtype=np.uint8)
            viewer = create_viewer(
                image,
                label_image,
                label1,
                label2,
                channel_names=args.channel_names,
                visible_channels=visible_channels,
                # label_df=label_df,
                # spot_df=spot_df,
                upper_limit=args.upper_limit,
            )

            # # Allow recomputation of features
            # def redo_analysis_fun(
            #     label_image: npt.NDArray[Any],
            #     label1: npt.NDArray[Any],
            #     label2: npt.NDArray[Any],
            #     # fix values for loop variables
            #     label_fn: str = label_fn,
            #     spot1_fn: str = spot1_fn,
            #     spot2_fn: str = spot2_fn,
            #     summary_fn: str = summary_fn,
            #     spot_fn: str = spot_fn,
            # ) -> tuple[
            #     npt.NDArray[Any] | None,
            #     npt.NDArray[Any] | None,
            #     npt.NDArray[Any] | None,
            #     pd.DataFrame | None,
            #     pd.DataFrame | None,
            # ]:
            #     logger.info("Repeating analysis")
            #     # Manual editing may duplicate label IDs
            #     label_image, m = label(label_image, return_num=True)
            #     label1, m1 = label(label1, return_num=True)
            #     label2, m2 = label(label2, return_num=True)
            #     label_image = compact_mask(label_image, m=m)
            #     label1 = compact_mask(label1, m=m1)
            #     label2 = compact_mask(label2, m=m2)

            #     # Save new labels
            #     imwrite(label_fn, label_image, compression="zlib")
            #     imwrite(spot1_fn, label1, compression="zlib")
            #     imwrite(spot2_fn, label2, compression="zlib")

            #     analysis_fun(label_image, label1, label2)

            #     # Reload analysis
            #     label_df = pd.read_csv(summary_fn)
            #     spot_df = pd.read_csv(spot_fn)

            #     return label_image, label1, label2, label_df, spot_df

            # add_analysis_function(viewer, redo_analysis_fun)

            show_viewer(viewer)

    logger.info("Done")


if __name__ == "__main__":
    main()
