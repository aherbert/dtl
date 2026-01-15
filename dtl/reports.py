"""Utility functions."""

import glob
import logging
import os
from typing import Any

import pandas as pd


def load_data(
    files: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find the data in the input file or directory paths.

    Collects the summary.csv and spot.csv files.

    Args:
        files: CSV file or directory paths.

    Returns:
        summary data, spot data

    Raises:
        RuntimeError: if a path is not a file or directory
    """
    summary_data = []
    spot_data = []
    for fn in files:
        if os.path.isfile(fn):
            df1, df2 = _is_data(fn)
            if df1 is not None:
                summary_data.append(df1)
            if df2 is not None:
                spot_data.append(df2)
        elif os.path.isdir(fn):
            for file in glob.glob(os.path.join(fn, "*.csv")):
                file = os.path.join(fn, file)
                df1, df2 = _is_data(file)
                if df1 is not None:
                    summary_data.append(df1)
                if df2 is not None:
                    spot_data.append(df2)
        else:
            raise RuntimeError("Not a file or directory: " + fn)

    summary_df = pd.concat(summary_data) if summary_data else pd.DataFrame()
    spot_df = pd.concat(spot_data) if spot_data else pd.DataFrame()
    return summary_df, spot_df


def _is_data(fn: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load summary or spot data from the file."""
    if fn.endswith("summary.csv"):
        return pd.read_csv(fn), None
    if fn.endswith("spots.csv"):
        return None, pd.read_csv(fn)
    return None, None


def number_of_reports() -> int:
    """Get the number of supported reports."""
    return 2


def create_report(
    summary_df: pd.DataFrame,
    spot_df: pd.DataFrame,
    report_id: int,
    out: str | None = None,
    **kwargs: dict[str, Any],
) -> tuple[pd.DataFrame, str]:
    """Create and optionally save the report.

    Args:
        summary_df: Summary data.
        spot_df: Spot data.
        report_id: Report identifier.
        out: Output directory.
        kwargs: Named arguments.

    Returns:
        report data, report title
    """
    data = pd.DataFrame()
    title = ""
    match report_id:
        case 1:
            title = "Mean count"
            keys = ["edge", "internal", "total"]
            data = pd.DataFrame(
                data={
                    "type": keys,
                    "mean": [summary_df[k].mean() for k in keys],
                }
            )
        case 2:
            title = "Mean distance"
            data = (
                spot_df[["type", "distance"]]
                .groupby(["type"])
                .mean()
                .reset_index("type")
            )
        case _:
            logging.getLogger(__name__).warning(
                "Unknown report number: %d", report_id
            )
    if out and not data.empty:
        data.to_csv(os.path.join(out, f"report{report_id}.csv"), index=False)
    return data, title
