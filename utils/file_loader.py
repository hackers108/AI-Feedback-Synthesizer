"""
Utility functions for loading datasets.
"""

from pathlib import Path
import pandas as pd


def load_feedback(file_path):
    """
    Load a feedback CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Feedback file not found: {file_path}"
        )

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise RuntimeError(
            f"Unable to read {file_path}: {e}"
        )

    return df