"""Data loading and verification utilities."""

from pathlib import Path
import hashlib
import pandas as pd


def verify_file_hash(file_path: Path, expected_hash: str) -> bool:
    """Compute SHA-256 hash of a file and compare it to expected hash."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest().lower() == expected_hash.lower()


def load_raw_data(data_dir: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load VM CPU readings and metadata from compressed CSV files.

    Args:
        data_dir: Path to directory containing the dataset files.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (cpu_df, metadata_df)
    """
    data_path = Path(data_dir)
    cpu_file = data_path / "vm_cpu_readings.csv.gz"
    meta_file = data_path / "vm_metadata.csv.gz"

    if not cpu_file.exists():
        raise FileNotFoundError(f"Missing file: {cpu_file}")
    if not meta_file.exists():
        raise FileNotFoundError(f"Missing file: {meta_file}")

    print(f"Loading readings from: {cpu_file.name} ...")
    cpu_df = pd.read_csv(cpu_file)

    print(f"Loading metadata from: {meta_file.name} ...")
    meta_df = pd.read_csv(meta_file)

    return cpu_df, meta_df


if __name__ == "__main__":
    # Test script directly
    DATA_DIR = Path("challenge_data")
    try:
        cpu_readings, metadata = load_raw_data(DATA_DIR)
        print("\n--- Ingestion Successful ---")
        print(f"Readings shape: {cpu_readings.shape}")
        print(f"Metadata shape: {metadata.shape}")
        print("\nReadings head:")
        print(cpu_readings.head(3))
        print("\nMetadata head:")
        print(metadata.head(3))
    except Exception as e:
        print(f"Error loading data: {e}")