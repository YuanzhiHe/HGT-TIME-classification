"""Download and preprocess ST-bank data for multimodal pretraining.

ST-bank contains ~2.18M WSI-Visium ST paired patches across 32 organs,
113 studies, and 1007 samples. Data is hosted on Google Drive.

References:
    - Loki (OmiCLIP): Nature Methods 2025
    - GitHub: https://github.com/GuangyuWangLab2021/Loki
    - Data: https://drive.google.com/drive/folders/1J15cO-pXTwkTjRAR-v-_nQkqXNfcCNn3

Usage:
    python download_stbank.py --out-dir /path/to/stbank_data [--skip-images]
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
from pathlib import Path

GDRIVE_FOLDER = "https://drive.google.com/drive/folders/1J15cO-pXTwkTjRAR-v-_nQkqXNfcCNn3"

# Known file IDs (from the public Google Drive folder)
FILES = {
    "image.tar.gz": {
        "description": "H&E image patches (~1.17 GB, 2.18M patches at 20x20px native)",
        "size_gb": 1.17,
    },
    "text.csv": {
        "description": "Gene sentences paired with patch IDs (~967 MB, top-50 genes per spot)",
        "size_gb": 0.97,
    },
    "links_to_raw_data.xlsx": {
        "description": "Provenance spreadsheet with source paper DOIs",
        "size_gb": 0.0001,
    },
}


def download_with_gdown(out_dir: Path, skip_images: bool = False) -> None:
    """Download ST-bank from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        print("ERROR: gdown is required. Install with: pip install gdown", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ST-bank to {out_dir}")
    print(f"Source: {GDRIVE_FOLDER}\n")

    # Download entire folder
    gdown.download_folder(
        url=GDRIVE_FOLDER,
        output=str(out_dir),
        quiet=False,
        use_cookies=False,
    )

    # Extract image archive
    tar_path = out_dir / "image.tar.gz"
    if tar_path.exists() and not skip_images:
        print("\nExtracting image.tar.gz ...")
        img_dir = out_dir / "images"
        img_dir.mkdir(exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(path=img_dir)
        print(f"Extracted to {img_dir}")
    elif skip_images:
        print("Skipping image extraction (--skip-images)")

    print("\nDone. Files:")
    for f in sorted(out_dir.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024) if f.is_file() else 0
        print(f"  {f.name:40s} {size_mb:>10.1f} MB")


def validate_download(out_dir: Path) -> bool:
    """Basic validation of downloaded files."""
    text_csv = out_dir / "text.csv"
    if not text_csv.exists():
        print("WARNING: text.csv not found", file=sys.stderr)
        return False

    # Count lines
    n_lines = 0
    with open(text_csv, "r") as f:
        for _ in f:
            n_lines += 1
    print(f"text.csv: {n_lines:,} lines (expected ~2,185,572 including header)")

    # Check images dir
    img_dir = out_dir / "images"
    if img_dir.exists():
        n_imgs = sum(1 for _ in img_dir.rglob("*.png"))
        print(f"Image patches: {n_imgs:,} PNG files (expected ~2,185,571)")
    else:
        print("Images not extracted yet")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ST-bank for multimodal pretraining")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--skip-images", action="store_true", help="Skip extracting image archive")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing download")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.validate_only:
        validate_download(out_dir)
    else:
        download_with_gdown(out_dir, skip_images=args.skip_images)
        validate_download(out_dir)


if __name__ == "__main__":
    main()
