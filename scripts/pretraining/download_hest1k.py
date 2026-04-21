"""Download and preprocess HEST-1k data for multimodal pretraining.

HEST-1k provides ~1,276 paired WSI + ST profiles across 26 organs and
multiple platforms (Visium, Visium HD, Xenium), hosted on HuggingFace.

References:
    - Jaume et al. "HEST-1k" NeurIPS 2024 Spotlight
    - GitHub: https://github.com/mahmoodlab/HEST
    - HuggingFace: https://huggingface.co/datasets/MahmoodLab/hest

Usage:
    python download_hest1k.py --out-dir /path/to/hest_data [--organ Breast] [--cancer-only]
    python download_hest1k.py --out-dir /path/to/hest_data --all
    python download_hest1k.py --out-dir /path/to/hest_data --ids TENX95 TENX96
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd

REPO_ID = "MahmoodLab/hest"
METADATA_CSV = "hf://datasets/MahmoodLab/hest/HEST_v1_3_0.csv"


def get_metadata(cache_dir: Path | None = None) -> pd.DataFrame:
    """Load HEST-1k metadata from HuggingFace."""
    csv_path = cache_dir / "HEST_v1_3_0.csv" if cache_dir else None
    if csv_path and csv_path.exists():
        return pd.read_csv(csv_path)

    print("Downloading HEST-1k metadata...")
    df = pd.read_csv(METADATA_CSV)
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
    return df


def filter_samples(
    meta_df: pd.DataFrame,
    organ: str | None = None,
    cancer_only: bool = False,
    platform: str | None = None,
    ids: list[str] | None = None,
) -> pd.DataFrame:
    """Filter HEST-1k metadata by organ, cancer status, platform, or IDs."""
    df = meta_df.copy()

    if ids:
        df = df[df["id"].isin(ids)]
        print(f"  Filtered to {len(df)} samples by ID")
        return df

    if organ:
        df = df[df["organ"].str.lower() == organ.lower()]
        print(f"  Filtered to {len(df)} samples for organ={organ}")

    if cancer_only:
        # Samples with oncotree_code are cancer; normal tissue has NaN
        df = df[df["oncotree_code"].notna()]
        print(f"  Filtered to {len(df)} cancer samples")

    if platform:
        platform_lower = platform.lower()
        df = df[df["st_technology"].str.lower().str.contains(platform_lower, na=False)]
        print(f"  Filtered to {len(df)} samples for platform containing '{platform}'")

    return df


def download_samples(
    sample_ids: list[str],
    out_dir: Path,
    include_patches: bool = True,
    include_wsi: bool = False,
) -> None:
    """Download specific HEST-1k samples from HuggingFace."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is required. Install with: pip install huggingface-hub",
              file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Build download patterns
    patterns = []
    for sid in sample_ids:
        # Always download ST data and metadata
        patterns.append(f"st/{sid}*")
        patterns.append(f"metadata/{sid}*")
        patterns.append(f"spatial_plots/{sid}*")
        if include_patches:
            patterns.append(f"patches/{sid}*")
        if include_wsi:
            patterns.append(f"wsis/{sid}*")

    print(f"Downloading {len(sample_ids)} samples ({len(patterns)} patterns)...")
    snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=patterns,
        repo_type="dataset",
        local_dir=str(out_dir),
    )

    # Unzip CellViT segmentation if present
    seg_dir = out_dir / "cellvit_seg"
    if seg_dir.exists():
        for zf in seg_dir.glob("*.zip"):
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(seg_dir)

    print(f"Download complete: {out_dir}")


def print_summary(meta_df: pd.DataFrame) -> None:
    """Print summary of selected samples."""
    print(f"\n{'='*60}")
    print(f"HEST-1k selection summary")
    print(f"{'='*60}")
    print(f"  Total samples: {len(meta_df)}")
    if "organ" in meta_df.columns:
        print(f"  Organs: {meta_df['organ'].nunique()}")
        for organ, count in meta_df["organ"].value_counts().head(10).items():
            print(f"    {organ}: {count}")
    if "st_technology" in meta_df.columns:
        print(f"  Platforms:")
        for tech, count in meta_df["st_technology"].value_counts().items():
            print(f"    {tech}: {count}")
    if "oncotree_code" in meta_df.columns:
        n_cancer = meta_df["oncotree_code"].notna().sum()
        print(f"  Cancer samples: {n_cancer}")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HEST-1k for multimodal pretraining")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")

    # Filtering options
    parser.add_argument("--all", action="store_true", help="Download all samples (~1 TB)")
    parser.add_argument("--organ", type=str, default=None, help="Filter by organ (e.g., Breast)")
    parser.add_argument("--cancer-only", action="store_true", help="Only cancer samples")
    parser.add_argument("--platform", type=str, default=None,
                        help="Filter by ST platform (e.g., Visium, Xenium)")
    parser.add_argument("--ids", nargs="+", default=None, help="Download specific sample IDs")

    # Content options
    parser.add_argument("--include-wsi", action="store_true",
                        help="Also download full WSIs (very large)")
    parser.add_argument("--no-patches", action="store_true",
                        help="Skip pre-extracted 224x224 patches")

    # Info only
    parser.add_argument("--list-only", action="store_true",
                        help="Only list matching samples without downloading")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    # Load metadata
    meta_df = get_metadata(cache_dir=out_dir / ".cache")

    # Filter
    if not args.all:
        meta_df = filter_samples(
            meta_df,
            organ=args.organ,
            cancer_only=args.cancer_only,
            platform=args.platform,
            ids=args.ids,
        )

    if len(meta_df) == 0:
        print("No samples match the specified filters.", file=sys.stderr)
        sys.exit(1)

    print_summary(meta_df)

    if args.list_only:
        for _, row in meta_df.iterrows():
            print(f"  {row['id']:12s}  {row.get('organ', 'N/A'):15s}  "
                  f"{row.get('st_technology', 'N/A'):15s}  {row.get('oncotree_code', 'normal')}")
        return

    # Download
    sample_ids = meta_df["id"].tolist()
    download_samples(
        sample_ids,
        out_dir,
        include_patches=not args.no_patches,
        include_wsi=args.include_wsi,
    )


if __name__ == "__main__":
    main()
