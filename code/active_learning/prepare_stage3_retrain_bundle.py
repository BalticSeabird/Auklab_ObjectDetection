#!/usr/bin/env python3
"""
Prepare retrain bundle from exported annotation data.

This helper normalizes an exported dataset into a single merge-ready folder with
an integrity manifest and duplicate checks by file hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


@dataclass
class BundleItem:
    source_image: str
    source_label: str
    dest_image: str
    dest_label: str
    split: str
    sha1: str


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def collect_pairs(export_root: Path) -> List[Tuple[Path, Path, str]]:
    pairs: List[Tuple[Path, Path, str]] = []

    for split in ("train", "valid", "test"):
        img_dir = export_root / split / "images"
        lbl_dir = export_root / split / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            continue

        for img in img_dir.iterdir():
            if not img.is_file() or img.suffix not in IMAGE_EXTS:
                continue
            label = lbl_dir / f"{img.stem}.txt"
            if label.exists():
                pairs.append((img, label, split))

    return pairs


def build_bundle(export_root: Path, out_root: Path, bundle_name: str) -> Dict:
    bundle_dir = out_root / bundle_name
    images_out = bundle_dir / "images" / "train"
    labels_out = bundle_dir / "labels" / "train"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(export_root)
    items: List[BundleItem] = []

    seen_hashes = set()
    duplicates = 0

    for img, lbl, split in pairs:
        digest = sha1_file(img)
        if digest in seen_hashes:
            duplicates += 1
            continue
        seen_hashes.add(digest)

        dest_stem = f"{split}_{img.stem}"
        dest_img = images_out / f"{dest_stem}{img.suffix.lower()}"
        dest_lbl = labels_out / f"{dest_stem}.txt"

        shutil.copy2(img, dest_img)
        shutil.copy2(lbl, dest_lbl)

        items.append(
            BundleItem(
                source_image=str(img),
                source_label=str(lbl),
                dest_image=str(dest_img),
                dest_label=str(dest_lbl),
                split=split,
                sha1=digest,
            )
        )

    manifest = {
        "created": datetime.now().isoformat(),
        "bundle_name": bundle_name,
        "source_export_root": str(export_root),
        "bundle_dir": str(bundle_dir),
        "total_pairs_discovered": len(pairs),
        "total_pairs_bundled": len(items),
        "duplicates_skipped": duplicates,
        "items": [asdict(i) for i in items],
    }

    manifest_path = bundle_dir / "merge_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "bundle_dir": str(bundle_dir),
        "manifest": str(manifest_path),
        "total_pairs_bundled": len(items),
        "duplicates_skipped": duplicates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a merge-ready retrain bundle from exported labels/images")
    parser.add_argument("--export-root", required=True, type=str, help="Export root containing train/valid/test images+labels")
    parser.add_argument("--output-root", default="data", type=str, help="Where to write bundle folder")
    parser.add_argument("--bundle-name", default="stage3_retrain_bundle", type=str, help="Bundle folder name")
    args = parser.parse_args()

    result = build_bundle(Path(args.export_root), Path(args.output_root), args.bundle_name)
    print(f"Bundle directory: {result['bundle_dir']}")
    print(f"Manifest: {result['manifest']}")
    print(f"Pairs bundled: {result['total_pairs_bundled']}")
    print(f"Duplicates skipped: {result['duplicates_skipped']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
