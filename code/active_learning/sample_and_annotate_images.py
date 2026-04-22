#!/usr/bin/env python3
"""
Sample existing images from multiple source directories and annotate them with YOLO model.

This script:
1. Discovers and samples existing images (user-defined n) from several source folders
2. Copies sampled images to a new folder organized by category
3. Annotates them using an existing YOLO model
4. Saves annotations in YOLO txt format (ready for Roboflow upload)

Designed for active learning workflows where you have existing image collections
and want to prepare a batch for annotation validation.

Usage:
    python sample_and_annotate_images.py \\
        --image-roots data/validation_images data/archive_images \\
        --output-dir data/validation_batch \\
        --samples 100 \\
        --model models/best.pt

    # Prioritize specific stations (includes all videos from those stations)
    python sample_and_annotate_images.py \\
        --image-roots data/videos \\
        --samples 150 \\
        --preferred-stations FAR3_SCALE 1064 \\
        --model models/best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    print("WARNING: ultralytics not installed. Please install with: pip install ultralytics torch")
    HAS_ULTRALYTICS = False

try:
    from tqdm import tqdm
except ImportError:
    # Simple fallback if tqdm not available
    def tqdm(iterable, desc="Progress"):
        total = len(iterable)
        for i, item in enumerate(iterable, 1):
            print(f"\r{desc}: {i}/{total}", end='', flush=True)
            yield item
        print()  # New line after completion


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def _extract_video_name(path: Path) -> Optional[str]:
    """Extract video/folder name from image path (parent directory name)."""
    # Get the parent directory name (the folder containing the image)
    parent_name = path.parent.name
    return parent_name if parent_name else None


def _extract_station_from_video(video_name: str) -> Optional[str]:
    """Extract station name from video name (e.g., 'FAR3_SCALE' from 'FAR3_SCALE_2023-05-24_12_40')."""
    # Match everything before a date pattern (YYYY-MM-DD or YYYY_MM_DD)
    match = re.match(r"(.+?)_\d{4}[-_]\d{2}[-_]\d{2}", video_name)
    return match.group(1) if match else None


@dataclass
class ImageInfo:
    path: Path
    category: str  # e.g., 'validation', 'archive'
    video_name: Optional[str] = None  # e.g., 'FAR3_SCALE_2023-05-24_12_40'


def discover_images(image_roots: List[Path], category_names: Optional[List[str]] = None) -> List[ImageInfo]:
    """Discover all image files in root directories and their subdirectories."""
    images: List[ImageInfo] = []

    for i, root in enumerate(image_roots):
        if not root.exists():
            print(f"Warning: Image root does not exist: {root}")
            continue

        # Use provided category name or derive from directory
        category = category_names[i] if category_names and i < len(category_names) else root.name

        for ext in IMAGE_EXTS:
            for img_path in root.rglob(f"*{ext}"):
                if img_path.is_file():
                    video_name = _extract_video_name(img_path)
                    images.append(ImageInfo(
                        path=img_path,
                        category=category,
                        video_name=video_name
                    ))

    return sorted(images, key=lambda img: str(img.path))


def sample_images(
    images: List[ImageInfo],
    num_samples: int,
    preferred_stations: Optional[List[str]] = None,
    seed: int = 42,
) -> List[ImageInfo]:
    """Randomly sample images from the discovered list, evenly distributed across categories/folders."""
    if not images:
        return []

    random.seed(seed)

    # Filter to preferred stations if specified
    if preferred_stations:
        preferred_stations_set = set(preferred_stations)
        filtered = []

        for img in images:
            if img.video_name:
                station = _extract_station_from_video(img.video_name)
                if station in preferred_stations_set:
                    filtered.append(img)

        if not filtered:
            print("Warning: No images found from specified stations. Using all images.")
            filtered = images
        elif num_samples > 0 and len(filtered) < num_samples:
            print(f"Warning: Only {len(filtered)} images found from specified stations (requested {num_samples})")
    else:
        filtered = images

    # Group images by category
    categories = {}
    for img in filtered:
        if img.category not in categories:
            categories[img.category] = []
        categories[img.category].append(img)

    # Shuffle within each category
    for cat in categories:
        random.shuffle(categories[cat])

    # Distribute samples evenly across categories
    num_categories = len(categories)
    if num_categories == 0:
        return []

    if num_samples <= 0:
        # Return all images
        result = []
        for cat in sorted(categories.keys()):
            result.extend(categories[cat])
        return result

    samples_per_category = max(1, num_samples // num_categories)
    remainder = num_samples % num_categories

    result = []
    cat_list = sorted(categories.keys())

    # Sample from each category
    for i, cat in enumerate(cat_list):
        # Distribute remainder images across first categories
        target_count = samples_per_category + (1 if i < remainder else 0)
        available = categories[cat]
        sampled = available[:min(target_count, len(available))]
        result.extend(sampled)

    return result


def copy_images(
    sampled_images: List[ImageInfo],
    output_dir: Path,
) -> List[dict]:
    """Copy sampled images to output directory, organized by category, with video name in filename."""
    output_dir.mkdir(parents=True, exist_ok=True)

    copied: List[dict] = []

    for img_info in tqdm(sampled_images, desc="Copying images"):
        category_dir = output_dir / img_info.category
        category_dir.mkdir(parents=True, exist_ok=True)

        # Add video name to filename if available
        if img_info.video_name:
            stem = img_info.path.stem
            suffix = img_info.path.suffix
            new_filename = f"{img_info.video_name}_{stem}{suffix}"
        else:
            new_filename = img_info.path.name

        dest_path = category_dir / new_filename

        try:
            shutil.copy2(img_info.path, dest_path)
            copied.append({
                "dest_path": str(dest_path),
                "source_path": str(img_info.path),
                "category": img_info.category,
                "filename": new_filename,
                "video_name": img_info.video_name,
            })
        except Exception as e:
            print(f"Failed to copy {img_info.path}: {e}")

    return copied


def annotate_images(
    output_dir: Path,
    model_path: Path,
    confidence_threshold: float = 0.3,
) -> dict:
    """
    Run YOLO inference on copied images and save annotations.

    Returns:
        Statistics about annotations
    """
    if not HAS_ULTRALYTICS:
        raise ImportError("ultralytics package is required. Install with: pip install ultralytics torch")

    print(f"\nLoading model: {model_path}")
    model = YOLO(str(model_path))
    class_names = model.names
    print(f"Model classes: {class_names}")

    # Create annotations directory
    annotations_dir = output_dir / "annotations"
    yolo_dir = annotations_dir / "yolo"
    json_dir = annotations_dir / "json"
    yolo_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    # Find all image files
    image_files = []
    for ext in IMAGE_EXTS:
        image_files.extend(output_dir.rglob(f"*{ext}"))

    # Exclude already-processed annotations directory
    image_files = [f for f in image_files if "annotations" not in f.parts]

    print(f"\nFound {len(image_files)} images to annotate")
    print(f"Confidence threshold: {confidence_threshold}\n")

    # Statistics
    stats = {
        'total_images': len(image_files),
        'images_with_detections': 0,
        'total_detections': 0,
        'detections_by_class': {name: 0 for name in class_names.values()},
        'images_processed': 0,
        'images_failed': 0,
    }

    # Process each image
    for img_path in tqdm(image_files, desc="Annotating images"):
        try:
            # Run inference
            results = model(
                str(img_path),
                conf=confidence_threshold,
                verbose=False
            )[0]

            # Read image to get dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                stats['images_failed'] += 1
                continue

            img_height, img_width = img.shape[:2]

            # Convert detections to YOLO format
            detections = []
            if len(results.boxes) > 0:
                stats['images_with_detections'] += 1

                for box in results.boxes:
                    # Get box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    # Convert to YOLO format (normalized center_x, center_y, width, height)
                    center_x = ((x1 + x2) / 2) / img_width
                    center_y = ((y1 + y2) / 2) / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height

                    # Get class and confidence
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())

                    detections.append({
                        'class_id': cls,
                        'class_name': class_names[cls],
                        'confidence': conf,
                        'bbox_normalized': {
                            'center_x': float(center_x),
                            'center_y': float(center_y),
                            'width': float(width),
                            'height': float(height)
                        },
                        'bbox_pixels': {
                            'x1': float(x1),
                            'y1': float(y1),
                            'x2': float(x2),
                            'y2': float(y2)
                        }
                    })

                    stats['total_detections'] += 1
                    stats['detections_by_class'][class_names[cls]] += 1

            # Save annotations
            _save_annotations(img_path, detections, yolo_dir, json_dir, img_width, img_height, model_path)
            stats['images_processed'] += 1

        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            stats['images_failed'] += 1

    return stats


def _save_annotations(img_path, detections, yolo_dir, json_dir, img_width, img_height, model_path):
    """Save annotations in YOLO and JSON formats."""
    img_name = img_path.stem

    # YOLO format (.txt file)
    yolo_path = yolo_dir / f"{img_name}.txt"
    with open(yolo_path, 'w') as f:
        for det in detections:
            bbox = det['bbox_normalized']
            f.write(f"{det['class_id']} {bbox['center_x']:.6f} {bbox['center_y']:.6f} "
                   f"{bbox['width']:.6f} {bbox['height']:.6f}\n")

    # JSON format (more detailed, includes confidence)
    json_path = json_dir / f"{img_name}.json"
    annotation = {
        'image': img_path.name,
        'image_width': img_width,
        'image_height': img_height,
        'detections': detections,
        'model': str(model_path.name),
    }

    with open(json_path, 'w') as f:
        json.dump(annotation, f, indent=2)


def write_outputs(
    output_dir: Path,
    copied_images: List[dict],
    stats: dict,
    num_samples: int,
) -> None:
    """Write manifest and statistics files."""
    manifest = {
        "created": datetime.now().isoformat(),
        "samples_requested": num_samples,
        "samples_copied": len(copied_images),
        "annotation_stats": stats,
        "images": copied_images,
    }

    manifest_path = output_dir / "sampling_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    csv_path = output_dir / "images_list.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dest_path", "source_path", "category", "filename", "video_name"],
        )
        writer.writeheader()
        for row in copied_images:
            writer.writerow(row)

    print(f"\nManifest: {manifest_path}")
    print(f"CSV list: {csv_path}")


def print_summary(stats: dict) -> None:
    """Print summary statistics."""
    print("\n" + "="*70)
    print("ANNOTATION SUMMARY")
    print("="*70)
    print(f"Total images: {stats['total_images']}")
    print(f"Images processed: {stats['images_processed']}")
    print(f"Images failed: {stats['images_failed']}")
    print(f"Images with detections: {stats['images_with_detections']} "
          f"({100*stats['images_with_detections']/stats['total_images']:.1f}%)")
    print(f"Total detections: {stats['total_detections']}")
    if stats['total_images'] > 0:
        print(f"Average detections per image: "
              f"{stats['total_detections']/stats['total_images']:.1f}")
    print()
    print("Detections by class:")
    for class_name, count in stats['detections_by_class'].items():
        if count > 0:
            print(f"  {class_name}: {count}")
    print("="*70)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample existing images and annotate them with YOLO model"
    )
    parser.add_argument(
        "--image-roots",
        type=str,
        nargs="+",
        required=True,
        help="One or more root directories containing existing images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/validation_batch",
        help="Output directory for copied and annotated images",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of images to sample (<=0 means all discovered images). Samples are evenly distributed across folders.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/best.pt",
        help="Path to YOLO model (.pt file)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Confidence threshold for detections (lower = more detections)",
    )
    parser.add_argument(
        "--category-names",
        type=str,
        nargs="*",
        default=None,
        help="Custom category names for each image root (defaults to directory names)",
    )
    parser.add_argument(
        "--preferred-stations",
        type=str,
        nargs="*",
        default=None,
        help="Station names to prioritize sampling from (e.g., FAR3_SCALE 1064). Includes all videos from these stations.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("SAMPLE AND ANNOTATE IMAGES")
    print("="*70)
    print()

    # Validate inputs
    image_roots = [Path(p) for p in args.image_roots]
    output_dir = Path(args.output_dir)
    model_path = Path(args.model)

    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        return 1

    # Discover images
    images = discover_images(image_roots, args.category_names)
    print(f"Discovered images: {len(images)}")

    if not images:
        print("Error: No images found in provided directories")
        return 1

    # Show statistics
    categories_found = {}
    videos_found = set()
    for img in images:
        categories_found[img.category] = categories_found.get(img.category, 0) + 1
        if img.video_name:
            videos_found.add(img.video_name)

    print("Images by category:")
    for cat, count in sorted(categories_found.items()):
        print(f"  {cat}: {count}")

    if videos_found:
        print(f"Videos found: {sorted(videos_found)}")

    if args.preferred_stations:
        print(f"Prioritizing stations: {args.preferred_stations}")
    print()

    # Sample images
    sampled = sample_images(
        images=images,
        num_samples=args.samples,
        preferred_stations=args.preferred_stations,
        seed=args.random_seed,
    )
    print(f"Sampled images: {len(sampled)}")
    print()

    # Copy images
    print("Copying images...")
    copied = copy_images(sampled, output_dir)
    print(f"Copied images: {len(copied)}\n")

    # Annotate images
    print("Annotating images...")
    stats = annotate_images(
        output_dir=output_dir,
        model_path=model_path,
        confidence_threshold=args.confidence,
    )

    # Write outputs
    write_outputs(output_dir, copied, stats, args.samples)
    print_summary(stats)

    print("\nNEXT STEPS:")
    print("="*70)
    print("1. Review the sampled and annotated images:")
    print(f"   - Images: {output_dir}")
    print(f"   - Annotations: {output_dir}/annotations/yolo/")
    print()
    print("2. Correct any annotation errors using your annotation tool")
    print()
    print("3. Upload to Roboflow for validation:")
    print(f"   python code/active_learning/upload_to_roboflow.py \\")
    print(f"       --frames-dir {output_dir} \\")
    print(f"       --use-annotations \\")
    print(f"       --api-key YOUR_API_KEY")
    print("="*70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
