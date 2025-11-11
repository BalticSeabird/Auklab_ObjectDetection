#!/usr/bin/env python3
"""
Pre-annotate extracted frames using the existing YOLO model.

This provides a starting point for manual annotation - much faster than
annotating from scratch. Annotators can review and correct the predictions.

Usage:
    python pre_annotate_frames.py
    python pre_annotate_frames.py --confidence 0.25  # Use lower threshold
    python pre_annotate_frames.py --model path/to/model.pt
"""

import argparse
from pathlib import Path
import json
import cv2

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    print("WARNING: ultralytics not installed. Please install with: pip install ultralytics")
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


class FramePreAnnotator:
    def __init__(self, model_path, confidence_threshold=0.3):
        """
        Initialize pre-annotator.
        
        Args:
            model_path: Path to YOLO model (.pt file)
            confidence_threshold: Minimum confidence for detections (lower = more detections)
        """
        if not HAS_ULTRALYTICS:
            raise ImportError("ultralytics package is required but not installed. "
                            "Please install with: pip install ultralytics torch")
        
        self.model_path = Path(model_path)
        self.confidence = confidence_threshold
        
        print(f"Loading model: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        
        # Get class names from model
        self.class_names = self.model.names
        print(f"Model classes: {self.class_names}")
        
    def pre_annotate_frames(self, frames_dir, output_dir=None):
        """
        Run inference on all extracted frames and save annotations.
        
        Args:
            frames_dir: Directory containing extracted frames
            output_dir: Where to save annotations (defaults to frames_dir/annotations)
        """
        frames_dir = Path(frames_dir)
        
        # Set output directory
        if output_dir is None:
            output_dir = frames_dir / "annotations"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            image_files.extend(frames_dir.rglob(f"*{ext}"))
        
        print(f"\nFound {len(image_files)} images to pre-annotate")
        print(f"Output directory: {output_dir}")
        print(f"Confidence threshold: {self.confidence}")
        print()
        
        # Statistics
        stats = {
            'total_images': len(image_files),
            'images_with_detections': 0,
            'total_detections': 0,
            'detections_by_class': {name: 0 for name in self.class_names.values()}
        }
        
        # Process each image
        for img_path in tqdm(image_files, desc="Pre-annotating frames"):
            # Run inference
            results = self.model(
                str(img_path),
                conf=self.confidence,
                verbose=False
            )[0]
            
            # Get image dimensions
            img = cv2.imread(str(img_path))
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
                        'class_name': self.class_names[cls],
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
                    stats['detections_by_class'][self.class_names[cls]] += 1
            
            # Save annotations
            self._save_annotations(img_path, detections, output_dir, img_width, img_height)
        
        # Print summary
        self._print_summary(stats)
        
        # Save summary to file
        summary_path = output_dir / "pre_annotation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n✅ Pre-annotation complete!")
        print(f"✅ Annotations saved to: {output_dir}")
        print(f"✅ Summary saved to: {summary_path}")
        
    def _save_annotations(self, img_path, detections, output_dir, img_width, img_height):
        """Save annotations in multiple formats for compatibility."""
        
        # Create output paths
        img_name = img_path.stem
        
        # 1. YOLO format (.txt file)
        yolo_dir = output_dir / "yolo"
        yolo_dir.mkdir(exist_ok=True)
        yolo_path = yolo_dir / f"{img_name}.txt"
        
        with open(yolo_path, 'w') as f:
            for det in detections:
                bbox = det['bbox_normalized']
                f.write(f"{det['class_id']} {bbox['center_x']:.6f} {bbox['center_y']:.6f} "
                       f"{bbox['width']:.6f} {bbox['height']:.6f}\n")
        
        # 2. JSON format (more detailed, includes confidence)
        json_dir = output_dir / "json"
        json_dir.mkdir(exist_ok=True)
        json_path = json_dir / f"{img_name}.json"
        
        annotation = {
            'image': img_path.name,
            'image_width': img_width,
            'image_height': img_height,
            'detections': detections,
            'model': str(self.model_path.name),
            'confidence_threshold': self.confidence
        }
        
        with open(json_path, 'w') as f:
            json.dump(annotation, f, indent=2)
    
    def _print_summary(self, stats):
        """Print summary statistics."""
        print("\n" + "="*70)
        print("PRE-ANNOTATION SUMMARY")
        print("="*70)
        print(f"Total images: {stats['total_images']}")
        print(f"Images with detections: {stats['images_with_detections']} "
              f"({100*stats['images_with_detections']/stats['total_images']:.1f}%)")
        print(f"Total detections: {stats['total_detections']}")
        print(f"Average detections per image: "
              f"{stats['total_detections']/stats['total_images']:.1f}")
        print()
        print("Detections by class:")
        for class_name, count in stats['detections_by_class'].items():
            if count > 0:
                print(f"  {class_name}: {count}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-annotate extracted frames using existing YOLO model"
    )
    
    parser.add_argument(
        '--frames-dir',
        type=str,
        default='data/frames_for_annotation',
        help='Directory containing extracted frames'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/auklab_model_xlarge_combined_4564_v1.pt',
        help='Path to YOLO model'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.3,
        help='Confidence threshold (lower = more detections, default: 0.3)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for annotations (default: {frames_dir}/annotations)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*70)
    print("FRAME PRE-ANNOTATION")
    print("="*70)
    print()
    
    # Run pre-annotation
    annotator = FramePreAnnotator(
        model_path=args.model,
        confidence_threshold=args.confidence
    )
    
    annotator.pre_annotate_frames(
        frames_dir=args.frames_dir,
        output_dir=args.output_dir
    )
    
    print("\nNEXT STEPS:")
    print("="*70)
    print("1. Review the pre-annotations in your annotation tool")
    print("   - Correct any errors (false positives/negatives)")
    print("   - Adjust bounding boxes if needed")
    print("   - Pay special attention to edge detections and high-count scenes")
    print()
    print("2. Export corrected annotations in your training format")
    print()
    print("3. Combine with existing training dataset")
    print()
    print("4. Retrain your object detection model")
    print()
    print("5. Re-run event detection and compare results")
    print("   - Expect significant improvement in edge cases")
    print("   - Target: 70%+ F1 score (up from 61.2%)")
    print("="*70)


if __name__ == '__main__':
    main()
