#!/usr/bin/env python3
"""
Upload Active Learning Batches to Roboflow

This script uploads extracted frames from the active learning pipeline to Roboflow,
organized by problem type (edge_detection, spike, dip, high_count, count_transition, fish).
Each problem type is uploaded as a separate batch for easier annotation workflow.

Usage:
    # Upload all batches
    python code/active_learning/upload_to_roboflow.py \
        --frames-dir data/active_learning_TRI3_batch1/frames \
        --api-key YOUR_API_KEY \
        --workspace ai-course-2024 \
        --project fish_seabirds_combined-625bd

    # Upload specific batches only
    python code/active_learning/upload_to_roboflow.py \
        --frames-dir data/active_learning_TRI3_batch1/frames \
        --api-key YOUR_API_KEY \
        --workspace ai-course-2024 \
        --project fish_seabirds_combined-625bd \
        --batches edge_detection fish

    # Use pre-annotations (YOLO format)
    python code/active_learning/upload_to_roboflow.py \
        --frames-dir data/active_learning_TRI3_batch1/frames \
        --use-annotations \
        --api-key YOUR_API_KEY

Requirements:
    pip install roboflow
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import json
from datetime import datetime

try:
    from roboflow import Roboflow
    HAS_ROBOFLOW = True
except ImportError:
    print("ERROR: roboflow package not installed.")
    print("Install with: pip install roboflow")
    HAS_ROBOFLOW = False
    sys.exit(1)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    print("Warning: tqdm not installed. Install for progress bars: pip install tqdm")
    HAS_TQDM = False
    # Fallback progress indicator
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, **kwargs):
            self.iterable = iterable
            self.total = total or (len(iterable) if iterable else 0)
            self.desc = desc
            self.n = 0
        
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.update(1)
        
        def update(self, n=1):
            self.n += n
            if self.total:
                print(f"\r{self.desc}: {self.n}/{self.total}", end='', flush=True)
        
        def close(self):
            print()  # New line after completion


class RoboflowBatchUploader:
    """
    Handles uploading active learning batches to Roboflow with progress tracking and resume capability.
    """
    
    def __init__(self, api_key: str, workspace: str, project: str):
        """
        Initialize Roboflow connection.
        
        Args:
            api_key: Roboflow API key
            workspace: Workspace name/ID
            project: Project name/ID
        """
        self.rf = Roboflow(api_key=api_key)
        self.project = self.rf.workspace(workspace).project(project)
        print(f"✓ Connected to Roboflow project: {workspace}/{project}")
    
    def _get_upload_state_file(self, frames_dir: Path) -> Path:
        """Get path to upload state file for resume functionality."""
        return frames_dir / ".upload_state.json"
    
    def _load_upload_state(self, state_file: Path) -> dict:
        """Load previous upload state."""
        if state_file.exists():
            with open(state_file, 'r') as f:
                return json.load(f)
        return {'uploaded_files': [], 'completed_batches': []}
    
    def _save_upload_state(self, state_file: Path, state: dict):
        """Save upload state for resume."""
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _clear_upload_state(self, state_file: Path):
        """Clear upload state after successful completion."""
        if state_file.exists():
            state_file.unlink()
    
    def upload_batches(self, frames_dir: Path, 
                      batch_types: Optional[List[str]] = None,
                      use_annotations: bool = False,
                      split: str = "train",
                      batch_name_prefix: str = "active_learning",
                      batch_id: Optional[str] = None,
                      resume: bool = True):
        """
        Upload frames organized by problem type to Roboflow.
        
        Args:
            frames_dir: Directory containing problem type subdirectories
            batch_types: List of batch types to upload (None = all)
            use_annotations: Whether to upload pre-annotations (from annotations/yolo/)
            split: Dataset split (train/valid/test)
            batch_name_prefix: Prefix for batch names in Roboflow
            batch_id: Unique ID for this batch (auto-generated if None)
            resume: Whether to resume from previous interrupted upload
        """
        frames_dir = Path(frames_dir)
        state_file = self._get_upload_state_file(frames_dir)
        
        # Generate unique batch ID if not provided
        if batch_id is None:
            batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load previous state if resuming
        upload_state = self._load_upload_state(state_file) if resume else {'uploaded_files': [], 'completed_batches': [], 'batch_id': batch_id}
        
        # Use existing batch_id if resuming
        if 'batch_id' in upload_state:
            batch_id = upload_state['batch_id']
        else:
            upload_state['batch_id'] = batch_id
        
        if upload_state['uploaded_files'] or upload_state['completed_batches']:
            print(f"\n⚠ Found previous upload state:")
            print(f"  - Batch ID: {batch_id}")
            print(f"  - {len(upload_state['uploaded_files'])} files already uploaded")
            print(f"  - {len(upload_state['completed_batches'])} batches completed")
            response = input("Resume from previous upload? (y/n): ").strip().lower()
            if response != 'y':
                # Generate new batch ID for fresh start
                batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                upload_state = {'uploaded_files': [], 'completed_batches': [], 'batch_id': batch_id}
                self._clear_upload_state(state_file)
        
        # Available batch types
        available_batches = {
            'edge_detection': 'Edge Detections',
            'spike': 'Spike Artifacts',
            'dip': 'Dip Artifacts',
            'high_count': 'High Count Scenes',
            'count_transition': 'Count Transitions',
            'fish': 'Fish Detections'
        }
        
        # Determine which batches to upload
        if batch_types is None:
            batch_types = [b for b in available_batches.keys() 
                          if (frames_dir / b).exists()]
        else:
            batch_types = [b for b in batch_types if b in available_batches]
        
        # Filter out completed batches
        batch_types = [b for b in batch_types if b not in upload_state['completed_batches']]
        
        if not batch_types:
            print("✗ No batches found to upload (or all already completed)")
            return
        
        # Check for annotations directory
        annotations_dir = frames_dir / "annotations" / "yolo" if use_annotations else None
        if use_annotations and not annotations_dir.exists():
            print(f"⚠ Annotations directory not found: {annotations_dir}")
            print("  Uploading without annotations")
            use_annotations = False
        
        print("\n" + "="*80)
        print("UPLOAD PLAN")
        print("="*80)
        print(f"Batch ID: {batch_id}")
        print(f"Frames directory: {frames_dir}")
        print(f"Use annotations: {use_annotations}")
        print(f"Split: {split}")
        print(f"Resume mode: {resume and len(upload_state['uploaded_files']) > 0}")
        print(f"\nBatches to upload:")
        total_images = 0
        for batch_type in batch_types:
            batch_dir = frames_dir / batch_type
            if batch_dir.exists():
                n_images = len(list(batch_dir.glob("*.jpg"))) + len(list(batch_dir.glob("*.png")))
                total_images += n_images
                batch_full_name = f"{batch_name_prefix}_{batch_id}_{batch_type}"
                print(f"  - {available_batches[batch_type]:25s}: {n_images:4d} images → {batch_full_name}")
        
        if upload_state['uploaded_files']:
            print(f"\nAlready uploaded: {len(upload_state['uploaded_files'])} files")
        
        # Confirm upload
        print("\n" + "="*80)
        response = input("Proceed with upload? (y/n): ").strip().lower()
        if response != 'y':
            print("Upload cancelled")
            return
        
        # Upload each batch
        print("\n" + "="*80)
        print("UPLOADING BATCHES")
        print("="*80)
        
        total_uploaded = 0
        total_failed = 0
        
        try:
            for i, batch_type in enumerate(batch_types, 1):
                print(f"\n[{i}/{len(batch_types)}] Uploading: {available_batches[batch_type]}")
                print("-" * 80)
                
                uploaded, failed = self._upload_batch(
                    batch_dir=frames_dir / batch_type,
                    batch_name=f"{batch_name_prefix}_{batch_id}_{batch_type}",
                    annotations_dir=annotations_dir,
                    split=split,
                    upload_state=upload_state,
                    state_file=state_file
                )
                
                total_uploaded += uploaded
                total_failed += failed
                
                # Mark batch as completed
                upload_state['completed_batches'].append(batch_type)
                self._save_upload_state(state_file, upload_state)
                
                print(f"  ✓ Batch complete - Uploaded: {uploaded}, Failed: {failed}")
            
            # Clear state after successful completion
            self._clear_upload_state(state_file)
            
        except KeyboardInterrupt:
            print("\n\n⚠ Upload interrupted by user")
            print(f"Progress saved. Run again with same parameters to resume.")
            self._save_upload_state(state_file, upload_state)
            return
        except Exception as e:
            print(f"\n\n✗ Upload failed with error: {e}")
            print(f"Progress saved. Run again with same parameters to resume.")
            self._save_upload_state(state_file, upload_state)
            raise
        
        # Final summary
        print("\n" + "="*80)
        print("UPLOAD COMPLETE")
        print("="*80)
        print(f"Total uploaded: {total_uploaded}")
        print(f"Total failed: {total_failed}")
        print("\nNext steps:")
        print("  1. Go to Roboflow and review the uploaded images")
        print("  2. Validate/correct the annotations")
        print("  3. Assign images to annotation team if needed")
        print("  4. Export corrected annotations when done")
        print("="*80)
    
    def _upload_batch(self, batch_dir: Path, batch_name: str,
                     annotations_dir: Optional[Path], split: str,
                     upload_state: dict, state_file: Path):
        """Upload a single batch of images with progress tracking."""
        
        uploaded = 0
        failed = 0
        skipped = 0
        
        # Find all images
        image_files = list(batch_dir.glob("*.jpg")) + list(batch_dir.glob("*.png"))
        
        if not image_files:
            print(f"  ⚠ No images found in {batch_dir}")
            return uploaded, failed
        
        # Filter out already uploaded files
        uploaded_files_set = set(upload_state['uploaded_files'])
        remaining_files = [f for f in image_files if str(f) not in uploaded_files_set]
        skipped = len(image_files) - len(remaining_files)
        
        if skipped > 0:
            print(f"  ℹ Skipping {skipped} already uploaded files")
        
        if not remaining_files:
            print(f"  ✓ All files already uploaded")
            return 0, 0
        
        # Upload with progress bar
        pbar = tqdm(remaining_files, desc=f"  Uploading {batch_dir.name}", 
                   unit="img", ncols=100)
        
        for img_path in pbar:
            try:
                # Find corresponding annotation if available
                annotation_path = None
                if annotations_dir:
                    annotation_path = annotations_dir / f"{img_path.stem}.txt"
                    if not annotation_path.exists():
                        annotation_path = None
                
                # Upload to Roboflow
                if annotation_path:
                    self.project.upload(
                        image_path=str(img_path),
                        annotation_path=str(annotation_path),
                        split=split,
                        batch_name=batch_name,
                        tag_names=[batch_dir.name]
                    )
                else:
                    self.project.upload(
                        image_path=str(img_path),
                        split=split,
                        batch_name=batch_name,
                        tag_names=[batch_dir.name]
                    )
                
                uploaded += 1
                
                # Track uploaded file
                upload_state['uploaded_files'].append(str(img_path))
                
                # Save state periodically (every 10 files)
                if uploaded % 10 == 0:
                    self._save_upload_state(state_file, upload_state)
                
            except KeyboardInterrupt:
                pbar.close()
                raise
            except Exception as e:
                pbar.write(f"  ✗ Failed: {img_path.name} - {str(e)[:50]}")
                failed += 1
        
        pbar.close()
        
        # Final state save
        self._save_upload_state(state_file, upload_state)
        
        return uploaded, failed


def main():
    parser = argparse.ArgumentParser(
        description="Upload active learning batches to Roboflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload all batches with pre-annotations
  python upload_to_roboflow.py \\
      --frames-dir data/active_learning_TRI3_batch1/frames \\
      --use-annotations \\
      --api-key YOUR_KEY \\
      --workspace ai-course-2024 \\
      --project fish_seabirds_combined-625bd

  # Upload only fish and edge_detection batches
  python upload_to_roboflow.py \\
      --frames-dir data/active_learning_TRI3_batch1/frames \\
      --batches fish edge_detection \\
      --api-key YOUR_KEY

  # Upload without pre-annotations (for manual annotation from scratch)
  python upload_to_roboflow.py \\
      --frames-dir data/active_learning_TRI3_batch1/frames \\
      --api-key YOUR_KEY
        """
    )
    
    parser.add_argument('--frames-dir', type=str, required=True,
                       help='Directory containing extracted frames (with subdirectories)')
    
    parser.add_argument('--api-key', type=str, required=True,
                       help='Roboflow API key')
    
    parser.add_argument('--workspace', type=str, default='ai-course-2024',
                       help='Roboflow workspace name')
    
    parser.add_argument('--project', type=str, default='fish_seabirds_combined-625bd',
                       help='Roboflow project name')
    
    parser.add_argument('--batches', nargs='+',
                       choices=['edge_detection', 'spike', 'dip', 'high_count', 
                               'count_transition', 'fish'],
                       help='Specific batch types to upload (default: all)')
    
    parser.add_argument('--use-annotations', action='store_true',
                       help='Upload with pre-annotations from annotations/yolo/')
    
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'valid', 'test'],
                       help='Dataset split (default: train)')
    
    parser.add_argument('--batch-name-prefix', type=str, default='active_learning',
                       help='Prefix for batch names in Roboflow')
    
    parser.add_argument('--batch-id', type=str,
                       help='Unique batch ID (auto-generated timestamp if not provided)')
    
    args = parser.parse_args()
    
    # Initialize uploader
    uploader = RoboflowBatchUploader(
        api_key=args.api_key,
        workspace=args.workspace,
        project=args.project
    )
    
    # Upload batches
    uploader.upload_batches(
        frames_dir=Path(args.frames_dir),
        batch_types=args.batches,
        use_annotations=args.use_annotations,
        split=args.split,
        batch_name_prefix=args.batch_name_prefix,
        batch_id=args.batch_id
    )


if __name__ == '__main__':
    main()
