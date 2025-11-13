#!/usr/bin/env python3
"""
Quick upload script for active learning batches to Roboflow.
Uses API key from environment variable or interactive prompt.

Usage:
    # Set API key once (in your shell profile)
    export ROBOFLOW_API_KEY="your_key_here"
    
    # Then just run:
    python upload_batches_simple.py data/active_learning_TRI3_batch1/frames
    
    # With pre-annotations:
    python upload_batches_simple.py data/active_learning_TRI3_batch1/frames --with-annotations
    
    # Specific batches only:
    python upload_batches_simple.py data/active_learning_TRI3_batch1/frames --batches fish edge_detection
"""

import os
import sys
import argparse
from pathlib import Path

# Try to import the uploader
try:
    from upload_to_roboflow import RoboflowBatchUploader
except ImportError:
    print("Error: Could not import upload_to_roboflow module")
    print("Make sure you're running from the active_learning directory")
    sys.exit(1)


def get_api_key():
    """Get API key from environment or prompt user."""
    api_key = os.environ.get('ROBOFLOW_API_KEY')
    
    if not api_key:
        print("ROBOFLOW_API_KEY environment variable not set.")
        print()
        api_key = input("Enter your Roboflow API key: ").strip()
        
        if not api_key:
            print("Error: API key required")
            sys.exit(1)
    
    return api_key


def main():
    parser = argparse.ArgumentParser(
        description="Simple upload for active learning batches to Roboflow"
    )
    
    parser.add_argument('frames_dir', type=str,
                       help='Directory containing extracted frames')
    
    parser.add_argument('--with-annotations', action='store_true',
                       help='Upload with pre-annotations')
    
    parser.add_argument('--batches', nargs='+',
                       choices=['edge_detection', 'spike', 'dip', 'high_count', 
                               'count_transition', 'fish'],
                       help='Specific batches to upload (default: all)')
    
    parser.add_argument('--workspace', type=str, default='ai-course-2024',
                       help='Roboflow workspace (default: ai-course-2024)')
    
    parser.add_argument('--project', type=str, default='fish_seabirds_combined-625bd',
                       help='Roboflow project (default: fish_seabirds_combined-625bd)')
    
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh upload (ignore previous progress)')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = get_api_key()
    
    print("\n" + "="*80)
    print("ACTIVE LEARNING BATCH UPLOAD TO ROBOFLOW")
    print("="*80)
    print(f"Workspace: {args.workspace}")
    print(f"Project: {args.project}")
    print(f"Frames directory: {args.frames_dir}")
    print(f"With annotations: {args.with_annotations}")
    if args.batches:
        print(f"Batches: {', '.join(args.batches)}")
    else:
        print("Batches: All available")
    print("="*80)
    
    # Initialize and upload
    try:
        uploader = RoboflowBatchUploader(
            api_key=api_key,
            workspace=args.workspace,
            project=args.project
        )
        
        uploader.upload_batches(
            frames_dir=Path(args.frames_dir),
            batch_types=args.batches,
            use_annotations=args.with_annotations,
            resume=not args.no_resume
        )
        
    except Exception as e:
        print(f"\n✗ Error during upload: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
