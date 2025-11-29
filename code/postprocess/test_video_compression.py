#!/usr/bin/env python3
"""
Test script for video compression after YOLO detection.

This script tests different compression strategies to find the optimal
balance between file size and quality for event clips with YOLO overlays.
"""

import subprocess
import cv2
from pathlib import Path
import os

def get_video_info(video_path):
    """Get video file size and basic properties"""
    size_mb = os.path.getsize(video_path) / (1024 * 1024)
    
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    return {
        'size_mb': size_mb,
        'width': width,
        'height': height,
        'fps': fps,
        'frames': frame_count,
        'duration': frame_count / fps if fps > 0 else 0
    }

def compress_video_h264(input_path, output_path, crf=23, preset='medium'):
    """
    Compress video using H.264 codec with specified quality.
    
    CRF values:
    - 0 = lossless (huge file)
    - 18 = visually lossless (large file)
    - 23 = default (good quality, moderate size)
    - 28 = lower quality (small file)
    - 51 = worst quality
    
    Preset values (speed vs compression):
    - ultrafast, superfast, veryfast, faster, fast
    - medium (default)
    - slow, slower, veryslow
    """
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output
        '-i', str(input_path),
        '-c:v', 'libx264',  # H.264 codec
        '-preset', preset,
        '-crf', str(crf),
        '-c:a', 'aac',  # Audio codec
        '-b:a', '128k',  # Audio bitrate
        '-movflags', '+faststart',  # Enable fast start for web playback
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Compression failed: {e.stderr[:200]}")
        return False

def compress_video_h265(input_path, output_path, crf=28, preset='medium'):
    """
    Compress video using H.265/HEVC codec (better compression than H.264).
    Generally achieves 25-50% smaller files than H.264 at same quality.
    """
    cmd = [
        'ffmpeg',
        '-y',
        '-i', str(input_path),
        '-c:v', 'libx265',  # H.265 codec
        '-preset', preset,
        '-crf', str(crf),
        '-c:a', 'aac',
        '-b:a', '128k',
        '-tag:v', 'hvc1',  # Compatibility tag
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"H.265 compression failed: {e.stderr[:200]}")
        return False

def test_compression_methods(test_video):
    """Test different compression methods on a sample video"""
    test_video = Path(test_video)
    
    if not test_video.exists():
        print(f"Test video not found: {test_video}")
        return
    
    print("="*80)
    print("VIDEO COMPRESSION TEST")
    print("="*80)
    
    # Original video info
    print("\nORIGINAL VIDEO:")
    orig_info = get_video_info(test_video)
    print(f"  Size: {orig_info['size_mb']:.2f} MB")
    print(f"  Resolution: {orig_info['width']}x{orig_info['height']}")
    print(f"  FPS: {orig_info['fps']}")
    print(f"  Duration: {orig_info['duration']:.1f}s")
    print(f"  Frames: {orig_info['frames']}")
    
    # Test different compression settings
    test_configs = [
        {'codec': 'h264', 'crf': 23, 'preset': 'fast', 'name': 'H.264 CRF23 Fast'},
        {'codec': 'h264', 'crf': 28, 'preset': 'fast', 'name': 'H.264 CRF28 Fast'},
        {'codec': 'h264', 'crf': 28, 'preset': 'medium', 'name': 'H.264 CRF28 Medium'},
        {'codec': 'h265', 'crf': 28, 'preset': 'fast', 'name': 'H.265 CRF28 Fast'},
        {'codec': 'h265', 'crf': 28, 'preset': 'medium', 'name': 'H.265 CRF28 Medium'},
    ]
    
    results = []
    
    for config in test_configs:
        output_path = test_video.parent / f"{test_video.stem}_compressed_{config['codec']}_crf{config['crf']}_{config['preset']}.mp4"
        
        print(f"\nTesting: {config['name']}")
        print(f"  Output: {output_path.name}")
        
        if config['codec'] == 'h264':
            success = compress_video_h264(test_video, output_path, config['crf'], config['preset'])
        else:
            success = compress_video_h265(test_video, output_path, config['crf'], config['preset'])
        
        if success and output_path.exists():
            comp_info = get_video_info(output_path)
            compression_ratio = (1 - comp_info['size_mb'] / orig_info['size_mb']) * 100
            
            print(f"  ✓ Compressed size: {comp_info['size_mb']:.2f} MB")
            print(f"  ✓ Compression: {compression_ratio:.1f}% reduction")
            print(f"  ✓ Size ratio: {comp_info['size_mb'] / orig_info['size_mb']:.2%} of original")
            
            results.append({
                'name': config['name'],
                'size_mb': comp_info['size_mb'],
                'reduction': compression_ratio,
                'path': output_path
            })
        else:
            print(f"  ✗ Compression failed")
    
    # Summary
    print("\n" + "="*80)
    print("COMPRESSION RESULTS SUMMARY")
    print("="*80)
    print(f"{'Method':<25} {'Size (MB)':<12} {'Reduction':<15} {'Ratio':<10}")
    print("-"*80)
    print(f"{'Original':<25} {orig_info['size_mb']:<12.2f} {'-':<15} {'100%':<10}")
    
    for result in sorted(results, key=lambda x: x['size_mb']):
        ratio = result['size_mb'] / orig_info['size_mb']
        print(f"{result['name']:<25} {result['size_mb']:<12.2f} {result['reduction']:<14.1f}% {ratio:<9.1%}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    print("For event clips with YOLO overlays:")
    print("  • H.264 CRF28 Medium: Good balance of speed, quality, and size")
    print("  • H.265 CRF28 Fast: Better compression if H.265 is supported")
    print("  • Best for archival: H.264 CRF23")
    print("  • Best for small files: H.265 CRF28 Medium")
    print("\nRecommended: H.264 CRF28 with 'fast' preset")
    print("  - Fast encoding (important for batch processing)")
    print("  - Good compression (~60-70% smaller)")
    print("  - Wide compatibility")
    print("="*80)

def compress_with_recommended_settings(input_path, output_path):
    """
    Apply recommended compression settings for event clips.
    This is the function to integrate into extract_event_clips.py
    """
    return compress_video_h264(input_path, output_path, crf=28, preset='fast')

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_video_compression.py <video_file>")
        print("\nThis will test different compression methods on the video.")
        sys.exit(1)
    
    test_video = sys.argv[1]
    test_compression_methods(test_video)
