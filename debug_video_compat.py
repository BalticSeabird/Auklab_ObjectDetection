#!/usr/bin/env python3
"""
Hybrid video processor with DALI GPU decoding and OpenCV fallback
"""

import cv2
import numpy as np
import os
from pathlib import Path

def test_opencv_fallback():
    """Test if OpenCV can handle the video properly"""
    video_path = "vid/input2.mkv"
    
    print(f"🧪 Testing OpenCV fallback for: {video_path}")
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("   ❌ OpenCV cannot open the video")
            return False
            
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"   📊 Video properties:")
        print(f"      Resolution: {width}x{height}")
        print(f"      FPS: {fps}")
        print(f"      Total frames: {total_frames}")
        
        # Test reading frames
        frame_count = 0
        consecutive_failures = 0
        
        while frame_count < 100:  # Test first 100 frames
            ret, frame = cap.read()
            
            if not ret:
                consecutive_failures += 1
                if consecutive_failures > 5:
                    print(f"   ⚠️ Too many consecutive read failures at frame {frame_count}")
                    break
                continue
            else:
                consecutive_failures = 0
                
            frame_count += 1
            
            if frame_count % 25 == 0:
                print(f"   ✅ Successfully read frame {frame_count}")
        
        cap.release()
        
        if frame_count >= 90:  # Allow some tolerance
            print(f"   ✅ OpenCV can handle this video! Read {frame_count} frames successfully")
            return True
        else:
            print(f"   ❌ OpenCV had issues. Only read {frame_count} frames")
            return False
            
    except Exception as e:
        print(f"   ❌ OpenCV test failed: {e}")
        return False

def convert_to_h264():
    """Convert the problematic video to H.264 for better DALI compatibility"""
    input_path = "vid/input2.mkv"
    output_path = "vid/input2_h264.mp4"
    
    if os.path.exists(output_path):
        print(f"   ✅ H.264 version already exists: {output_path}")
        return output_path
    
    print(f"🔄 Converting {input_path} to H.264...")
    
    # Use ffmpeg to convert to H.264 with constant frame rate
    cmd = [
        "ffmpeg", "-i", input_path,
        "-c:v", "libx264",           # H.264 codec
        "-preset", "fast",           # Encoding speed
        "-crf", "23",                # Quality (lower = better)
        "-r", "25",                  # Force 25 FPS
        "-pix_fmt", "yuv420p",       # Standard pixel format
        "-avoid_negative_ts", "make_zero",  # Fix timestamp issues
        "-fflags", "+genpts",        # Generate presentation timestamps
        "-c:a", "aac",               # Audio codec
        "-b:a", "128k",              # Audio bitrate
        "-y",                        # Overwrite output
        output_path
    ]
    
    try:
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"   ✅ Conversion successful: {output_path}")
            return output_path
        else:
            print(f"   ❌ Conversion failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"   ❌ Conversion error: {e}")
        return None

if __name__ == "__main__":
    print("🔧 Video Compatibility Debug")
    print("=" * 40)
    
    # Test OpenCV first
    opencv_works = test_opencv_fallback()
    
    if opencv_works:
        print("\n💡 OpenCV can handle this video.")
        print("   The issue is specifically with DALI's HEVC decoder.")
        print("   Attempting H.264 conversion for DALI compatibility...")
        
        h264_path = convert_to_h264()
        
        if h264_path:
            print(f"\n✅ Try running the production script with: {h264_path}")
        else:
            print("\n⚠️ Conversion failed. Consider using OpenCV fallback mode.")
    else:
        print("\n❌ The video has fundamental issues that affect both DALI and OpenCV.")
        print("   The video file may be corrupted or have unsupported encoding.")