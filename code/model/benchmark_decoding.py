#!/usr/bin/env python3
"""
DALI vs PyAV Performance Benchmark
Compare GPU-accelerated DALI decoding vs CPU-based PyAV decoding
"""

import time
import numpy as np
import os
import sys
from pathlib import Path

# Test if modules are available
try:
    import nvidia.dali as dali
    from nvidia.dali import pipeline_def
    import nvidia.dali.fn as fn
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    from nvidia.dali import types
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False

try:
    import av
    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False

import cv2

@pipeline_def
def benchmark_dali_pipeline(video_path, batch_size, frame_skip=1):
    """DALI pipeline for benchmarking"""
    videos = fn.readers.video(
        device="gpu",
        file_root="",
        filenames=[video_path],
        sequence_length=batch_size,
        step=frame_skip,
        stride=1,
        normalized=False,
        random_shuffle=False,
        pad_last_batch=True,
        name="video_reader"
    )
    
    frames = videos[0]
    
    # Resize to 960x960
    frames = fn.resize(
        frames,
        device="gpu",
        size=[960, 960],
        interp_type=types.INTERP_LINEAR
    )
    
    # Normalize to [0,1]
    frames = fn.cast(frames, device="gpu", dtype=types.FLOAT)
    frames = frames / 255.0
    
    # Transpose HWC to CHW
    frames = fn.transpose(frames, device="gpu", perm=[2, 0, 1])
    
    return frames

def benchmark_dali_decoding(video_path, batch_size=8, max_batches=50, frame_skip=25):
    """Benchmark DALI GPU decoding performance"""
    
    if not DALI_AVAILABLE:
        return None
    
    print(f"\n🔥 Benchmarking DALI GPU Decoding")
    print(f"   Video: {video_path}")
    print(f"   Batch size: {batch_size}")
    print(f"   Max batches: {max_batches}")
    print(f"   Frame skip: {frame_skip}")
    
    try:
        # Create pipeline
        pipeline = benchmark_dali_pipeline(
            video_path=str(video_path),
            batch_size=batch_size,
            frame_skip=frame_skip
        )
        pipeline.build()
        
        # Create iterator
        iterator = DALIGenericIterator(
            [pipeline],
            output_map=["frames"],
            reader_name="video_reader",
            last_batch_policy="fill",
            auto_reset=True
        )
        
        # Benchmark
        decode_times = []
        total_frames = 0
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(iterator):
            if batch_idx >= max_batches:
                break
            
            batch_start = time.time()
            
            # Get frames (this includes all DALI processing)
            frames = batch[0]["frames"]
            
            # Force computation by accessing data
            if hasattr(frames, 'cpu'):
                frame_data = frames.cpu().numpy()
            else:
                frame_data = frames
            
            batch_time = time.time() - batch_start
            decode_times.append(batch_time * 1000)  # Convert to ms
            
            actual_batch_size = frame_data.shape[0]
            total_frames += actual_batch_size
            
            if batch_idx % 10 == 0:
                print(f"   Batch {batch_idx}: {batch_time*1000:.1f}ms, {actual_batch_size} frames")
        
        total_time = time.time() - start_time
        
        # Cleanup
        del iterator
        del pipeline
        
        # Calculate metrics
        avg_decode_time = np.mean(decode_times)
        fps = total_frames / total_time
        throughput = batch_size * 1000 / avg_decode_time if avg_decode_time > 0 else 0
        
        results = {
            'method': 'DALI GPU',
            'total_time': total_time,
            'total_frames': total_frames,
            'avg_decode_time_ms': avg_decode_time,
            'fps': fps,
            'throughput_images_per_sec': throughput,
            'batches_processed': len(decode_times)
        }
        
        print(f"✅ DALI Results:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Frames processed: {total_frames}")
        print(f"   Average decode time: {avg_decode_time:.1f}ms per batch")
        print(f"   FPS: {fps:.1f}")
        print(f"   Throughput: {throughput:.1f} images/second")
        
        return results
        
    except Exception as e:
        print(f"❌ DALI benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def benchmark_pyav_decoding(video_path, batch_size=8, max_batches=50, frame_skip=25):
    """Benchmark PyAV CPU decoding performance"""
    
    if not PYAV_AVAILABLE:
        return None
    
    print(f"\n🐍 Benchmarking PyAV CPU Decoding")
    print(f"   Video: {video_path}")
    print(f"   Batch size: {batch_size}")
    print(f"   Max batches: {max_batches}")
    print(f"   Frame skip: {frame_skip}")
    
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        
        decode_times = []
        total_frames = 0
        frame_buffer = []
        frame_count = 0
        batches_processed = 0
        
        start_time = time.time()
        
        for frame in container.decode(stream):
            if batches_processed >= max_batches:
                break
            
            if frame_count % frame_skip == 0:
                batch_start = time.time()
                
                # Convert frame (CPU operation)
                img = frame.to_ndarray(format='bgr24')
                
                # Resize (CPU operation)  
                img = cv2.resize(img, (960, 960))
                
                # Normalize (CPU operation)
                img = img.astype(np.float32) / 255.0
                
                # Transpose HWC to CHW (CPU operation)
                img = np.transpose(img, (2, 0, 1))
                
                frame_buffer.append(img)
                
                # When batch is full, record timing
                if len(frame_buffer) == batch_size:
                    batch_time = time.time() - batch_start
                    decode_times.append(batch_time * 1000)  # Convert to ms
                    
                    total_frames += len(frame_buffer)
                    batches_processed += 1
                    
                    if batches_processed % 10 == 0:
                        print(f"   Batch {batches_processed}: {batch_time*1000:.1f}ms, {len(frame_buffer)} frames")
                    
                    frame_buffer = []
            
            frame_count += 1
        
        container.close()
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        avg_decode_time = np.mean(decode_times) if decode_times else 0
        fps = total_frames / total_time if total_time > 0 else 0
        throughput = batch_size * 1000 / avg_decode_time if avg_decode_time > 0 else 0
        
        results = {
            'method': 'PyAV CPU',
            'total_time': total_time,
            'total_frames': total_frames,
            'avg_decode_time_ms': avg_decode_time,
            'fps': fps,
            'throughput_images_per_sec': throughput,
            'batches_processed': len(decode_times)
        }
        
        print(f"✅ PyAV Results:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Frames processed: {total_frames}")
        print(f"   Average decode time: {avg_decode_time:.1f}ms per batch")
        print(f"   FPS: {fps:.1f}")
        print(f"   Throughput: {throughput:.1f} images/second")
        
        return results
        
    except Exception as e:
        print(f"❌ PyAV benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def benchmark_opencv_decoding(video_path, batch_size=8, max_batches=50, frame_skip=25):
    """Benchmark OpenCV CPU decoding performance as baseline"""
    
    print(f"\n📸 Benchmarking OpenCV CPU Decoding (Baseline)")
    print(f"   Video: {video_path}")
    print(f"   Batch size: {batch_size}")
    print(f"   Max batches: {max_batches}")
    print(f"   Frame skip: {frame_skip}")
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        decode_times = []
        total_frames = 0
        frame_buffer = []
        frame_count = 0
        batches_processed = 0
        
        start_time = time.time()
        
        while cap.isOpened() and batches_processed < max_batches:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                batch_start = time.time()
                
                # Resize (CPU operation)
                img = cv2.resize(frame, (960, 960))
                
                # Normalize (CPU operation)
                img = img.astype(np.float32) / 255.0
                
                # Transpose HWC to CHW (CPU operation)
                img = np.transpose(img, (2, 0, 1))
                
                frame_buffer.append(img)
                
                # When batch is full, record timing
                if len(frame_buffer) == batch_size:
                    batch_time = time.time() - batch_start
                    decode_times.append(batch_time * 1000)  # Convert to ms
                    
                    total_frames += len(frame_buffer)
                    batches_processed += 1
                    
                    if batches_processed % 10 == 0:
                        print(f"   Batch {batches_processed}: {batch_time*1000:.1f}ms, {len(frame_buffer)} frames")
                    
                    frame_buffer = []
            
            frame_count += 1
        
        cap.release()
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        avg_decode_time = np.mean(decode_times) if decode_times else 0
        fps = total_frames / total_time if total_time > 0 else 0
        throughput = batch_size * 1000 / avg_decode_time if avg_decode_time > 0 else 0
        
        results = {
            'method': 'OpenCV CPU',
            'total_time': total_time,
            'total_frames': total_frames,
            'avg_decode_time_ms': avg_decode_time,
            'fps': fps,
            'throughput_images_per_sec': throughput,
            'batches_processed': len(decode_times)
        }
        
        print(f"✅ OpenCV Results:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Frames processed: {total_frames}")
        print(f"   Average decode time: {avg_decode_time:.1f}ms per batch")
        print(f"   FPS: {fps:.1f}")
        print(f"   Throughput: {throughput:.1f} images/second")
        
        return results
        
    except Exception as e:
        print(f"❌ OpenCV benchmark failed: {e}")
        return None

def run_comprehensive_benchmark(video_path, batch_sizes=[4, 8, 16], frame_skip=25):
    """Run comprehensive benchmark comparing all methods"""
    
    print(f"🚀 Comprehensive Video Decoding Benchmark")
    print(f"=" * 80)
    print(f"Video: {video_path}")
    print(f"Batch sizes to test: {batch_sizes}")
    print(f"Frame skip: {frame_skip}")
    print(f"DALI available: {DALI_AVAILABLE}")
    print(f"PyAV available: {PYAV_AVAILABLE}")
    
    all_results = []
    
    for batch_size in batch_sizes:
        print(f"\n" + "="*80)
        print(f"TESTING BATCH SIZE: {batch_size}")
        print(f"="*80)
        
        # Test DALI if available
        if DALI_AVAILABLE:
            dali_result = benchmark_dali_decoding(video_path, batch_size=batch_size, frame_skip=frame_skip)
            if dali_result:
                dali_result['batch_size'] = batch_size
                all_results.append(dali_result)
        else:
            print("⚠️  DALI not available - skipping GPU benchmark")
        
        # Test PyAV if available  
        if PYAV_AVAILABLE:
            pyav_result = benchmark_pyav_decoding(video_path, batch_size=batch_size, frame_skip=frame_skip)
            if pyav_result:
                pyav_result['batch_size'] = batch_size
                all_results.append(pyav_result)
        else:
            print("⚠️  PyAV not available - skipping PyAV benchmark")
        
        # Test OpenCV as baseline
        opencv_result = benchmark_opencv_decoding(video_path, batch_size=batch_size, frame_skip=frame_skip)
        if opencv_result:
            opencv_result['batch_size'] = batch_size
            all_results.append(opencv_result)
    
    # Print summary
    print(f"\n" + "="*80)
    print(f"BENCHMARK SUMMARY")
    print(f"="*80)
    
    print(f"{'Method':<12} | {'Batch':<5} | {'Decode ms':<9} | {'FPS':<8} | {'Throughput':<11} | {'Speedup':<7}")
    print("-" * 80)
    
    # Find baseline (OpenCV CPU with batch size 8)
    baseline_throughput = None
    for result in all_results:
        if result['method'] == 'OpenCV CPU' and result['batch_size'] == 8:
            baseline_throughput = result['throughput_images_per_sec']
            break
    
    if baseline_throughput is None and all_results:
        baseline_throughput = all_results[0]['throughput_images_per_sec']
    
    for result in all_results:
        speedup = result['throughput_images_per_sec'] / baseline_throughput if baseline_throughput > 0 else 1.0
        
        print(f"{result['method']:<12} | {result['batch_size']:<5} | {result['avg_decode_time_ms']:<9.1f} | "
              f"{result['fps']:<8.1f} | {result['throughput_images_per_sec']:<11.1f} | {speedup:<7.2f}x")
    
    # Find best performers
    print(f"\n🏆 BEST PERFORMERS:")
    if all_results:
        best_throughput = max(all_results, key=lambda x: x['throughput_images_per_sec'])
        best_fps = max(all_results, key=lambda x: x['fps'])
        
        print(f"   Highest throughput: {best_throughput['method']} (batch {best_throughput['batch_size']}) - {best_throughput['throughput_images_per_sec']:.1f} img/s")
        print(f"   Highest FPS: {best_fps['method']} (batch {best_fps['batch_size']}) - {best_fps['fps']:.1f} fps")
    
    return all_results

def main():
    """Main benchmark entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Video Decoding Performance Benchmark')
    parser.add_argument('video_path', type=str, help='Path to test video file')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[4, 8, 16], 
                        help='Batch sizes to test')
    parser.add_argument('--frame-skip', type=int, default=25, 
                        help='Process every Nth frame')
    parser.add_argument('--max-batches', type=int, default=50,
                        help='Maximum batches to process per test')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"❌ Video file not found: {args.video_path}")
        return
    
    # Set global max_batches for individual tests
    global MAX_BATCHES
    MAX_BATCHES = args.max_batches
    
    # Run comprehensive benchmark
    results = run_comprehensive_benchmark(
        video_path=args.video_path,
        batch_sizes=args.batch_sizes,
        frame_skip=args.frame_skip
    )
    
    print(f"\n✅ Benchmark completed! Tested {len(results)} configurations.")

if __name__ == "__main__":
    main()