#!/usr/bin/env python3
"""
Production Batch Inference for Full Video Processing
- Optimized for 1+ hour video files
- Uses proven PyAV decoding + batch TensorRT (323+ FPS)
- Optimal batch size: 21 frames
- Memory efficient for long videos
"""

import pandas as pd
from pathlib import Path
import av  # PyAV for video processing
import os
import torch
import torchvision.ops as ops
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import sys
import argparse
from datetime import datetime, timedelta

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class ProductionBatchProcessor:
    """Production-ready batch processor for long videos"""
    
    def __init__(self, engine_path, batch_size=21):
        print(f"🚀 Initializing Production Batch Processor")
        print(f"   Engine: {engine_path}")
        print(f"   Optimal batch size: {batch_size}")
        
        self.batch_size = batch_size
        self.device = torch.device('cuda:0')
        
        # Load TensorRT engine
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Create CUDA stream for asynchronous execution
        self.stream = cuda.Stream()
        
        # Get tensor info
        self.input_name = None
        self.output_names = []
        
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
            
            if is_input:
                self.input_name = tensor_name
            else:
                self.output_names.append(tensor_name)
        
        # Pre-allocate memory for optimal batch size
        self._allocate_memory()
        
        # Performance tracking
        self.start_time = None
        self.processed_frames = 0
        self.total_detections = 0
        self.inference_times = []
        
        print(f"✅ Ready for production processing!")
    
    def _allocate_memory(self):
        """Pre-allocate GPU memory for optimal performance"""
        # Input memory
        input_shape = (self.batch_size, 3, 960, 960)
        input_size = int(np.prod(input_shape) * 4)
        
        self.d_input = cuda.mem_alloc(input_size)
        self.h_input = cuda.pagelocked_empty(input_shape, dtype=np.float32)
        
        # Output memory (allocate for all output tensors)
        self.output_allocations = []
        
        # Set a temporary input shape to get output shapes
        self.context.set_input_shape(self.input_name, input_shape)
        
        for output_name in self.output_names:
            try:
                output_shape = self.context.get_tensor_shape(output_name)
                
                # Convert TensorRT Dims to tuple
                if hasattr(output_shape, '__len__'):
                    shape_tuple = tuple(output_shape)
                else:
                    # Handle TensorRT Dims object
                    shape_tuple = tuple(output_shape[i] for i in range(len(output_shape)))
                
                output_size = int(np.prod(shape_tuple) * 4)
                d_output = cuda.mem_alloc(output_size)
                h_output = cuda.pagelocked_empty(shape_tuple, dtype=np.float32)
                
                self.output_allocations.append({
                    'name': output_name,
                    'device': d_output,
                    'host': h_output,
                    'shape': shape_tuple
                })
                
            except Exception as e:
                print(f"   Warning: Could not allocate for {output_name}: {e}")
        
        total_memory = (input_size + sum(np.prod(out['shape']) * 4 for out in self.output_allocations)) / (1024**2)
        print(f"   GPU memory allocated: {total_memory:.1f} MB")
    
    def process_full_video(self, video_path, output_csv, frame_skip=25, progress_interval=300):
        """Process complete video with progress tracking"""
        
        print(f"\n📹 Processing Full Video: {video_path}")
        print(f"   Frame skip: {frame_skip} (every {frame_skip}th frame)")
        print(f"   Output: {output_csv}")
        print(f"   Progress updates every: {progress_interval} seconds")
        
        self.start_time = time.time()
        last_progress_time = self.start_time
        
        # Open video
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        
        # Get video info
        total_frames = stream.frames
        fps = float(stream.average_rate)
        duration_hours = (total_frames / fps) / 3600 if fps > 0 else 0
        
        print(f"   Video info: {total_frames:,} frames, {fps:.1f} FPS, {duration_hours:.2f} hours")
        
        results_list = []
        frame_buffer = []
        frame_indices = []
        frame_count = 0
        
        try:
            # Process video frames
            for frame in container.decode(stream):
                if frame_count % frame_skip == 0:
                    # Convert frame
                    img = frame.to_ndarray(format='bgr24')
                    frame_buffer.append(img)
                    frame_indices.append(frame_count)
                    
                    # Process when batch is full
                    if len(frame_buffer) == self.batch_size:
                        detections = self._process_batch(frame_buffer, frame_indices)
                        results_list.extend(detections)
                        
                        # Clear batch
                        frame_buffer = []
                        frame_indices = []
                        
                        # Progress update
                        current_time = time.time()
                        if current_time - last_progress_time >= progress_interval:
                            self._print_progress(frame_count, total_frames, fps)
                            last_progress_time = current_time
                            
                            # Save intermediate results
                            if results_list:
                                self._save_results(results_list, output_csv + ".temp")
                
                frame_count += 1
            
            # Process remaining frames
            if frame_buffer:
                detections = self._process_batch(frame_buffer, frame_indices)
                results_list.extend(detections)
            
        except KeyboardInterrupt:
            print(f"\n⚠️  Processing interrupted by user")
        except Exception as e:
            print(f"\n❌ Error during processing: {e}")
        finally:
            container.close()
        
        # Final save
        if results_list:
            self._save_results(results_list, output_csv)
            
            # Remove temp file if it exists
            temp_file = output_csv + ".temp"
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        # Final statistics
        self._print_final_stats(frame_count, total_frames, fps)
        
        return results_list
    
    def _process_batch(self, frames, indices):
        """Process batch with TensorRT + GPU NMS"""
        batch_start = time.time()
        
        # Preprocess frames
        for i, img in enumerate(frames):
            if img.shape[:2] != (960, 960):
                img = cv2.resize(img, (960, 960))
            
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC to CHW
            self.h_input[i] = img
        
        # Set dynamic shape
        actual_batch_size = len(frames)
        input_shape = (actual_batch_size, 3, 960, 960)
        self.context.set_input_shape(self.input_name, input_shape)
        
        # Copy to GPU and execute (asynchronous)
        cuda.memcpy_htod_async(self.d_input, self.h_input[:actual_batch_size], self.stream)
        
        # Set tensor addresses
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        for output_alloc in self.output_allocations:
            self.context.set_tensor_address(output_alloc['name'], int(output_alloc['device']))
        
        # Execute inference with proper stream
        success = self.context.execute_async_v3(self.stream.handle)
        
        if not success:
            return []
        
        # Synchronize before copying results
        self.stream.synchronize()
        
        # Get main output (usually the largest - detection results)
        main_output = max(self.output_allocations, key=lambda x: np.prod(x['shape']))
        cuda.memcpy_dtoh(main_output['host'][:actual_batch_size], main_output['device'])
        
        # Track inference time
        inference_time = (time.time() - batch_start) * 1000
        self.inference_times.append(inference_time)
        
        # Process detections with GPU NMS
        predictions = main_output['host'][:actual_batch_size]
        batch_detections = self._gpu_nms_batch(predictions)
        
        # Convert to results
        results = []
        for i, (frame_idx, detections) in enumerate(zip(indices, batch_detections)):
            if detections is not None:
                boxes, scores, classes = detections
                for j in range(len(boxes)):
                    results.append({
                        'frame': frame_idx,
                        'class': int(classes[j]),
                        'confidence': float(scores[j]),
                        'xmin': float(boxes[j][0]),
                        'ymin': float(boxes[j][1]),
                        'xmax': float(boxes[j][2]),
                        'ymax': float(boxes[j][3])
                    })
                
                self.total_detections += len(boxes)
        
        self.processed_frames += len(frames)
        return results
    
    def _gpu_nms_batch(self, predictions, conf_threshold=0.25, nms_threshold=0.45):
        """GPU-based NMS for batch predictions"""
        batch_results = []
        
        for pred in predictions:
            try:
                if pred is None or pred.size == 0:
                    batch_results.append(None)
                    continue
                
                # Convert to torch tensor
                pred_tensor = torch.from_numpy(pred).to(self.device)
                
                # Reshape if needed - handle different output formats
                if pred_tensor.dim() == 1:
                    # Flatten format - try to reshape
                    if pred_tensor.numel() % 7 == 0:
                        pred_tensor = pred_tensor.view(-1, 7)
                    else:
                        batch_results.append(None)
                        continue
                elif pred_tensor.dim() == 2 and pred_tensor.shape[0] == 7:
                    # Transpose if needed
                    pred_tensor = pred_tensor.transpose(0, 1)
                
                if pred_tensor.shape[-1] < 5:
                    batch_results.append(None)
                    continue
                
                # Extract detection components
                boxes = pred_tensor[:, :4]
                confidence = pred_tensor[:, 4]
                
                # Filter by confidence
                conf_mask = confidence > conf_threshold
                if not conf_mask.any():
                    batch_results.append(None)
                    continue
                
                boxes = boxes[conf_mask]
                confidence = confidence[conf_mask]
                
                # Convert to corner format for NMS
                x_center, y_center, width, height = boxes.unbind(1)
                x1 = x_center - width * 0.5
                y1 = y_center - height * 0.5
                x2 = x_center + width * 0.5
                y2 = y_center + height * 0.5
                corner_boxes = torch.stack([x1, y1, x2, y2], dim=1)
                
                # Get classes
                if pred_tensor.shape[-1] > 5:
                    class_scores = pred_tensor[conf_mask, 5:]
                    class_ids = torch.argmax(class_scores, dim=1)
                else:
                    class_ids = torch.zeros(len(boxes), dtype=torch.long, device=self.device)
                
                # Apply NMS
                if len(corner_boxes) > 0:
                    keep_indices = ops.nms(corner_boxes, confidence, nms_threshold)
                    
                    final_boxes = corner_boxes[keep_indices]
                    final_scores = confidence[keep_indices]
                    final_classes = class_ids[keep_indices]
                    
                    batch_results.append((final_boxes, final_scores, final_classes))
                else:
                    batch_results.append(None)
                    
            except Exception as e:
                # Skip problematic predictions
                batch_results.append(None)
        
        return batch_results
    
    def _save_results(self, results, output_path):
        """Save detection results to CSV"""
        if results:
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
    
    def _print_progress(self, current_frame, total_frames, video_fps):
        """Print detailed progress information"""
        elapsed = time.time() - self.start_time
        processing_fps = self.processed_frames / elapsed if elapsed > 0 else 0
        
        # Calculate completion percentage
        completion = (current_frame / total_frames) * 100 if total_frames > 0 else 0
        
        # Estimate remaining time
        frames_remaining = total_frames - current_frame
        eta_seconds = (frames_remaining / video_fps) / (processing_fps / video_fps) if processing_fps > 0 else 0
        eta_time = timedelta(seconds=int(eta_seconds))
        
        # Current video time
        current_video_time = timedelta(seconds=int(current_frame / video_fps)) if video_fps > 0 else timedelta(0)
        
        # Performance metrics
        avg_inference_time = np.mean(self.inference_times[-100:]) if self.inference_times else 0
        speedup = processing_fps / video_fps if video_fps > 0 else 0
        
        print(f"\n📊 Progress Update:")
        print(f"   Video time: {current_video_time} ({completion:.1f}% complete)")
        print(f"   Processing: {processing_fps:.1f} FPS ({speedup:.1f}x real-time)")
        print(f"   Inference: {avg_inference_time:.1f}ms per batch ({self.batch_size} frames)")
        print(f"   Detections: {self.total_detections:,} total")
        print(f"   ETA: {eta_time}")
    
    def _print_final_stats(self, processed_frames, total_frames, video_fps):
        """Print comprehensive final statistics"""
        total_time = time.time() - self.start_time
        processing_fps = self.processed_frames / total_time
        
        print(f"\n" + "="*60)
        print(f"🎯 PRODUCTION PROCESSING COMPLETE")
        print(f"="*60)
        
        print(f"\n📊 Processing Summary:")
        print(f"   Frames processed: {self.processed_frames:,}/{processed_frames:,}")
        print(f"   Total time: {total_time/3600:.2f} hours")
        print(f"   Processing speed: {processing_fps:.1f} FPS")
        print(f"   Speedup: {processing_fps/video_fps:.1f}x real-time" if video_fps > 0 else "")
        
        print(f"\n⚡ Performance Metrics:")
        if self.inference_times:
            avg_inference = np.mean(self.inference_times)
            print(f"   Avg inference time: {avg_inference:.1f}ms per batch")
            print(f"   Effective FPS: {self.batch_size * 1000 / avg_inference:.1f}")
        
        print(f"\n🐟 Detection Results:")
        print(f"   Total detections: {self.total_detections:,}")
        print(f"   Detections per frame: {self.total_detections/self.processed_frames:.1f}")
        
        # Performance assessment
        target_achieved = processing_fps >= 80
        print(f"\n🎯 Performance Assessment:")
        print(f"   Target (80+ FPS): {'✅ ACHIEVED' if target_achieved else '❌ MISSED'}")
        print(f"   Sustained performance: {'✅ YES' if processing_fps >= 200 else '⚠️ REDUCED'}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Production Batch Video Processing")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--engine", default="models/auklab_model_xlarge_combined_4564_v1_clean.trt", 
                       help="Path to TensorRT engine")
    parser.add_argument("--output", help="Output CSV file (auto-generated if not specified)")
    parser.add_argument("--batch-size", type=int, default=21, 
                       help="Batch size for processing (default: 21)")
    parser.add_argument("--frame-skip", type=int, default=25, 
                       help="Process every Nth frame (default: 25)")
    parser.add_argument("--progress-interval", type=int, default=300, 
                       help="Progress update interval in seconds (default: 300)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video_path):
        print(f"❌ Video file not found: {args.video_path}")
        return
    
    if not os.path.exists(args.engine):
        print(f"❌ TensorRT engine not found: {args.engine}")
        return
    
    # Generate output filename if not specified
    if not args.output:
        video_stem = Path(args.video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results_{video_stem}_{timestamp}.csv"
    
    print(f"🎬 Production Video Processing")
    print(f"   Input: {args.video_path}")
    print(f"   Engine: {args.engine}")
    print(f"   Output: {args.output}")
    print(f"   Batch size: {args.batch_size}")
    
    # Create processor and run
    try:
        processor = ProductionBatchProcessor(args.engine, args.batch_size)
        results = processor.process_full_video(
            video_path=args.video_path,
            output_csv=args.output,
            frame_skip=args.frame_skip,
            progress_interval=args.progress_interval
        )
        
        print(f"\n✅ Processing complete! Results saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import cv2  # Import here to avoid startup issues
    
    if len(sys.argv) == 1:
        # Interactive mode for testing
        print("🧪 Interactive Mode - Testing with available video files")
        
        video_files = list(Path("vid").glob("*.mp4"))
        if video_files:
            video_path = str(video_files[0])
            output_csv = f"production_test_{Path(video_path).stem}.csv"
            
            print(f"Testing with: {video_path}")
            
            try:
                processor = ProductionBatchProcessor(
                    "models/auklab_model_xlarge_combined_4564_v1_clean.trt", 
                    batch_size=21
                )
                
                results = processor.process_full_video(
                    video_path=video_path,
                    output_csv=output_csv,
                    frame_skip=25,
                    progress_interval=60  # More frequent updates for testing
                )
                
                print(f"✅ Test complete! Results in {output_csv}")
                
            except Exception as e:
                print(f"❌ Test failed: {e}")
        else:
            print("❌ No video files found in vid/ directory")
    else:
        main()