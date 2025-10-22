#!/usr/bin/env python3
"""
Production Batch Inference - 1 Hour Video Processing
Based on your proven PyAV approach + new batch TensorRT engine
Usage: python3 production_batch_simple.py <video_path> [output_csv]
"""

# ============================================================================
# CONFIGURATION SECTION - Edit these paths as needed
# ============================================================================

# Input video file path
INPUT_VIDEO_PATH = "vid/input.mp4"  # Change this to your video file

# Output files
OUTPUT_CSV_PATH = "dump/batch_results.csv"  # Detection results
OUTPUT_VIDEO_PATH = None  # Set to save annotated video (optional, None = no video output)

# TensorRT engine path
ENGINE_PATH = "models/auklab_model_xlarge_combined_4564_v1_batch.trt"

# Processing settings
BATCH_SIZE = 21  # Optimal batch size (found to give 323.2 FPS)
FRAME_SKIP = 25  # Process every Nth frame (25 = ~1 FPS for 25 FPS video)
CONFIDENCE_THRESHOLD = 0.25  # Detection confidence threshold

# ============================================================================

import pandas as pd
from pathlib import Path
import os
import time
import sys
import numpy as np
from datetime import datetime, timedelta

# Try to import PyAV, fall back to OpenCV if not available
try:
    import av
    PYAV_AVAILABLE = True
    print("✅ Using PyAV for video decoding")
except ImportError:
    import cv2
    PYAV_AVAILABLE = False
    print("⚠️  PyAV not available, using OpenCV")

# Import TensorRT components
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    import torch
    import torchvision.ops as ops
    TRT_AVAILABLE = True
except ImportError as e:
    print(f"❌ TensorRT/CUDA not available: {e}")
    TRT_AVAILABLE = False
    sys.exit(1)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class FastBatchProcessor:
    """Simple, fast batch processor for long videos"""
    
    def __init__(self, engine_path, batch_size=21):
        self.batch_size = batch_size
        self.device = torch.device('cuda:0')
        
        print(f"🚀 Loading TensorRT Engine: {engine_path}")
        
        # Load engine
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Get tensor names
        self.input_name = None
        self.output_names = []
        
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
            
            if is_input:
                self.input_name = tensor_name
            else:
                self.output_names.append(tensor_name)
        
        # Pre-allocate memory
        self._setup_memory()
        
        # Performance tracking
        self.processed_frames = 0
        self.total_detections = 0
        self.start_time = None
        
        print(f"✅ Ready for processing with batch size {batch_size}")
    
    def _setup_memory(self):
        """Setup GPU memory for optimal batch size"""
        # Input memory
        input_shape = (self.batch_size, 3, 960, 960)
        input_size = int(np.prod(input_shape) * 4)
        
        self.d_input = cuda.mem_alloc(input_size)
        self.h_input = cuda.pagelocked_empty(input_shape, dtype=np.float32)
        
        # Output memory - ALLOCATE FOR ALL OUTPUTS
        self.context.set_input_shape(self.input_name, input_shape)
        
        # Pre-allocate memory for ALL outputs
        self.d_outputs = {}
        self.h_outputs = {}
        
        print(f"   Found {len(self.output_names)} output tensors:")
        
        for output_name in self.output_names:
            try:
                output_shape = self.context.get_tensor_shape(output_name)
                output_shape_tuple = tuple(output_shape[i] for i in range(len(output_shape)))
                print(f"     {output_name}: {output_shape_tuple}")
            except:
                # Fallback shapes for common outputs
                if "Shape" in output_name:
                    output_shape_tuple = (self.batch_size, 2)  # Shape output
                else:
                    output_shape_tuple = (self.batch_size, 8400, 85)  # Main detection output
                print(f"     {output_name}: {output_shape_tuple} (fallback)")
            
            output_size = int(np.prod(output_shape_tuple) * 4)
            self.d_outputs[output_name] = cuda.mem_alloc(output_size)
            self.h_outputs[output_name] = cuda.pagelocked_empty(output_shape_tuple, dtype=np.float32)
        
        total_memory = input_size + sum(np.prod(self.h_outputs[name].shape) * 4 for name in self.output_names)
        print(f"   Memory allocated: {total_memory / (1024**2):.1f} MB")
    
    def process_video_fast(self, video_path, output_csv, frame_skip=25):
        """Process video as fast as possible"""
        
        print(f"\n📹 Processing: {video_path}")
        print(f"   Frame skip: {frame_skip}")
        print(f"   Output: {output_csv}")
        
        self.start_time = time.time()
        last_update = self.start_time
        
        results_list = []
        frame_buffer = []
        frame_indices = []
        frame_count = 0
        
        if PYAV_AVAILABLE:
            # Use your proven PyAV approach with better error handling
            try:
                container = av.open(str(video_path))
                
                # Check if video streams exist
                if len(container.streams.video) == 0:
                    print("⚠️  No video streams found in file, switching to OpenCV...")
                    container.close()
                    # Fall through to OpenCV processing
                else:
                    stream = container.streams.video[0]
                    stream.thread_type = 'AUTO'
                    
                    total_frames = stream.frames if hasattr(stream, 'frames') else 0
                    fps = float(stream.average_rate) if hasattr(stream, 'average_rate') else 25
                    
                    print(f"   Video: {total_frames:,} frames at {fps:.1f} FPS")
                    
                    try:
                        for frame in container.decode(stream):
                            if frame_count % frame_skip == 0:
                                img = frame.to_ndarray(format='bgr24')
                                frame_buffer.append(img)
                                frame_indices.append(frame_count)
                                
                                # Process batch when full
                                if len(frame_buffer) == self.batch_size:
                                    detections = self._process_batch_fast(frame_buffer, frame_indices)
                                    results_list.extend(detections)
                                    frame_buffer = []
                                    frame_indices = []
                                    
                                    # Quick progress update
                                    if time.time() - last_update > 60:  # Every minute
                                        self._quick_progress(frame_count, total_frames)
                                        last_update = time.time()
                            
                            frame_count += 1
                        
                        # Process remaining frames
                        if frame_buffer:
                            detections = self._process_batch_fast(frame_buffer, frame_indices)
                            results_list.extend(detections)
                        
                        container.close()
                        # Continue to save results (don't return here)
                        
                    except Exception as e:
                        print(f"⚠️  PyAV processing failed: {e}")
                        container.close()
                        # Fall through to OpenCV processing
                        
            except Exception as e:
                print(f"⚠️  PyAV cannot open file: {e}")
                print("   Switching to OpenCV...")
                # Fall through to OpenCV processing
        
        # OpenCV fallback (only if PyAV failed or not available)
        if not PYAV_AVAILABLE or len(results_list) == 0:
            # OpenCV fallback
            print("   Using OpenCV video processing...")
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_skip == 0:
                    frame_buffer.append(frame)
                    frame_indices.append(frame_count)
                    
                    if len(frame_buffer) == self.batch_size:
                        detections = self._process_batch_fast(frame_buffer, frame_indices)
                        results_list.extend(detections)
                        frame_buffer = []
                        frame_indices = []
                        
                        if time.time() - last_update > 60:
                            self._quick_progress(frame_count, total_frames)
                            last_update = time.time()
                
                frame_count += 1
            
            if frame_buffer:
                detections = self._process_batch_fast(frame_buffer, frame_indices)
                results_list.extend(detections)
            
            cap.release()
        
        # Save results
        if results_list:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_csv)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            df = pd.DataFrame(results_list)
            df.to_csv(output_csv, index=False)
            print(f"\n💾 Saved {len(results_list):,} detections to {output_csv}")
        else:
            print(f"\n⚠️  No detections found - CSV file not created")
        
        # Final stats
        self._final_stats()
        
        return results_list
    
    def _process_batch_fast(self, frames, indices):
        """Hybrid CPU-GPU batch processing - stable TensorRT + GPU post-processing"""
        actual_batch_size = len(frames)
        
        # Preprocess frames quickly
        for i, img in enumerate(frames):
            if img.shape[:2] != (960, 960):
                import cv2
                img = cv2.resize(img, (960, 960))
            
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            self.h_input[i] = img
        
        # Set shape and execute
        input_shape = (actual_batch_size, 3, 960, 960)
        self.context.set_input_shape(self.input_name, input_shape)
        
        # Copy input data
        cuda.memcpy_htod(self.d_input, self.h_input[:actual_batch_size])
        
        # Set tensor addresses for input
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        
        # Set tensor addresses for ALL outputs
        for output_name in self.output_names:
            self.context.set_tensor_address(output_name, int(self.d_outputs[output_name]))
        
        # Execute inference (keep this CPU-based for stability)
        success = self.context.execute_async_v3(0)
        
        if not success:
            return []
        
        # Get results from main detection output (first output)
        main_output_name = self.output_names[0]
        cuda.memcpy_dtoh(self.h_outputs[main_output_name][:actual_batch_size], self.d_outputs[main_output_name])
        
        # HYBRID APPROACH: Copy to GPU for post-processing only
        predictions = self.h_outputs[main_output_name][:actual_batch_size]
        results = []
        
        # Process each frame - use vectorized NumPy for speed
        for i, frame_idx in enumerate(indices):
            # Get predictions for this frame: (7, 18900)
            frame_pred = predictions[i]  # Shape: (7, 18900)
            
            # Transpose to (18900, 7) for easier processing
            detections = frame_pred.T  # Shape: (18900, 7)
            
            # Vectorized confidence filtering using NumPy
            confidences = detections[:, 4]  # Confidence scores
            conf_mask = confidences > CONFIDENCE_THRESHOLD
            
            if np.any(conf_mask):
                # Filter valid detections
                valid_dets = detections[conf_mask]  # Shape: (N, 7)
                
                if len(valid_dets) > 0:
                    # Convert to PyTorch for GPU NMS (only for NMS step)
                    gpu_dets = torch.from_numpy(valid_dets).to(self.device)
                    
                    # Extract coordinates and convert from center to corner format
                    x_center = gpu_dets[:, 0]
                    y_center = gpu_dets[:, 1] 
                    width = gpu_dets[:, 2]
                    height = gpu_dets[:, 3]
                    
                    # Convert to corner format (GPU vectorized)
                    x1 = torch.clamp(x_center - width / 2, min=0)
                    y1 = torch.clamp(y_center - height / 2, min=0)
                    x2 = torch.clamp(x_center + width / 2, max=960)
                    y2 = torch.clamp(y_center + height / 2, max=960)
                    
                    # Stack boxes for NMS: (N, 4)
                    boxes = torch.stack([x1, y1, x2, y2], dim=1)
                    scores = gpu_dets[:, 4]
                    
                    # Get class predictions (argmax over class scores)
                    if gpu_dets.shape[1] > 5:
                        class_scores = gpu_dets[:, 5:]  # Shape: (N, num_classes)
                        class_ids = torch.argmax(class_scores, dim=1)
                    else:
                        class_ids = torch.zeros(len(gpu_dets), dtype=torch.long, device=self.device)
                    
                    # GPU-accelerated NMS using PyTorch
                    if len(boxes) > 0:
                        try:
                            nms_indices = ops.nms(boxes, scores, iou_threshold=0.5)
                            
                            # Extract final detections
                            final_boxes = boxes[nms_indices]
                            final_scores = scores[nms_indices] 
                            final_classes = class_ids[nms_indices]
                            
                            # Convert back to CPU for result storage (minimal data transfer)
                            final_boxes_cpu = final_boxes.cpu().numpy()
                            final_scores_cpu = final_scores.cpu().numpy()
                            final_classes_cpu = final_classes.cpu().numpy()
                            
                            # Create detection entries
                            for j in range(len(final_boxes_cpu)):
                                results.append({
                                    'frame': frame_idx,
                                    'class': int(final_classes_cpu[j]),
                                    'confidence': float(final_scores_cpu[j]),
                                    'xmin': float(final_boxes_cpu[j][0]),
                                    'ymin': float(final_boxes_cpu[j][1]),
                                    'xmax': float(final_boxes_cpu[j][2]),
                                    'ymax': float(final_boxes_cpu[j][3])
                                })
                                
                                self.total_detections += 1
                        except Exception as e:
                            # Fallback to CPU processing if GPU NMS fails
                            print(f"GPU NMS failed, using CPU fallback: {e}")
                            # Fall back to simple CPU processing for this frame
                            for det in valid_dets:
                                x_center, y_center, width, height = det[0], det[1], det[2], det[3]
                                confidence = det[4]
                                
                                # Convert center format to corner format
                                xmin = max(0, x_center - width/2)
                                ymin = max(0, y_center - height/2)
                                xmax = min(960, x_center + width/2)
                                ymax = min(960, y_center + height/2)
                                
                                # Find class with highest score (from index 5 onwards)
                                if det.shape[0] > 5:
                                    class_scores = det[5:]
                                    class_id = np.argmax(class_scores)
                                else:
                                    class_id = 0
                                
                                results.append({
                                    'frame': frame_idx,
                                    'class': int(class_id),
                                    'confidence': float(confidence),
                                    'xmin': float(xmin),
                                    'ymin': float(ymin), 
                                    'xmax': float(xmax),
                                    'ymax': float(ymax)
                                })
                                
                                self.total_detections += 1
        
        self.processed_frames += actual_batch_size
        return results
    
    def _quick_progress(self, current_frame, total_frames):
        """Quick progress update"""
        elapsed = time.time() - self.start_time
        fps = self.processed_frames / elapsed if elapsed > 0 else 0
        
        completion = (current_frame / total_frames * 100) if total_frames > 0 else 0
        
        print(f"📊 Progress: {completion:.1f}% | {fps:.1f} FPS | {self.total_detections:,} detections")
    
    def _final_stats(self):
        """Final performance summary"""
        elapsed = time.time() - self.start_time
        fps = self.processed_frames / elapsed
        
        print(f"\n🎯 PROCESSING COMPLETE!")
        print(f"   Frames processed: {self.processed_frames:,}")
        print(f"   Total time: {elapsed/60:.1f} minutes")
        print(f"   Average FPS: {fps:.1f}")
        print(f"   Total detections: {self.total_detections:,}")
        print(f"   Performance: {'✅ EXCELLENT' if fps > 200 else '⚠️ REDUCED' if fps > 80 else '❌ SLOW'}")


def main():
    """Main execution"""
    # Use command line arguments if provided, otherwise use configuration
    if len(sys.argv) < 2:
        print("Usage: python3 production_batch_simple.py <video_path> [output_csv]")
        print("Or edit the configuration section at the top of this script")
        
        # Use configuration defaults
        video_path = INPUT_VIDEO_PATH
        
        # Generate output CSV name based on input video
        video_name = Path(video_path).stem  # Get filename without extension
        output_csv = f"dump/{video_name}.csv"
        
        print(f"🧪 Using configured paths:")
        print(f"   Video: {video_path}")
        print(f"   Output CSV: {output_csv}")
    else:
        video_path = sys.argv[1]
        
        # Output CSV
        if len(sys.argv) >= 3:
            output_csv = sys.argv[2]
        else:
            # Generate output CSV name based on input video filename
            video_name = Path(video_path).stem  # Get filename without extension
            output_csv = f"dump/{video_name}.csv"
    
    # Engine path (always use configuration)
    engine_path = ENGINE_PATH
    
    if not os.path.exists(engine_path):
        print(f"❌ Engine not found: {engine_path}")
        print(f"   Please check ENGINE_PATH in configuration section")
        return
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        print(f"   Please check INPUT_VIDEO_PATH in configuration section or provide valid command line argument")
        return
    
    # Ensure dump directory exists
    os.makedirs("dump", exist_ok=True)
    
    print(f"🎬 Starting production batch processing...")
    print(f"   Video: {video_path}")
    print(f"   Engine: {engine_path}")
    print(f"   Output CSV: {output_csv}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Frame skip: {FRAME_SKIP}")
    
    try:
        processor = FastBatchProcessor(engine_path, batch_size=BATCH_SIZE)
        results = processor.process_video_fast(video_path, output_csv, frame_skip=FRAME_SKIP)
        
        print(f"\n✅ SUCCESS! Results saved to {output_csv}")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()