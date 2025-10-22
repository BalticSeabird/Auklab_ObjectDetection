#!/usr/bin/env python3
"""
Clean Production Batch Inference - Working CPU-only baseline
No PyTorch tensors to avoid TensorRT corruption
Simple, reliable, working implementation
Usage: python3 production_batch_simple_clean.py <video_path> [output_csv]
"""

import pandas as pd
from pathlib import Path
import os
import time
import sys
import numpy as np
import cv2

# TensorRT imports
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Configuration
BATCH_SIZE = 21
FRAME_SKIP = 25
CONFIDENCE_THRESHOLD = 0.25
ENGINE_PATH = "models/auklab_model_xlarge_combined_4564_v1_batch.trt"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class SimpleBatchProcessor:
    """Clean, simple batch processor - CPU only, no PyTorch"""
    
    def __init__(self, engine_path, batch_size=21):
        self.batch_size = batch_size
        
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
        
        # Output memory
        self.context.set_input_shape(self.input_name, input_shape)
        
        self.d_outputs = {}
        self.h_outputs = {}
        
        print(f"   Found {len(self.output_names)} output tensors:")
        
        for output_name in self.output_names:
            output_shape = self.context.get_tensor_shape(output_name)
            output_shape_tuple = tuple(output_shape[i] for i in range(len(output_shape)))
            print(f"     {output_name}: {output_shape_tuple}")
            
            output_size = int(np.prod(output_shape_tuple) * 4)
            self.d_outputs[output_name] = cuda.mem_alloc(output_size)
            self.h_outputs[output_name] = cuda.pagelocked_empty(output_shape_tuple, dtype=np.float32)
        
        total_memory = input_size + sum(np.prod(self.h_outputs[name].shape) * 4 for name in self.output_names)
        print(f"   Memory allocated: {total_memory / (1024**2):.1f} MB")
    
    def process_video(self, video_path, output_csv, frame_skip=25):
        """Process video with CPU-only post-processing"""
        
        print(f"\n📹 Processing: {video_path}")
        print(f"   Frame skip: {frame_skip}")
        print(f"   Output: {output_csv}")
        
        self.start_time = time.time()
        last_update = self.start_time
        
        results_list = []
        frame_buffer = []
        frame_indices = []
        frame_count = 0
        
        # Use OpenCV for simplicity and reliability
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"   Video: {total_frames:,} frames at {fps:.1f} FPS")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                frame_buffer.append(frame)
                frame_indices.append(frame_count)
                
                if len(frame_buffer) == self.batch_size:
                    detections = self._process_batch_cpu_only(frame_buffer, frame_indices)
                    results_list.extend(detections)
                    frame_buffer = []
                    frame_indices = []
                    
                    if time.time() - last_update > 60:
                        self._quick_progress(frame_count, total_frames)
                        last_update = time.time()
            
            frame_count += 1
        
        if frame_buffer:
            detections = self._process_batch_cpu_only(frame_buffer, frame_indices)
            results_list.extend(detections)
        
        cap.release()
        
        # Save results
        if results_list:
            output_dir = os.path.dirname(output_csv)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            df = pd.DataFrame(results_list)
            df.to_csv(output_csv, index=False)
            print(f"\n💾 Saved {len(results_list):,} detections to {output_csv}")
        else:
            print(f"\n⚠️  No detections found - CSV file not created")
        
        self._final_stats()
        return results_list
    
    def _process_batch_cpu_only(self, frames, indices):
        """CPU-only batch processing - no PyTorch tensors"""
        actual_batch_size = len(frames)
        
        # Preprocess frames
        for i, img in enumerate(frames):
            if img.shape[:2] != (960, 960):
                img = cv2.resize(img, (960, 960))
            
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            self.h_input[i] = img
        
        # Set shape and execute
        input_shape = (actual_batch_size, 3, 960, 960)
        self.context.set_input_shape(self.input_name, input_shape)
        
        # Copy input data
        cuda.memcpy_htod(self.d_input, self.h_input[:actual_batch_size])
        
        # Set tensor addresses
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        for output_name in self.output_names:
            self.context.set_tensor_address(output_name, int(self.d_outputs[output_name]))
        
        # Execute inference
        success = self.context.execute_async_v3(0)
        
        if not success:
            return []
        
        # Get results from main detection output
        main_output_name = self.output_names[0]
        cuda.memcpy_dtoh(self.h_outputs[main_output_name][:actual_batch_size], self.d_outputs[main_output_name])
        
        # Process detections with pure NumPy - no PyTorch tensors
        predictions = self.h_outputs[main_output_name][:actual_batch_size]
        results = []
        
        for i, frame_idx in enumerate(indices):
            # Get predictions for this frame
            frame_pred = predictions[i]  # Shape: (7, 18900)
            
            # Transpose to (18900, 7) for easier processing
            detections = frame_pred.T  # Shape: (18900, 7)
            
            # Confidence filtering using NumPy
            confidences = detections[:, 4]
            conf_mask = confidences > CONFIDENCE_THRESHOLD
            
            if np.any(conf_mask):
                valid_dets = detections[conf_mask]
                
                if len(valid_dets) > 0:
                    # Simple CPU-only NMS
                    final_dets = self._cpu_nms(valid_dets)
                    
                    for det in final_dets:
                        x_center, y_center, width, height = det[0], det[1], det[2], det[3]
                        confidence = det[4]
                        
                        # Convert center format to corner format
                        xmin = max(0, x_center - width/2)
                        ymin = max(0, y_center - height/2)
                        xmax = min(960, x_center + width/2)
                        ymax = min(960, y_center + height/2)
                        
                        # Get class
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
    
    def _cpu_nms(self, detections, iou_threshold=0.5):
        """CPU-only Non-Maximum Suppression using NumPy"""
        if len(detections) == 0:
            return []
        
        # Extract coordinates and scores
        x_center = detections[:, 0]
        y_center = detections[:, 1]
        width = detections[:, 2]
        height = detections[:, 3]
        scores = detections[:, 4]
        
        # Convert to corner format
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # Calculate areas
        areas = width * height
        
        # Sort by scores
        order = scores.argsort()[::-1]
        
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            
            iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)
            
            # Keep only boxes with IoU below threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return detections[keep]
    
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
    if len(sys.argv) < 2:
        print("Usage: python3 production_batch_simple_clean.py <video_path> [output_csv]")
        return
    
    video_path = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_csv = sys.argv[2]
    else:
        video_name = Path(video_path).stem
        output_csv = f"dump/{video_name}.csv"
    
    engine_path = ENGINE_PATH
    
    if not os.path.exists(engine_path):
        print(f"❌ Engine not found: {engine_path}")
        return
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    os.makedirs("dump", exist_ok=True)
    
    print(f"🎬 Starting clean CPU-only processing...")
    print(f"   Video: {video_path}")
    print(f"   Engine: {engine_path}")
    print(f"   Output CSV: {output_csv}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Frame skip: {FRAME_SKIP}")
    
    try:
        processor = SimpleBatchProcessor(engine_path, batch_size=BATCH_SIZE)
        results = processor.process_video(video_path, output_csv, frame_skip=FRAME_SKIP)
        
        print(f"\n✅ SUCCESS! Results saved to {output_csv}")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()