#!/usr/bin/env python3
"""
Multi-GPU Object Detection Pipeline
- GPU 0: TensorRT inference only (isolated)
- GPU 1: PyTorch post-processing only (isolated)
- CUDA memory transfer between GPUs for maximum performance
"""

import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import gc
import traceback
from contextlib import contextmanager

# TensorRT imports (GPU 0 only)
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# PyTorch imports (GPU 1 only)
import torch
import torchvision

print("🚀 Multi-GPU Object Detection Pipeline")
print(f"   TensorRT version: {trt.__version__}")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA GPUs available: {torch.cuda.device_count()}")

class MultiGPUInferenceEngine:
    """
    High-performance multi-GPU inference engine
    - GPU 0: TensorRT inference (isolated context)
    - GPU 1: PyTorch post-processing (isolated context)
    """
    
    def __init__(self, engine_path, batch_size=21):
        self.engine_path = Path(engine_path)
        self.batch_size = batch_size
        self.input_shape = (3, 960, 960)  # Match working engine
        self.class_names = ['fish', 'seabird']
        
        # GPU allocation
        self.inference_gpu = 0  # TensorRT on GPU 0
        self.postprocess_gpu = 1  # PyTorch on GPU 1
        
        print(f"🔧 Initializing multi-GPU engine:")
        print(f"   Inference GPU: {self.inference_gpu} (TensorRT)")
        print(f"   Post-processing GPU: {self.postprocess_gpu} (PyTorch)")
        print(f"   Batch size: {self.batch_size}")
        
        # Initialize TensorRT engine on GPU 0
        self._init_tensorrt_engine()
        
        # Initialize PyTorch context on GPU 1
        self._init_pytorch_context()
        
        # Performance tracking
        self.total_inference_time = 0
        self.total_postprocess_time = 0
        self.total_batches = 0
    
    def _init_tensorrt_engine(self):
        """Initialize TensorRT engine on GPU 0 (isolated)"""
        print("🔄 Loading TensorRT engine on GPU 0...")
        
        # Set CUDA device for TensorRT
        cuda.Device(self.inference_gpu).make_context()
        
        # Load TensorRT engine
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # Get tensor names and setup properly
        self.input_name = None
        self.output_names = []
        
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
            
            if is_input:
                self.input_name = tensor_name
            else:
                self.output_names.append(tensor_name)
        
        # Set input shape properly
        input_shape = (self.batch_size, *self.input_shape)
        self.context.set_input_shape(self.input_name, input_shape)
        
        # Allocate GPU memory for TensorRT
        input_shape = (self.batch_size, *self.input_shape)
        input_size = int(np.prod(input_shape) * 4)
        self.d_input = cuda.mem_alloc(input_size)
        
        # Output allocations - get actual shapes from context
        self.d_outputs = {}
        self.h_outputs = {}
        
        print(f"   Found {len(self.output_names)} output tensors:")
        
        for output_name in self.output_names:
            output_shape = self.context.get_tensor_shape(output_name)
            output_shape_tuple = tuple(output_shape[i] for i in range(len(output_shape)))
            print(f"     {output_name}: {output_shape_tuple}")
            
            output_size = int(np.prod(output_shape_tuple) * 4)
            self.d_outputs[output_name] = cuda.mem_alloc(output_size)
            self.h_outputs[output_name] = np.empty(output_shape_tuple, dtype=np.float32)
        
        # Create CUDA stream for TensorRT
        self.trt_stream = cuda.Stream()
        
        cuda.Context.pop()  # Release TensorRT context
        print("✅ TensorRT engine initialized on GPU 0")
    
    def _init_pytorch_context(self):
        """Initialize PyTorch context on GPU 1 (isolated)"""
        print("🔄 Initializing PyTorch context on GPU 1...")
        
        # Set PyTorch to use GPU 1 exclusively
        torch.cuda.set_device(self.postprocess_gpu)
        
        # Pre-allocate common tensors on GPU 1
        with torch.cuda.device(self.postprocess_gpu):
            # Dummy operation to initialize CUDA context
            _ = torch.zeros(1).cuda()
        
        print("✅ PyTorch context initialized on GPU 1")
    
    @contextmanager
    def tensorrt_context(self):
        """Context manager for TensorRT operations on GPU 0"""
        cuda.Device(self.inference_gpu).make_context()
        try:
            yield
        finally:
            cuda.Context.pop()
    
    @contextmanager 
    def pytorch_context(self):
        """Context manager for PyTorch operations on GPU 1"""
        with torch.cuda.device(self.postprocess_gpu):
            yield
    
    def preprocess_batch(self, images):
        """Preprocess batch of images for inference"""
        batch = np.zeros((len(images), *self.input_shape), dtype=np.float32)
        
        for i, img in enumerate(images):
            if img is None:
                continue
                
            # Resize to 960x960 (matching engine input)
            resized = cv2.resize(img, (960, 960))
            normalized = resized.astype(np.float32) / 255.0
            
            # Convert BGR to RGB and transpose
            rgb = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
            transposed = np.transpose(rgb, (2, 0, 1))
            
            batch[i] = transposed
            
        return batch
    
    def run_inference(self, batch_data):
        """Run TensorRT inference on GPU 0"""
        with self.tensorrt_context():
            start_time = time.time()
            
            # Copy input to GPU 0
            cuda.memcpy_htod_async(self.d_input, batch_data, self.trt_stream)
            
            # Set tensor addresses
            self.context.set_tensor_address(self.input_name, int(self.d_input))
            
            for output_name in self.output_names:
                self.context.set_tensor_address(output_name, int(self.d_outputs[output_name]))
            
            # Execute inference
            self.context.execute_async_v3(self.trt_stream.handle)
            
            # Copy outputs from GPU 0
            for output_name in self.output_names:
                cuda.memcpy_dtoh_async(self.h_outputs[output_name], self.d_outputs[output_name], self.trt_stream)
            
            # Synchronize TensorRT stream
            self.trt_stream.synchronize()
            
            inference_time = time.time() - start_time
            self.total_inference_time += inference_time
            
            # Return main detection output (first output)
            main_output_name = self.output_names[0]
            return self.h_outputs[main_output_name]
    
    def gpu_postprocess(self, raw_output, batch_size):
        """GPU-accelerated post-processing on GPU 1"""
        with self.pytorch_context():
            start_time = time.time()
            
            # Convert to PyTorch tensor on GPU 1
            detections_tensor = torch.from_numpy(raw_output[:batch_size]).cuda(self.postprocess_gpu)
            
            # Efficient batch processing
            batch_results = []
            
            for batch_idx in range(batch_size):
                frame_detections = detections_tensor[batch_idx]  # Shape: (84, 8400)
                
                # Extract boxes, scores, classes efficiently
                boxes = frame_detections[:4].T  # (8400, 4)
                scores_classes = frame_detections[4:].T  # (8400, 80)
                
                # Get max scores and classes
                max_scores, class_indices = torch.max(scores_classes, dim=1)
                
                # Filter by confidence threshold
                conf_mask = max_scores > 0.25
                if not conf_mask.any():
                    batch_results.append([])
                    continue
                
                # Apply mask
                filtered_boxes = boxes[conf_mask]
                filtered_scores = max_scores[conf_mask]
                filtered_classes = class_indices[conf_mask]
                
                # Convert to format for NMS: [x1, y1, x2, y2]
                x_center, y_center, width, height = filtered_boxes.T
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                nms_boxes = torch.stack([x1, y1, x2, y2], dim=1)
                
                # GPU NMS using torchvision
                keep_indices = torchvision.ops.nms(nms_boxes, filtered_scores, iou_threshold=0.45)
                
                # Final detections
                final_boxes = nms_boxes[keep_indices]
                final_scores = filtered_scores[keep_indices]
                final_classes = filtered_classes[keep_indices]
                
                # Convert back to CPU as numpy for compatibility
                detections = []
                for i in range(len(keep_indices)):
                    x1, y1, x2, y2 = final_boxes[i].cpu().numpy()
                    score = final_scores[i].cpu().item()
                    cls = final_classes[i].cpu().item()
                    
                    detections.append({
                        'x1': float(x1), 'y1': float(y1),
                        'x2': float(x2), 'y2': float(y2),
                        'confidence': float(score),
                        'class': int(cls),
                        'class_name': self.class_names[int(cls)] if int(cls) < len(self.class_names) else f'class_{int(cls)}'
                    })
                
                batch_results.append(detections)
            
            postprocess_time = time.time() - start_time
            self.total_postprocess_time += postprocess_time
            
            return batch_results
    
    def process_batch(self, images, frame_indices):
        """Process a batch of images using multi-GPU pipeline"""
        self.total_batches += 1
        
        # Preprocess on CPU
        batch_data = self.preprocess_batch(images)
        actual_batch_size = len(images)
        
        # Inference on GPU 0
        raw_output = self.run_inference(batch_data)
        
        # Post-processing on GPU 1
        batch_results = self.gpu_postprocess(raw_output, actual_batch_size)
        
        # Format results
        all_detections = []
        for i, (detections, frame_idx) in enumerate(zip(batch_results, frame_indices)):
            for det in detections:
                detection = {
                    'frame_number': frame_idx,
                    'timestamp': frame_idx / 30.0,  # Assuming 30 FPS
                    **det
                }
                all_detections.append(detection)
        
        return all_detections
    
    def get_performance_stats(self):
        """Get detailed performance statistics"""
        if self.total_batches == 0:
            return {}
        
        avg_inference = self.total_inference_time / self.total_batches
        avg_postprocess = self.total_postprocess_time / self.total_batches
        total_avg = avg_inference + avg_postprocess
        
        return {
            'avg_inference_time': avg_inference,
            'avg_postprocess_time': avg_postprocess,
            'avg_total_time': total_avg,
            'inference_fps': self.batch_size / avg_inference if avg_inference > 0 else 0,
            'postprocess_fps': self.batch_size / avg_postprocess if avg_postprocess > 0 else 0,
            'total_fps': self.batch_size / total_avg if total_avg > 0 else 0,
            'inference_percentage': (avg_inference / total_avg * 100) if total_avg > 0 else 0,
            'postprocess_percentage': (avg_postprocess / total_avg * 100) if total_avg > 0 else 0
        }

def process_video_multigpu(video_path, output_dir="dump", batch_size=21):
    """Process video using multi-GPU pipeline"""
    
    # Setup paths
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    engine_path = Path("models/auklab_model_xlarge_combined_4564_v1_batch_fixed.trt")
    
    if not engine_path.exists():
        print(f"❌ Engine file not found: {engine_path}")
        return None
    
    print(f"🎬 Processing video: {video_path.name}")
    print(f"📁 Output directory: {output_dir}")
    
    # Initialize multi-GPU engine
    try:
        engine = MultiGPUInferenceEngine(engine_path, batch_size)
    except Exception as e:
        print(f"❌ Failed to initialize multi-GPU engine: {e}")
        traceback.print_exc()
        return None
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📹 Video info:")
    print(f"   Total frames: {total_frames:,}")
    print(f"   FPS: {fps}")
    print(f"   Duration: {total_frames/fps:.1f} seconds")
    
    # Processing variables
    all_detections = []
    frame_count = 0
    batch_images = []
    batch_indices = []
    
    start_time = time.time()
    last_progress_time = start_time
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            batch_images.append(frame)
            batch_indices.append(frame_count)
            frame_count += 1
            
            # Process batch when full or at end of video
            if len(batch_images) == batch_size or frame_count == total_frames:
                try:
                    # Process batch using multi-GPU pipeline
                    detections = engine.process_batch(batch_images, batch_indices)
                    all_detections.extend(detections)
                    
                    # Progress reporting
                    current_time = time.time()
                    if current_time - last_progress_time >= 10.0:  # Every 10 seconds
                        elapsed = current_time - start_time
                        progress = frame_count / total_frames * 100
                        current_fps = frame_count / elapsed if elapsed > 0 else 0
                        
                        print(f"📊 Progress: {progress:.1f}% | {current_fps:.1f} FPS | {len(all_detections):,} detections")
                        print(f"   Multi-GPU acceleration: ✅ GPU0(TRT) + GPU1(PyTorch)")
                        
                        # Performance breakdown
                        stats = engine.get_performance_stats()
                        if stats:
                            print(f"   Inference: {stats['inference_fps']:.1f} FPS ({stats['inference_percentage']:.1f}%)")
                            print(f"   Post-proc: {stats['postprocess_fps']:.1f} FPS ({stats['postprocess_percentage']:.1f}%)")
                        
                        last_progress_time = current_time
                    
                except Exception as e:
                    print(f"❌ Error processing batch at frame {frame_count}: {e}")
                    traceback.print_exc()
                    continue
                
                # Clear batch
                batch_images = []
                batch_indices = []
                
                # Memory cleanup
                gc.collect()
                torch.cuda.empty_cache()
        
        # Final processing
        total_time = time.time() - start_time
        final_fps = frame_count / total_time if total_time > 0 else 0
        
        print(f"\n💾 Saving {len(all_detections):,} detections...")
        
        # Save results
        if all_detections:
            output_file = output_dir / f"{video_path.stem}_multigpu.csv"
            df = pd.DataFrame(all_detections)
            df.to_csv(output_file, index=False)
            
            print(f"✅ Saved to: {output_file}")
        else:
            print("⚠️ No detections found")
            output_file = None
        
        # Final performance report
        stats = engine.get_performance_stats()
        print(f"\n🎯 MULTI-GPU PROCESSING COMPLETE!")
        print(f"   Frames processed: {frame_count:,}")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Average FPS: {final_fps:.1f}")
        print(f"   Total detections: {len(all_detections):,}")
        print(f"   Multi-GPU acceleration: ✅ Isolated GPU contexts")
        
        if stats:
            print(f"   Performance breakdown:")
            print(f"     Inference (GPU 0): {stats['inference_fps']:.1f} FPS ({stats['inference_percentage']:.1f}%)")
            print(f"     Post-proc (GPU 1): {stats['postprocess_fps']:.1f} FPS ({stats['postprocess_percentage']:.1f}%)")
            print(f"     Combined pipeline: {stats['total_fps']:.1f} FPS")
        
        if final_fps >= 20:
            print(f"   Performance: ✅ EXCELLENT")
        elif final_fps >= 10:
            print(f"   Performance: ✅ GOOD")
        else:
            print(f"   Performance: ⚠️ NEEDS OPTIMIZATION")
        
        return output_file
        
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
        return None
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        return None
    
    finally:
        cap.release()
        print("🔄 Cleanup completed")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python production_batch_multigpu.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        sys.exit(1)
    
    # Check GPU availability
    if torch.cuda.device_count() < 2:
        print("❌ This script requires 2 GPUs for optimal performance")
        print(f"   Available GPUs: {torch.cuda.device_count()}")
        print("   Consider using the single-GPU version instead")
        sys.exit(1)
    
    print(f"🚀 Starting multi-GPU processing with {torch.cuda.device_count()} GPUs")
    
    result = process_video_multigpu(video_path)
    
    if result:
        print(f"\n🎉 SUCCESS! Results saved to: {result}")
    else:
        print(f"\n❌ Processing failed or was interrupted")