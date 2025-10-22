#!/usr/bin/env python3
"""
Dual-GPU Production Batch Inference with DALI GPU Decoding and Native GPU NMS
Optimized for dual RTX 4090 setup with load balancing within single process
"""

import time
import argparse
import csv
import numpy as np
import os
import threading
import queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# DALI imports
import nvidia.dali as dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

# TensorRT imports
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Initialize CUDA contexts for both GPUs
cuda.init()
gpu0_ctx = cuda.Device(0).make_context()
gpu1_ctx = cuda.Device(1).make_context()

class DualGPUInferenceEngine:
    """TensorRT inference engine for dual-GPU load balancing"""
    
    def __init__(self, engine_path, gpu_id, batch_size=8):
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        self.device_id = gpu_id
        
        print(f"   🚀 Initializing GPU {gpu_id} engine: {engine_path}")
        
        # Set correct GPU context
        if gpu_id == 0:
            gpu0_ctx.push()
        else:
            gpu1_ctx.push()
        
        try:
            # Load TensorRT engine
            trt_logger = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            
            runtime = trt.Runtime(trt_logger)
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            self.context = self.engine.create_execution_context()
            
            # Create CUDA stream for this GPU
            self.stream = cuda.Stream()
            
            # Pre-allocate GPU memory
            self._allocate_memory()
            
            print(f"   ✅ GPU {gpu_id} engine ready")
            
        finally:
            if gpu_id == 0:
                gpu0_ctx.pop()
            else:
                gpu1_ctx.pop()
    
    def _allocate_memory(self):
        """Pre-allocate GPU memory for inference"""
        # Get tensor shapes
        input_shape = (self.batch_size, 3, 960, 960)
        output_shape = (self.batch_size, 300, 6)  # Native NMS output
        
        # Calculate memory sizes
        input_size = int(np.prod(input_shape) * 4)  # float32
        output_size = int(np.prod(output_shape) * 4)
        
        # Allocate GPU memory
        self.input_gpu = cuda.mem_alloc(input_size)
        self.output_gpu = cuda.mem_alloc(output_size)
        
        # Pre-allocate host memory
        self.output_host = np.empty(output_shape, dtype=np.float32)
        
        print(f"   💾 GPU {self.gpu_id} memory allocated: {(input_size + output_size)//1024//1024:.1f}MB")
    
    def infer_batch(self, batch_frames):
        """Run inference on a batch of frames"""
        # Set correct GPU context
        if self.gpu_id == 0:
            gpu0_ctx.push()
        else:
            gpu1_ctx.push()
        
        try:
            actual_batch_size = len(batch_frames)
            
            # Prepare input data
            if actual_batch_size < self.batch_size:
                # Pad batch to expected size
                padded_batch = np.zeros((self.batch_size, 3, 960, 960), dtype=np.float32)
                padded_batch[:actual_batch_size] = batch_frames
                input_data = padded_batch
            else:
                input_data = batch_frames
            
            # Ensure contiguous array
            input_data = np.ascontiguousarray(input_data.astype(np.float32))
            
            # Copy to GPU
            cuda.memcpy_htod_async(self.input_gpu, input_data, self.stream)
            
            # Set tensor addresses
            self.context.set_input_shape("images", input_data.shape)
            self.context.set_tensor_address("images", int(self.input_gpu))
            self.context.set_tensor_address("output0", int(self.output_gpu))
            
            # Run inference
            start_time = time.time()
            self.context.execute_async_v3(self.stream.handle)
            self.stream.synchronize()
            inference_time = (time.time() - start_time) * 1000
            
            # Copy result back
            cuda.memcpy_dtoh_async(self.output_host, self.output_gpu, self.stream)
            self.stream.synchronize()
            
            # Return only the actual batch results
            return self.output_host[:actual_batch_size], inference_time
            
        finally:
            if self.gpu_id == 0:
                gpu0_ctx.pop()
            else:
                gpu1_ctx.pop()

class DualGPUProductionProcessor:
    """Dual-GPU production processor with load balancing"""
    
    def __init__(self, engine_path, batch_size=8):
        self.batch_size = batch_size
        self.processed_frames = 0
        self.total_detections = 0
        self.inference_times = []
        self.start_time = None
        
        print(f"🚀 Initializing Dual-GPU Production Processor")
        print(f"   Engine: {engine_path}")
        print(f"   Batch size: {batch_size}")
        
        # Initialize both GPU engines
        self.gpu0_engine = DualGPUInferenceEngine(engine_path, 0, batch_size)
        self.gpu1_engine = DualGPUInferenceEngine(engine_path, 1, batch_size)
        
        # Create thread-safe queues for work distribution
        self.work_queue = queue.Queue(maxsize=20)  # Limit queue size
        self.result_queue = queue.Queue()
        
        # Initialize DALI processor
        self.dali_processor = DALIVideoProcessor(batch_size, device_id=0)
        
        print("✅ Dual-GPU processor ready!")
    
    def _gpu_worker(self, gpu_engine, worker_id):
        """Worker function for GPU inference"""
        while True:
            try:
                work_item = self.work_queue.get(timeout=1.0)
                if work_item is None:  # Shutdown signal
                    break
                
                batch_frames, frame_indices = work_item
                
                # Run inference
                predictions, inference_time = gpu_engine.infer_batch(batch_frames)
                
                # Process predictions
                batch_detections = self._process_native_nms_output(predictions)
                
                # Put results in result queue
                self.result_queue.put((frame_indices, batch_detections, inference_time, worker_id))
                
                self.work_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"   ⚠️ GPU {worker_id} worker error: {e}")
                self.work_queue.task_done()
    
    def _process_native_nms_output(self, predictions, conf_threshold=0.25):
        """Process output from TensorRT engine with native GPU NMS"""
        batch_results = []
        
        for pred in predictions:
            try:
                if pred is None or pred.size == 0:
                    batch_results.append(None)
                    continue
                
                # Filter detections by confidence threshold
                valid_detections = pred[pred[:, 4] > conf_threshold]
                
                if len(valid_detections) == 0:
                    batch_results.append(None)
                    continue
                
                # Extract coordinates, scores, and classes
                boxes = valid_detections[:, :4]  # [x1, y1, x2, y2]
                scores = valid_detections[:, 4]   # confidence
                classes = valid_detections[:, 5].astype(int)  # class IDs
                
                batch_results.append((boxes, scores, classes))
                
            except Exception as e:
                print(f"   Native NMS processing warning: {e}")
                batch_results.append(None)
        
        return batch_results
    
    def process_video_dual_gpu(self, video_path, output_csv, frame_skip=25):
        """Process video using dual GPU load balancing"""
        print(f"\n🚀 Processing Video with Dual-GPU Load Balancing: {video_path}")
        print(f"   Frame skip: {frame_skip} (every {frame_skip}th frame)")
        print(f"   Output: {output_csv}")
        
        self.start_time = time.time()
        
        # Setup DALI pipeline
        try:
            self.dali_processor.setup_pipeline(video_path)
        except Exception as e:
            print(f"❌ Failed to setup DALI pipeline: {e}")
            return
        
        # Start GPU worker threads
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit worker tasks
            future0 = executor.submit(self._gpu_worker, self.gpu0_engine, 0)
            future1 = executor.submit(self._gpu_worker, self.gpu1_engine, 1)
            
            # Start result collector thread
            result_collector = threading.Thread(target=self._collect_results, args=(output_csv,))
            result_collector.start()
            
            try:
                # Process DALI batches and distribute to GPUs
                batch_count = 0
                for batch_data in self.dali_processor:
                    if isinstance(batch_data, list) and len(batch_data) > 0:
                        batch_frames = batch_data[0]["images"]
                        
                        # Convert to numpy if needed
                        if hasattr(batch_frames, 'cpu'):
                            batch_frames = batch_frames.cpu().numpy()
                        elif hasattr(batch_frames, 'numpy'):
                            batch_frames = batch_frames.numpy()
                        
                        # Generate frame indices
                        frame_indices = list(range(
                            batch_count * self.batch_size * frame_skip,
                            (batch_count + 1) * self.batch_size * frame_skip,
                            frame_skip
                        ))
                        
                        # Add to work queue (blocks if queue is full)
                        self.work_queue.put((batch_frames, frame_indices))
                        batch_count += 1
                        
                        if batch_count % 10 == 0:
                            print(f"   📊 Processed {batch_count} batches")
            
            except Exception as e:
                print(f"❌ Error during dual-GPU processing: {e}")
            
            finally:
                # Shutdown workers
                self.work_queue.put(None)  # GPU 0 shutdown
                self.work_queue.put(None)  # GPU 1 shutdown
                
                # Wait for workers to finish
                future0.result()
                future1.result()
                
                # Signal result collector to stop
                self.result_queue.put(None)
                result_collector.join()
        
        self._print_final_stats()
    
    def _collect_results(self, output_csv):
        """Collect results from GPU workers and write to CSV"""
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['frame', 'class', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax'])
            
            while True:
                try:
                    result = self.result_queue.get(timeout=5.0)
                    if result is None:  # Shutdown signal
                        break
                    
                    frame_indices, batch_detections, inference_time, worker_id = result
                    self.inference_times.append(inference_time)
                    
                    # Write detections to CSV
                    for i, (frame_idx, detections) in enumerate(zip(frame_indices, batch_detections)):
                        if detections is not None:
                            boxes, scores, classes = detections
                            for j in range(len(boxes)):
                                writer.writerow([
                                    frame_idx,
                                    int(classes[j]),
                                    float(scores[j]),
                                    float(boxes[j][0]),
                                    float(boxes[j][1]),
                                    float(boxes[j][2]),
                                    float(boxes[j][3])
                                ])
                            
                            self.total_detections += len(boxes)
                    
                    self.processed_frames += len(frame_indices)
                    self.result_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"   ⚠️ Result collector error: {e}")
    
    def _print_final_stats(self):
        """Print final processing statistics"""
        elapsed = time.time() - self.start_time
        
        print(f"\n🎉 Dual-GPU Processing Complete!")
        print(f"   Total time: {elapsed/60:.1f} minutes")
        print(f"   Frames processed: {self.processed_frames}")
        print(f"   Average speed: {self.processed_frames/elapsed:.1f} frames/second")
        if self.inference_times:
            avg_inference = np.mean(self.inference_times)
            print(f"   Average inference time: {avg_inference:.1f}ms per batch")
            print(f"   Inference throughput: {self.batch_size*1000/avg_inference:.1f} images/second")
        print(f"   Total detections: {self.total_detections}")

# Import the DALI processor from the existing script
class DALIVideoProcessor:
    """DALI GPU video processor - simplified version"""
    
    def __init__(self, batch_size, device_id=0):
        self.batch_size = batch_size
        self.device_id = device_id
        self.frame_skip = 25
        
    def setup_pipeline(self, video_path):
        """Setup DALI pipeline with fallback handling"""
        @pipeline_def(batch_size=self.batch_size, num_threads=4, device_id=self.device_id)
        def dual_gpu_video_pipeline():
            video = fn.readers.video(
                device="gpu",
                file_root="",
                filenames=[str(video_path)],
                sequence_length=self.batch_size,
                step=self.frame_skip,
                stride=1,
                normalized=False,
                random_shuffle=False,
                pad_last_batch=True,
                skip_vfr_check=True,
                enable_frame_num=False,
                enable_timestamps=False,
                file_list_include_preceding_frame=True,
                dont_use_mmap=True,
                name="video_reader"
            )
            
            # Resize and preprocess
            resized = fn.resize(video, device="gpu", size=[960, 960], interp_type=types.INTERP_LINEAR, antialias=True)
            normalized = fn.cast(resized, dtype=types.FLOAT) / 255.0
            transposed = fn.transpose(normalized, perm=[0, 3, 1, 2])
            
            return transposed
        
        self.pipeline = dual_gpu_video_pipeline()
        self.iterator = DALIGenericIterator(
            [self.pipeline],
            ['images'],
            reader_name='video_reader',
            last_batch_policy=LastBatchPolicy.FILL,
            auto_reset=True
        )
    
    def __iter__(self):
        return self.iterator

def main():
    parser = argparse.ArgumentParser(description='Dual-GPU Production Batch Inference with DALI')
    parser.add_argument('video_path', type=str, help='Path to input video file')
    parser.add_argument('--output', type=str, default='output_dual_gpu.csv', help='Output CSV file')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--frame-skip', type=int, default=25, help='Frame skip interval')
    parser.add_argument('--engine', type=str, 
                        default='/home/jonas/Documents/vscode/Auklab_ObjectDetection/models/best_batch16_nms.trt',
                        help='TensorRT engine path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"❌ Video file not found: {args.video_path}")
        return
    
    print(f"🚀 Starting Dual-GPU DALI Processing")
    print(f"   Video: {args.video_path}")
    print(f"   Output: {args.output}")
    print(f"   Batch size per GPU: {args.batch_size}")
    print(f"   Frame skip: {args.frame_skip}")
    print(f"   Engine: {args.engine}")
    
    try:
        processor = DualGPUProductionProcessor(
            engine_path=args.engine,
            batch_size=args.batch_size
        )
        
        processor.process_video_dual_gpu(
            video_path=args.video_path,
            output_csv=args.output,
            frame_skip=args.frame_skip
        )
        
        print(f"✅ Processing completed successfully!")
        print(f"   Results saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()