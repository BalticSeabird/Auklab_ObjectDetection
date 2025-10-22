#!/usr/bin/env python3
"""
Test the clean TensorRT engine with batch processing.
This uses the newly generated clean engine without PyTorch contamination.
"""

import os
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class CleanTensorRTInference:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.context = None
        self.stream = None
        
        # Memory buffers will be allocated per batch size
        self.d_input = None
        self.d_outputs = {}
        self.current_batch_size = None
        
        self.load_engine()
    
    def load_engine(self):
        """Load the TensorRT engine."""
        print(f"🔧 Loading TensorRT engine: {self.engine_path}")
        
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(self.TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        
        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")
        
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        print("✅ Engine loaded successfully")
        print(f"📊 Engine requires {self.engine.device_memory_size / 1024 / 1024:.1f} MB device memory")
    
    def allocate_buffers(self, batch_size):
        """Allocate GPU memory buffers for specific batch size."""
        if self.current_batch_size == batch_size:
            return  # Already allocated for this batch size
        
        # Free existing buffers
        self.free_buffers()
        
        print(f"🔧 Allocating buffers for batch size: {batch_size}")
        
        # Input shape: [batch_size, 3, 960, 960]
        input_shape = (batch_size, 3, 960, 960)
        input_size = int(np.prod(input_shape)) * 4  # float32 = 4 bytes
        
        # Allocate input buffer
        self.d_input = cuda.mem_alloc(input_size)
        
        # Set input shape for dynamic batching
        self.context.set_input_shape("images", input_shape)
        
        # Allocate output buffers
        self.d_outputs = {}
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
                output_shape = self.context.get_tensor_shape(tensor_name)
                output_size = int(np.prod(output_shape)) * 4  # float32 = 4 bytes
                self.d_outputs[tensor_name] = cuda.mem_alloc(output_size)
                print(f"  📊 Output {tensor_name}: {output_shape} ({output_size / 1024 / 1024:.1f} MB)")
        
        self.current_batch_size = batch_size
        print(f"✅ Buffers allocated for batch size {batch_size}")
    
    def free_buffers(self):
        """Free GPU memory buffers."""
        if self.d_input is not None:
            self.d_input.free()
            self.d_input = None
        
        for buffer in self.d_outputs.values():
            buffer.free()
        self.d_outputs = {}
        
        self.current_batch_size = None
    
    def infer_batch(self, batch_data):
        """Run inference on a batch of data."""
        batch_size = batch_data.shape[0]
        
        # Ensure buffers are allocated for this batch size
        self.allocate_buffers(batch_size)
        
        # Copy input to GPU
        cuda.memcpy_htod_async(self.d_input, batch_data, self.stream)
        
        # Set tensor addresses for new TensorRT API
        self.context.set_tensor_address("images", int(self.d_input))
        for name, buffer in self.d_outputs.items():
            self.context.set_tensor_address(name, int(buffer))
        
        # Execute inference with new API
        success = self.context.execute_async_v3(self.stream.handle)
        
        if not success:
            raise RuntimeError("TensorRT inference failed")
        
        # Synchronize
        self.stream.synchronize()
        
        return True
    
    def cleanup(self):
        """Clean up resources."""
        self.free_buffers()
        # Note: CUDA streams are automatically cleaned up

def test_batch_performance():
    """Test different batch sizes and measure performance."""
    
    engine_path = "/home/jonas/Documents/vscode/Auklab_ObjectDetection/models/auklab_model_xlarge_combined_4564_v1_clean.trt"
    
    if not os.path.exists(engine_path):
        print(f"❌ Engine not found: {engine_path}")
        return
    
    print("🚀 Starting clean TensorRT batch performance test")
    
    # Initialize inference engine
    inference = CleanTensorRTInference(engine_path)
    
    # Test different batch sizes
    test_batches = [1, 2, 4, 8, 12, 16]
    warmup_runs = 3
    test_runs = 10
    
    results = {}
    
    for batch_size in test_batches:
        print(f"\n🧪 Testing batch size: {batch_size}")
        
        # Create dummy data
        input_data = np.random.randn(batch_size, 3, 960, 960).astype(np.float32)
        
        try:
            # Warmup runs
            print(f"🔥 Warming up ({warmup_runs} runs)...")
            for _ in range(warmup_runs):
                inference.infer_batch(input_data)
            
            # Timed runs
            print(f"⏱️ Running {test_runs} timed iterations...")
            times = []
            
            for _ in range(test_runs):
                start_time = time.time()
                inference.infer_batch(input_data)
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Calculate statistics
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            std_time = np.std(times)
            
            # Calculate throughput
            images_per_second = batch_size / avg_time
            fps_per_batch = 1.0 / avg_time
            
            results[batch_size] = {
                'avg_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'std_time': std_time,
                'images_per_second': images_per_second,
                'fps_per_batch': fps_per_batch
            }
            
            print(f"✅ Batch {batch_size:2d}: {avg_time*1000:6.1f}ms avg ({min_time*1000:5.1f}-{max_time*1000:5.1f}ms) | {images_per_second:6.1f} img/s | {fps_per_batch:5.1f} batch/s")
            
        except Exception as e:
            print(f"❌ Batch {batch_size} failed: {e}")
            results[batch_size] = None
    
    # Summary
    print("\n📊 PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"{'Batch':>5} | {'Avg Time':>8} | {'Images/s':>8} | {'Batch/s':>7} | {'Efficiency':>10}")
    print("-" * 70)
    
    best_throughput = 0
    best_batch = 0
    
    for batch_size in test_batches:
        if results[batch_size] is not None:
            r = results[batch_size]
            efficiency = r['images_per_second'] / batch_size * 100  # % of linear scaling
            print(f"{batch_size:5d} | {r['avg_time']*1000:7.1f}ms | {r['images_per_second']:7.1f} | {r['fps_per_batch']:6.1f} | {efficiency:9.1f}%")
            
            if r['images_per_second'] > best_throughput:
                best_throughput = r['images_per_second']
                best_batch = batch_size
        else:
            print(f"{batch_size:5d} | {'FAILED':>8} | {'FAILED':>8} | {'FAILED':>7} | {'FAILED':>10}")
    
    print("=" * 70)
    if best_batch > 0:
        print(f"🏆 Best performance: Batch size {best_batch} with {best_throughput:.1f} images/second")
        
        # Calculate GPU utilization estimate
        single_batch_time = results[1]['avg_time'] if results[1] else None
        if single_batch_time:
            theoretical_max = 1.0 / single_batch_time
            actual_max = best_throughput / best_batch
            utilization = (actual_max / theoretical_max) * 100
            print(f"📈 GPU utilization: {utilization:.1f}% of theoretical maximum")
    
    # Cleanup
    inference.cleanup()
    print("\n🎉 Clean TensorRT test completed successfully!")

if __name__ == "__main__":
    test_batch_performance()