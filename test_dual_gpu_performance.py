#!/usr/bin/env python3
"""
Test dual-GPU performance with synthetic data
"""

import time
import numpy as np
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

# TensorRT imports
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Initialize CUDA contexts for both GPUs
cuda.init()
gpu0_ctx = cuda.Device(0).make_context()
gpu1_ctx = cuda.Device(1).make_context()

class DualGPUPerformanceTest:
    """Test dual-GPU inference performance"""
    
    def __init__(self, engine_path, batch_size=8):
        self.batch_size = batch_size
        self.engine_path = engine_path
        
        # Initialize both engines
        self.gpu0_engine = self._create_engine(0)
        self.gpu1_engine = self._create_engine(1)
        
        print("✅ Dual-GPU engines ready for performance testing")
    
    def _create_engine(self, gpu_id):
        """Create TensorRT engine for specified GPU"""
        print(f"   🚀 Loading engine on GPU {gpu_id}")
        
        # Set correct GPU context
        if gpu_id == 0:
            gpu0_ctx.push()
        else:
            gpu1_ctx.push()
        
        try:
            # Load TensorRT engine
            trt_logger = trt.Logger(trt.Logger.WARNING)
            with open(self.engine_path, 'rb') as f:
                engine_data = f.read()
            
            runtime = trt.Runtime(trt_logger)
            engine = runtime.deserialize_cuda_engine(engine_data)
            context = engine.create_execution_context()
            
            # Create CUDA stream
            stream = cuda.Stream()
            
            # Pre-allocate memory
            input_shape = (self.batch_size, 3, 960, 960)
            output_shape = (self.batch_size, 300, 6)
            
            input_size = int(np.prod(input_shape) * 4)
            output_size = int(np.prod(output_shape) * 4)
            
            input_gpu = cuda.mem_alloc(input_size)
            output_gpu = cuda.mem_alloc(output_size)
            output_host = np.empty(output_shape, dtype=np.float32)
            
            print(f"   ✅ GPU {gpu_id} engine ready")
            
            return {
                'engine': engine,
                'context': context,
                'stream': stream,
                'input_gpu': input_gpu,
                'output_gpu': output_gpu,
                'output_host': output_host,
                'gpu_id': gpu_id
            }
            
        finally:
            if gpu_id == 0:
                gpu0_ctx.pop()
            else:
                gpu1_ctx.pop()
    
    def _run_inference(self, engine_info, batch_data):
        """Run inference on specific GPU"""
        gpu_id = engine_info['gpu_id']
        
        # Set correct GPU context
        if gpu_id == 0:
            gpu0_ctx.push()
        else:
            gpu1_ctx.push()
        
        try:
            # Prepare input
            input_data = np.ascontiguousarray(batch_data.astype(np.float32))
            
            # Copy to GPU
            cuda.memcpy_htod_async(engine_info['input_gpu'], input_data, engine_info['stream'])
            
            # Set tensor addresses
            engine_info['context'].set_input_shape("images", input_data.shape)
            engine_info['context'].set_tensor_address("images", int(engine_info['input_gpu']))
            engine_info['context'].set_tensor_address("output0", int(engine_info['output_gpu']))
            
            # Run inference
            start_time = time.time()
            engine_info['context'].execute_async_v3(engine_info['stream'].handle)
            engine_info['stream'].synchronize()
            inference_time = (time.time() - start_time) * 1000
            
            # Copy result back
            cuda.memcpy_dtoh_async(engine_info['output_host'], engine_info['output_gpu'], engine_info['stream'])
            engine_info['stream'].synchronize()
            
            return inference_time, gpu_id
            
        finally:
            if gpu_id == 0:
                gpu0_ctx.pop()
            else:
                gpu1_ctx.pop()
    
    def test_single_gpu_performance(self, num_batches=50):
        """Test single GPU performance"""
        print(f"\n🧪 Testing Single GPU Performance ({num_batches} batches)")
        
        # Generate synthetic data
        test_data = np.random.rand(self.batch_size, 3, 960, 960).astype(np.float32)
        
        # Test GPU 0
        gpu0_times = []
        start_time = time.time()
        
        for i in range(num_batches):
            inference_time, _ = self._run_inference(self.gpu0_engine, test_data)
            gpu0_times.append(inference_time)
        
        gpu0_total = time.time() - start_time
        avg_gpu0 = np.mean(gpu0_times)
        
        print(f"   GPU 0: {avg_gpu0:.1f}ms avg, {self.batch_size*1000/avg_gpu0:.1f} img/s")
        print(f"   GPU 0 Total: {gpu0_total:.1f}s, {num_batches*self.batch_size/gpu0_total:.1f} img/s overall")
        
        return avg_gpu0
    
    def test_dual_gpu_performance(self, num_batches=50):
        """Test dual GPU performance with load balancing"""
        print(f"\n🚀 Testing Dual GPU Performance ({num_batches} batches)")
        
        # Create work queues
        work_queue = queue.Queue()
        result_queue = queue.Queue()
        
        # Generate work items
        for i in range(num_batches):
            test_data = np.random.rand(self.batch_size, 3, 960, 960).astype(np.float32)
            work_queue.put((i, test_data))
        
        # Worker function
        def gpu_worker(engine_info):
            while True:
                try:
                    work_item = work_queue.get(timeout=1.0)
                    if work_item is None:
                        break
                    
                    batch_id, batch_data = work_item
                    inference_time, gpu_id = self._run_inference(engine_info, batch_data)
                    result_queue.put((batch_id, inference_time, gpu_id))
                    
                    work_queue.task_done()
                    
                except queue.Empty:
                    break
        
        # Start workers
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            future0 = executor.submit(gpu_worker, self.gpu0_engine)
            future1 = executor.submit(gpu_worker, self.gpu1_engine)
            
            # Collect results
            results = []
            for _ in range(num_batches):
                results.append(result_queue.get())
            
            # Signal workers to stop
            work_queue.put(None)
            work_queue.put(None)
            
            future0.result()
            future1.result()
        
        total_time = time.time() - start_time
        
        # Analyze results
        gpu0_times = [r[1] for r in results if r[2] == 0]
        gpu1_times = [r[1] for r in results if r[2] == 1]
        
        print(f"   GPU 0: {len(gpu0_times)} batches, avg {np.mean(gpu0_times):.1f}ms")
        print(f"   GPU 1: {len(gpu1_times)} batches, avg {np.mean(gpu1_times):.1f}ms")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Overall throughput: {num_batches*self.batch_size/total_time:.1f} img/s")
        print(f"   Load distribution: GPU0={len(gpu0_times)}, GPU1={len(gpu1_times)}")
        
        return total_time

def main():
    engine_path = "/home/jonas/Documents/vscode/Auklab_ObjectDetection/models/best_batch16_nms.trt"
    batch_size = 8
    
    print("🚀 Dual-GPU Performance Test")
    print("=" * 50)
    
    try:
        tester = DualGPUPerformanceTest(engine_path, batch_size)
        
        # Test single GPU first
        single_gpu_time = tester.test_single_gpu_performance(50)
        
        # Test dual GPU
        dual_gpu_time = tester.test_dual_gpu_performance(50)
        
        # Calculate speedup
        single_throughput = batch_size * 1000 / single_gpu_time  # img/s for single batch
        dual_throughput = 50 * batch_size / dual_gpu_time        # img/s for full test
        
        print(f"\n📊 Performance Summary:")
        print(f"   Single GPU throughput: {single_throughput:.1f} img/s")
        print(f"   Dual GPU throughput: {dual_throughput:.1f} img/s")
        print(f"   Speedup: {dual_throughput/single_throughput:.1f}x")
        
        if dual_throughput > 94.5:
            print(f"   🎉 SUCCESS! Exceeded baseline of 94.5 img/s")
        else:
            print(f"   ⚠️ Below baseline of 94.5 img/s")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()