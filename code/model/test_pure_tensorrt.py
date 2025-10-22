"""
Pure TensorRT Implementation - No PyTorch
Tests if TensorRT works fine without PyTorch tensor operations
"""

import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
from pathlib import Path

class PureTensorRTInference:
    def __init__(self, engine_path):
        """Initialize pure TensorRT inference without PyTorch"""
        print("🔧 Initializing Pure TensorRT Inference...")
        
        # Load TensorRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, 'rb') as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Get input/output info
        self.input_shape = (21, 3, 960, 960)  # Test with optimal batch size 21
        
        # Set input shape for dynamic shapes
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensor_name, self.input_shape)
        
        # Allocate GPU memory
        self.input_size = int(np.prod(self.input_shape) * 4)  # float32
        self.output_sizes = []
        
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
                output_shape = self.context.get_tensor_shape(tensor_name)
                output_size = int(np.prod(output_shape) * 4)
                self.output_sizes.append(output_size)
        
        # Allocate GPU memory
        self.d_input = cuda.mem_alloc(self.input_size)
        self.d_outputs = [cuda.mem_alloc(size) for size in self.output_sizes]
        
        # Create CUDA stream
        self.stream = cuda.Stream()
        
        print(f"✅ TensorRT engine loaded: {Path(engine_path).name}")
        print(f"📊 Input shape: {self.input_shape}")
        print(f"📊 Output count: {len(self.d_outputs)}")

    def preprocess_numpy(self, frames):
        """Preprocess frames using pure NumPy (no PyTorch)"""
        batch = np.zeros(self.input_shape, dtype=np.float32)
        
        for i, frame in enumerate(frames):
            if i >= self.input_shape[0]:
                break
                
            # Resize frame
            resized = cv2.resize(frame, (960, 960))
            
            # Convert BGR to RGB and normalize
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb.astype(np.float32) / 255.0
            
            # HWC to CHW
            transposed = np.transpose(normalized, (2, 0, 1))
            batch[i] = transposed
        
        return batch

    def inference(self, input_batch):
        """Run inference using pure TensorRT"""
        # Copy input to GPU
        cuda.memcpy_htod_async(self.d_input, input_batch, self.stream)
        
        # Set tensor addresses
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.context.set_tensor_address(tensor_name, int(self.d_input))
            else:  # OUTPUT
                output_idx = 0  # Simplified for single output
                self.context.set_tensor_address(tensor_name, int(self.d_outputs[output_idx]))
        
        # Run inference
        self.context.execute_async_v3(self.stream.handle)
        
        # Copy outputs back to CPU
        outputs = []
        output_idx = 0
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
                output_shape = self.context.get_tensor_shape(tensor_name)
                output = np.empty(output_shape, dtype=np.float32)
                cuda.memcpy_dtoh_async(output, self.d_outputs[output_idx], self.stream)
                outputs.append(output)
                output_idx += 1
        
        # Synchronize
        self.stream.synchronize()
        
        return outputs

def test_pure_tensorrt():
    """Test TensorRT without any PyTorch operations"""
    print("🚀 Testing Pure TensorRT Implementation (No PyTorch)")
    print("="*60)
    
    # Use existing TensorRT engine
    engine_path = '/home/jonas/Documents/vscode/Auklab_ObjectDetection/models/auklab_model_xlarge_combined_4564_v1_batch_fixed.trt'
    
    if not Path(engine_path).exists():
        print(f"❌ Engine not found: {engine_path}")
        return
    
    try:
        # Initialize inference
        inference = PureTensorRTInference(engine_path)
        
        # Create dummy frames (simulate video frames)
        print("📹 Creating test frames...")
        frames = []
        for i in range(21):  # Test with optimal batch size 21
            # Create a random frame
            frame = np.random.randint(0, 255, (960, 960, 3), dtype=np.uint8)
            frames.append(frame)
        
        # Run multiple inference tests
        print("🧪 Running inference tests...")
        inference_times = []
        
        for test_run in range(10):
            print(f"  Test {test_run + 1}/10...", end=' ')
            
            # Preprocess
            start_time = time.time()
            input_batch = inference.preprocess_numpy(frames)
            preprocess_time = time.time() - start_time
            
            # Inference
            start_time = time.time()
            outputs = inference.inference(input_batch)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Check outputs
            total_detections = 0
            for output in outputs:
                if len(output.shape) >= 2:
                    total_detections += output.shape[0] if output.shape[0] < 10000 else 0
            
            print(f"✅ {inference_time*1000:.1f}ms, {total_detections} detections")
        
        # Calculate statistics
        avg_inference_time = np.mean(inference_times)
        fps = 21 / avg_inference_time  # Batch size 21 for testing
        
        print(f"\n📊 Results:")
        print(f"   Average inference time: {avg_inference_time*1000:.1f}ms")
        print(f"   Effective FPS: {fps:.1f}")
        print(f"   GPU utilization: TensorRT only")
        
        if avg_inference_time < 0.1:  # Less than 100ms is good
            print("✅ SUCCESS: TensorRT working without PyTorch conflicts!")
            return True
        else:
            print("⚠️  Performance suboptimal but functional")
            return True
            
    except Exception as e:
        if "cuTensor" in str(e) or "permutate" in str(e):
            print(f"❌ FAILED: Same CuTensor error occurs even without PyTorch: {e}")
            print("   This indicates a fundamental TensorRT engine issue")
            return False
        else:
            print(f"❌ FAILED: Other error: {e}")
            return False

if __name__ == "__main__":
    success = test_pure_tensorrt()
    
    if success:
        print("\n💡 Next step: Since TensorRT works alone, the issue is PyTorch-TensorRT coexistence")
        print("   We can implement a separate process architecture or fix the tensor formats")
    else:
        print("\n💡 The TensorRT engine itself has fundamental issues")
        print("   Need to regenerate with different settings or CUDA versions")