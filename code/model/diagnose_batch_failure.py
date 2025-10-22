#!/usr/bin/env python3
"""
Detailed diagnostic of TensorRT batch processing failures
Focus on understanding WHY batching fails despite abundant memory
"""

import os
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path

class BatchFailureDiagnostic:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        print(f"🔍 Diagnosing batch failures for: {Path(engine_path).name}")
        
        # Load engine
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        
        if not self.engine:
            raise RuntimeError("Failed to load TensorRT engine")
            
        print(f"✅ Engine loaded successfully")
        self._analyze_engine_specs()
    
    def _analyze_engine_specs(self):
        """Analyze engine specifications"""
        print("\n📊 Engine Analysis:")
        
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_mode = self.engine.get_tensor_mode(tensor_name)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            tensor_dtype = self.engine.get_tensor_dtype(tensor_name)
            
            mode_str = "INPUT" if tensor_mode == trt.TensorIOMode.INPUT else "OUTPUT"
            print(f"   {mode_str}: {tensor_name} | Shape: {tensor_shape} | Type: {tensor_dtype}")
        
        # Check optimization profiles
        print(f"\n🎯 Optimization Profiles: {self.engine.num_optimization_profiles}")
        for profile_idx in range(self.engine.num_optimization_profiles):
            print(f"   Profile {profile_idx}:")
            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    min_shape = self.engine.get_tensor_profile_shape(tensor_name, profile_idx)[0]
                    opt_shape = self.engine.get_tensor_profile_shape(tensor_name, profile_idx)[1]
                    max_shape = self.engine.get_tensor_profile_shape(tensor_name, profile_idx)[2]
                    print(f"     {tensor_name}: min{min_shape} opt{opt_shape} max{max_shape}")

    def test_batch_size(self, batch_size, verbose=True):
        """Test specific batch size with detailed error tracking"""
        if verbose:
            print(f"\n🧪 Testing batch size {batch_size}...")
        
        try:
            # Create execution context
            context = self.engine.create_execution_context()
            
            # Set input shape
            input_shape = (batch_size, 3, 960, 960)
            input_name = None
            
            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    input_name = tensor_name
                    break
            
            if not input_name:
                raise RuntimeError("No input tensor found")
            
            # Check if shape is valid for this engine
            try:
                success = context.set_input_shape(input_name, input_shape)
                if not success:
                    print(f"   ❌ Failed to set input shape {input_shape}")
                    return False
            except Exception as e:
                print(f"   ❌ Exception setting input shape: {e}")
                return False
            
            if verbose:
                print(f"   ✅ Input shape set: {input_shape}")
            
            # Calculate memory requirements
            input_size = int(np.prod(input_shape) * 4)  # float32, cast to int
            if verbose:
                print(f"   📊 Input memory required: {input_size / 1024**3:.3f} GB")
            
            # Allocate GPU memory
            try:
                d_input = cuda.mem_alloc(input_size)
                if verbose:
                    print(f"   ✅ GPU input memory allocated")
            except Exception as e:
                print(f"   ❌ Failed to allocate input memory: {e}")
                return False
            
            # Allocate output memory
            output_sizes = []
            d_outputs = []
            
            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
                    output_shape = context.get_tensor_shape(tensor_name)
                    output_size = int(np.prod(output_shape) * 4)  # float32, cast to int
                    output_sizes.append(output_size)
                    
                    try:
                        d_output = cuda.mem_alloc(output_size)
                        d_outputs.append(d_output)
                        if verbose:
                            print(f"   ✅ Output {tensor_name}: {output_shape} ({output_size / 1024**2:.1f} MB)")
                    except Exception as e:
                        print(f"   ❌ Failed to allocate output memory for {tensor_name}: {e}")
                        return False
            
            # Create CUDA stream
            try:
                stream = cuda.Stream()
                if verbose:
                    print(f"   ✅ CUDA stream created")
            except Exception as e:
                print(f"   ❌ Failed to create CUDA stream: {e}")
                return False
            
            # Create dummy input data
            try:
                input_data = np.random.rand(*input_shape).astype(np.float32)
                cuda.memcpy_htod_async(d_input, input_data, stream)
                if verbose:
                    print(f"   ✅ Input data copied to GPU")
            except Exception as e:
                print(f"   ❌ Failed to copy input data: {e}")
                return False
            
            # Set tensor addresses
            try:
                context.set_tensor_address(input_name, int(d_input))
                
                output_idx = 0
                for i in range(self.engine.num_io_tensors):
                    tensor_name = self.engine.get_tensor_name(i)
                    if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
                        context.set_tensor_address(tensor_name, int(d_outputs[output_idx]))
                        output_idx += 1
                        
                if verbose:
                    print(f"   ✅ Tensor addresses set")
            except Exception as e:
                print(f"   ❌ Failed to set tensor addresses: {e}")
                return False
            
            # Execute inference
            try:
                start_time = time.time()
                success = context.execute_async_v3(stream.handle)
                stream.synchronize()
                end_time = time.time()
                
                if success:
                    inference_time = (end_time - start_time) * 1000
                    if verbose:
                        print(f"   ✅ Inference successful: {inference_time:.2f}ms")
                        fps = batch_size / (inference_time / 1000)
                        print(f"   🚀 Effective FPS: {fps:.1f}")
                    return True
                else:
                    print(f"   ❌ Inference execution failed")
                    return False
                    
            except Exception as e:
                print(f"   ❌ Exception during inference: {e}")
                return False
                
        except Exception as e:
            print(f"   ❌ General exception: {e}")
            return False
        finally:
            # Cleanup
            try:
                if 'd_input' in locals():
                    d_input.free()
                for d_output in d_outputs if 'd_outputs' in locals() else []:
                    d_output.free()
                if 'stream' in locals():
                    del stream
                if 'context' in locals():
                    del context
            except:
                pass

    def progressive_batch_test(self):
        """Test batch sizes progressively to find exact failure point"""
        print(f"\n🔬 Progressive Batch Testing:")
        
        # Test powers of 2 and some in-between values
        test_batches = [1, 2, 4, 6, 8, 12, 16, 20, 21, 24, 28, 32]
        
        results = {}
        last_working = None
        first_failing = None
        
        for batch_size in test_batches:
            success = self.test_batch_size(batch_size, verbose=False)
            results[batch_size] = success
            
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   Batch {batch_size:2d}: {status}")
            
            if success:
                last_working = batch_size
            elif first_failing is None:
                first_failing = batch_size
        
        print(f"\n📈 Results Summary:")
        print(f"   Last working batch size: {last_working}")
        print(f"   First failing batch size: {first_failing}")
        
        # If we have a transition point, test around it
        if last_working and first_failing and first_failing - last_working > 1:
            print(f"\n🔍 Testing transition point ({last_working} → {first_failing}):")
            for batch_size in range(last_working + 1, first_failing):
                success = self.test_batch_size(batch_size, verbose=False)
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"   Batch {batch_size:2d}: {status}")

    def memory_analysis(self):
        """Analyze memory usage patterns"""
        print(f"\n💾 Memory Analysis:")
        
        # Get GPU memory info
        free, total = cuda.mem_get_info()
        print(f"   GPU Memory: {free//1024**3}GB free / {total//1024**3}GB total")
        
        # Calculate memory requirements for different batch sizes
        single_frame_size = int(3 * 960 * 960 * 4)  # float32, cast to int
        print(f"   Single frame: {single_frame_size / 1024**2:.1f} MB")
        
        for batch_size in [1, 8, 16, 21, 32]:
            input_size = batch_size * single_frame_size
            print(f"   Batch {batch_size:2d}: {input_size / 1024**2:.1f} MB input")

if __name__ == "__main__":
    engine_path = "/home/jonas/Documents/vscode/Auklab_ObjectDetection/models/auklab_model_xlarge_combined_4564_v1_batch_fixed.trt"
    
    if not os.path.exists(engine_path):
        print(f"❌ Engine not found: {engine_path}")
        exit(1)
    
    try:
        diagnostic = BatchFailureDiagnostic(engine_path)
        diagnostic.memory_analysis()
        diagnostic.progressive_batch_test()
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()