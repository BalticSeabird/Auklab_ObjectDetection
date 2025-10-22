#!/usr/bin/env python3
"""
Clean TensorRT engine regeneration script.
This script runs in complete isolation from PyTorch to avoid any contamination.
"""

import os
import sys
import numpy as np
import onnx
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # This initializes CUDA

# Constants
ENGINE_PATH = "/home/jonas/Documents/vscode/Auklab_ObjectDetection/models/auklab_model_xlarge_combined_4564_v1_clean.trt"
ONNX_PATH = "/home/jonas/Documents/vscode/Auklab_ObjectDetection/models/auklab_model_xlarge_combined_4564_v1_batch_fixed.onnx"

def create_clean_engine():
    """Create a clean TensorRT engine in isolation."""
    
    print("🔧 Creating clean TensorRT engine...")
    print(f"ONNX Model: {ONNX_PATH}")
    print(f"Engine Path: {ENGINE_PATH}")
    
    # Verify ONNX file exists
    if not os.path.exists(ONNX_PATH):
        print(f"❌ ONNX file not found: {ONNX_PATH}")
        return False
    
    # Remove old engine if it exists
    if os.path.exists(ENGINE_PATH):
        print(f"🗑️ Removing old engine: {ENGINE_PATH}")
        os.remove(ENGINE_PATH)
    
    # TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    
    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX model
    print("📖 Parsing ONNX model...")
    with open(ONNX_PATH, 'rb') as model:
        if not parser.parse(model.read()):
            print("❌ Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    print("✅ ONNX model parsed successfully")
    
    # Create builder config
    config = builder.create_builder_config()
    
    # Conservative memory settings for stability
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 * 1024 * 1024 * 1024)  # 6GB
    
    # Set optimization profiles for dynamic batching
    profile = builder.create_optimization_profile()
    
    # Input tensor shape [batch, channels, height, width]
    input_name = network.get_input(0).name
    print(f"📊 Input tensor: {input_name}")
    
    # Conservative batch size range for stability
    profile.set_shape(input_name, 
                     (1, 3, 960, 960),      # min: batch=1
                     (8, 3, 960, 960),      # opt: batch=8  
                     (16, 3, 960, 960))     # max: batch=16
    
    config.add_optimization_profile(profile)
    
    # Build engine
    print("🏗️ Building TensorRT engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("❌ Failed to build TensorRT engine")
        return False
    
    # Save engine
    print(f"💾 Saving engine to: {ENGINE_PATH}")
    with open(ENGINE_PATH, 'wb') as f:
        f.write(serialized_engine)
    
    print("✅ Clean TensorRT engine created successfully!")
    
    # Get file size 
    file_size = os.path.getsize(ENGINE_PATH)
    print(f"📁 Engine size: {file_size / 1024 / 1024:.1f} MB")
    
    return True

def test_clean_engine():
    """Test the newly created engine."""
    
    if not os.path.exists(ENGINE_PATH):
        print("❌ Engine file not found")
        return False
    
    print("\n🧪 Testing clean engine...")
    
    # TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # Load engine
    with open(ENGINE_PATH, 'rb') as f:
        engine_data = f.read()
    
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_data)
    
    if engine is None:
        print("❌ Failed to deserialize engine")
        return False
    
    print("✅ Engine loaded successfully")
    
    # Create execution context
    context = engine.create_execution_context()
    
    # Test with batch size 4
    batch_size = 4
    input_shape = (batch_size, 3, 960, 960)
    output_shape = (batch_size, 67, 8400)  # Based on ONNX output info
    
    print(f"🔍 Testing with batch size: {batch_size}")
    
    # Set input shape
    context.set_input_shape("images", input_shape)
    
    # Allocate memory
    input_size = int(np.prod(input_shape) * np.dtype(np.float32).itemsize)
    output_size = int(np.prod(output_shape) * np.dtype(np.float32).itemsize)
    
    print(f"📊 Input size: {input_size / 1024 / 1024:.1f} MB")
    print(f"📊 Output size: {output_size / 1024 / 1024:.1f} MB")
    
    try:
        # Allocate GPU memory
        d_input = cuda.mem_alloc(input_size)
        d_output = cuda.mem_alloc(output_size)
        
        # Create dummy input data
        h_input = np.random.randn(*input_shape).astype(np.float32)
        h_output = np.zeros(output_shape, dtype=np.float32)
        
        # Copy input to GPU
        cuda.memcpy_htod(d_input, h_input)
        
        # Run inference
        bindings = [int(d_input), int(d_output)]
        
        if context.execute_v2(bindings):
            print("✅ Inference executed successfully")
            
            # Copy output back
            cuda.memcpy_dtoh(h_output, d_output)
            
            print(f"📈 Output shape: {h_output.shape}")
            print(f"📈 Output range: [{h_output.min():.3f}, {h_output.max():.3f}]")
            
            return True
        else:
            print("❌ Inference execution failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    finally:
        # Cleanup
        try:
            d_input.free()
            d_output.free()
        except:
            pass

if __name__ == "__main__":
    print("🚀 Starting clean TensorRT engine regeneration")
    print(f"Python: {sys.version}")
    print(f"TensorRT: {trt.__version__}")
    print(f"PyCUDA: {cuda.Device.count()} devices available")
    
    # Step 1: Create clean engine
    if create_clean_engine():
        # Step 2: Test the engine
        if test_clean_engine():
            print("\n🎉 SUCCESS: Clean engine created and tested!")
            print("Ready for batch processing and performance optimization")
        else:
            print("\n❌ Engine created but test failed")
    else:
        print("\n❌ Failed to create engine")