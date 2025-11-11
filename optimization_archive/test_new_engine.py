#!/usr/bin/env python3
"""
Test script for new TensorRT engine with native GPU NMS
"""

import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

def load_test_image():
    """Load and preprocess a test image"""
    # Use one of the existing test images
    img_path = "images/BONDEN6_20250705T042000_000624_000656_0000.png"
    if not os.path.exists(img_path):
        # Create a simple test image if file doesn't exist
        img = np.random.randint(0, 255, (960, 960, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (960, 960))
    
    # Preprocess for YOLO
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, 0)  # Add batch dimension
    return img

def test_engine_simple():
    """Simple engine test with single image"""
    engine_path = "models/best_batch16_nms.trt"
    
    print(f"🧪 Testing TensorRT engine: {engine_path}")
    
    # Load engine
    trt_logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f:
        engine = trt.Runtime(trt_logger).deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    print(f"   ✅ Engine loaded successfully")
    print(f"   📊 Number of bindings: {engine.num_io_tensors}")
    
    # Check bindings  
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        dtype = engine.get_tensor_dtype(name)
        shape = engine.get_tensor_shape(name)
        mode = engine.get_tensor_mode(name)
        is_input = (mode == trt.TensorIOMode.INPUT)
        print(f"   {'📥' if is_input else '📤'} {name}: {shape} ({dtype})")
    
    # Prepare input data
    input_img = load_test_image()
    print(f"   📷 Input image shape: {input_img.shape}")
    
    # Set dynamic shapes for batch size 1
    context.set_input_shape("images", input_img.shape)
    
    # Get output shape after setting input
    output_shape = context.get_tensor_shape("output0")
    
    # Allocate GPU memory
    input_size = int(np.prod(input_img.shape) * 4)  # float32, ensure int
    output_size = int(np.prod(output_shape) * 4)
    
    input_gpu = cuda.mem_alloc(input_size)
    output_gpu = cuda.mem_alloc(output_size)
    
    print(f"   💾 Allocated GPU memory: input={input_size//1024//1024:.1f}MB, output={output_size//1024//1024:.1f}MB")
    print(f"   📤 Expected output shape: {output_shape}")
    
    # Copy input to GPU
    input_data = np.ascontiguousarray(input_img.astype(np.float32))
    cuda.memcpy_htod(input_gpu, input_data)
    
    # Set tensor addresses
    context.set_tensor_address("images", int(input_gpu))
    context.set_tensor_address("output0", int(output_gpu))
    
    # Run inference
    start_time = time.time()
    stream = cuda.Stream()
    context.execute_async_v3(stream.handle)
    stream.synchronize()
    inference_time = time.time() - start_time
    
    # Copy output back
    output_data = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(output_data, output_gpu)
    
    print(f"   ⚡ Inference time: {inference_time*1000:.1f}ms")
    print(f"   📤 Output shape: {output_data.shape}")
    
    # Basic validation
    if len(output_data.shape) == 3 and output_data.shape[2] >= 6:
        detections = output_data[0]  # First batch
        valid_detections = detections[detections[:, 4] > 0.1]  # Confidence > 0.1
        print(f"   🎯 Valid detections (conf>0.1): {len(valid_detections)}")
        
        if len(valid_detections) > 0:
            max_conf = np.max(valid_detections[:, 4])
            print(f"   📈 Max confidence: {max_conf:.3f}")
    
    # Cleanup
    input_gpu.free()
    output_gpu.free()
    
    print(f"   ✅ Engine test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_engine_simple()
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()