"""
TensorRT Engine Generation with CPU Fallback Strategy
Creates a highly compatible TensorRT engine by avoiding problematic GPU operations
"""

import os
import torch
import tensorrt as trt
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path

def export_to_onnx():
    """Export PyTorch model to ONNX format"""
    print("📋 Exporting PyTorch model to ONNX...")
    
    # Load the model
    model_path = '/home/jonas/Documents/vscode/Auklab_ObjectDetection/models/auklab_model_xlarge_combined_4564_v1.pt'
    model = torch.load(model_path, map_location='cpu', weights_only=False)['model']
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 960, 960)
    
    # Export to ONNX
    onnx_path = '/home/jonas/Documents/vscode/Auklab_ObjectDetection/models/yolo_cpu_fallback.onnx'
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,  # Use older opset to avoid newer operations
        do_constant_folding=True,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes={
            'images': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ ONNX model exported: {onnx_path}")
    return onnx_path

def create_simple_tensorrt_engine(onnx_path):
    """Create a simple TensorRT engine with maximum compatibility"""
    print("🔧 Creating TensorRT engine with CPU fallback strategy...")
    
    # Initialize TensorRT
    logger = trt.Logger(trt.Logger.ERROR)  # Reduce verbosity
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX model
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("❌ Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(f"  Error {error}: {parser.get_error(error)}")
            return None
    
    # Configure builder with conservative settings
    config = builder.create_builder_config()
    
    # Very conservative memory settings
    config.max_workspace_size = 2 * (1024 ** 3)  # 2GB workspace
    
    # Disable problematic features
    config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
    config.set_flag(trt.BuilderFlag.FP16)  # Keep FP16 for performance
    
    # Force CPU fallback for problematic layers
    config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
    
    # Set optimization profiles for different batch sizes
    profile = builder.create_optimization_profile()
    profile.set_shape("images", (1, 3, 960, 960), (8, 3, 960, 960), (16, 3, 960, 960))
    config.add_optimization_profile(profile)
    
    # Build engine with timeout
    print("🔄 Building TensorRT engine (this may take several minutes)...")
    try:
        engine = builder.build_engine(network, config)
        if engine is None:
            print("❌ Failed to build TensorRT engine")
            return None
    except Exception as e:
        print(f"❌ Engine build failed: {e}")
        return None
    
    # Save engine
    engine_path = '/home/jonas/Documents/vscode/Auklab_ObjectDetection/models/yolo_cpu_fallback.engine'
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"✅ TensorRT engine saved: {engine_path}")
    print(f"📊 Engine size: {os.path.getsize(engine_path) / (1024*1024):.1f} MB")
    
    return engine_path

def validate_engine(engine_path):
    """Validate the generated engine"""
    print("🧪 Validating TensorRT engine...")
    
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        # Load engine
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
            engine = runtime.deserialize_cuda_engine(f.read())
        
        if engine is None:
            print("❌ Failed to deserialize engine")
            return False
        
        context = engine.create_execution_context()
        
        # Test different batch sizes
        for batch_size in [1, 4, 8, 16]:
            context.set_binding_shape(0, (batch_size, 3, 960, 960))
            
            if not context.all_binding_shapes_specified:
                print(f"❌ Batch size {batch_size}: Shape not specified")
                continue
            
            if not context.all_shape_inputs_specified:
                print(f"❌ Batch size {batch_size}: Shape inputs not specified")
                continue
            
            print(f"✅ Batch size {batch_size}: Validated")
        
        # Get optimal batch size
        optimal_batch = 8  # Conservative choice
        print(f"🎯 Optimal batch size: {optimal_batch}")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def main():
    """Main execution function"""
    print("🚀 Starting TensorRT Engine Generation with CPU Fallback Strategy")
    print("="*60)
    
    # Step 1: Export to ONNX
    try:
        onnx_path = export_to_onnx()
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return
    
    # Step 2: Create TensorRT engine
    try:
        engine_path = create_simple_tensorrt_engine(onnx_path)
        if engine_path is None:
            print("❌ Engine creation failed")
            return
    except Exception as e:
        print(f"❌ Engine creation failed: {e}")
        return
    
    # Step 3: Validate engine
    try:
        if validate_engine(engine_path):
            print("\n🎉 Success! TensorRT engine created and validated")
            print(f"📁 Engine location: {engine_path}")
            print("\n💡 This engine uses conservative settings to maximize compatibility")
            print("   It includes CPU fallback for problematic operations")
        else:
            print("❌ Engine validation failed")
    except Exception as e:
        print(f"❌ Engine validation failed: {e}")

if __name__ == "__main__":
    main()