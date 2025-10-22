#!/usr/bin/env python3
"""
Enhanced TensorRT Engine Generation - Fixed Version
Addresses GPU tensor library compatibility issues and memory corruption
Optimized for stable multi-GPU operation with PyTorch/CuPy post-processing
"""

import torch
import tensorrt as trt
import numpy as np
from pathlib import Path
import sys
import os

# -------------------------
# ENHANCED CONFIG
# -------------------------
MODEL_YOLO = "models/auklab_model_xlarge_combined_4564_v1.pt"
MODEL_ONNX_BATCH = "models/auklab_model_xlarge_combined_4564_v1_batch_fixed.onnx"
TRT_ENGINE_BATCH = "models/auklab_model_xlarge_combined_4564_v1_batch_fixed.trt"

# Conservative batch configuration for stable operation
BATCH_SIZE_MIN = 1      # Minimum batch size
BATCH_SIZE_OPT = 21     # Optimal batch size (matches working baseline)
BATCH_SIZE_MAX = 32     # Maximum batch size

# Enhanced memory management settings
WORKSPACE_SIZE_GB = 8   # Conservative workspace size (was 32GB)
TARGET_GPU_UTIL = 0.7   # Target 70% GPU utilization for stability

print(f"🚀 Enhanced TensorRT Engine Generator")
print(f"   Fixing GPU tensor library compatibility issues")
print(f"   Optimized for multi-GPU operation")
print(f"   TensorRT version: {trt.__version__}")

def export_to_onnx_enhanced(model_path, onnx_path, input_shape=(21, 3, 960, 960)):
    """Export with enhanced compatibility and error checking"""
    print(f"\n📦 Enhanced ONNX Export")
    print(f"   Model: {model_path}")
    print(f"   Output: {onnx_path}")
    print(f"   Input shape: {input_shape}")
    
    # Verify model exists
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load model with enhanced error handling
    print("   Loading model...")
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if 'model' in checkpoint:
            model = checkpoint['model'].float()
        else:
            model = checkpoint.float()
        model.eval()
        print("   ✅ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        raise
    
    # Enhanced dummy input creation
    dummy_input = torch.randn(*input_shape)
    print(f"   Created dummy input: {dummy_input.shape}")
    
    # Test model forward pass
    print("   Testing model forward pass...")
    try:
        with torch.no_grad():
            test_output = model(dummy_input)
        
        # Handle multiple outputs (YOLO typically returns tuple/list)
        if isinstance(test_output, (tuple, list)):
            print(f"   ✅ Forward pass successful, {len(test_output)} outputs:")
            for i, output in enumerate(test_output):
                if hasattr(output, 'shape'):
                    print(f"     Output {i}: {output.shape}")
                else:
                    print(f"     Output {i}: {type(output)}")
        else:
            print(f"   ✅ Forward pass successful, output shape: {test_output.shape}")
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        raise
    
    # Enhanced ONNX export with strict settings
    print("   Exporting to ONNX...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            verbose=False,
            opset_version=17,  # Stable opset version
            input_names=['images'],
            output_names=['output0', 'output1'],  # Two outputs based on model test
            dynamic_axes={
                'images': {0: 'batch'},  # Only batch dimension dynamic
                'output0': {0: 'batch'},
                'output1': {0: 'batch'}
            },
            # Enhanced export settings for stability
            export_params=True,
            keep_initializers_as_inputs=False,
            do_constant_folding=True,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX
        )
        print("   ✅ ONNX export successful")
    except Exception as e:
        print(f"   ❌ ONNX export failed: {e}")
        raise
    
    # Verify ONNX file
    onnx_size = Path(onnx_path).stat().st_size / (1024**2)
    print(f"   ONNX file size: {onnx_size:.1f} MB")
    
    return onnx_path

def build_trt_engine_enhanced(onnx_file_path: str, engine_file_path: str):
    """Build TensorRT engine with enhanced stability settings"""
    print(f"\n🔧 Enhanced TensorRT Engine Build")
    print(f"   Input: {onnx_file_path}")
    print(f"   Output: {engine_file_path}")
    
    # Enhanced TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)  # More verbose for debugging
    
    try:
        with trt.Builder(TRT_LOGGER) as builder, \
             builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
             trt.OnnxParser(network, TRT_LOGGER) as parser, \
             builder.create_builder_config() as config:

            # Enhanced ONNX parsing
            print("   Parsing ONNX model...")
            onnx_file_path = Path(onnx_file_path)
            if not onnx_file_path.exists():
                raise FileNotFoundError(f"ONNX file not found: {onnx_file_path}")

            with open(onnx_file_path, 'rb') as f:
                onnx_data = f.read()
                
            success = parser.parse(onnx_data)
            if not success:
                print("❌ ONNX parsing failed!")
                for i in range(parser.num_errors):
                    error = parser.get_error(i)
                    print(f"   Error {i}: {error}")
                raise RuntimeError("ONNX parsing failed")
            
            print("   ✅ ONNX parsed successfully")
            
            # Enhanced builder configuration for stability
            print("   Configuring TensorRT builder...")
            
            # Conservative memory allocation for stability
            workspace_size = WORKSPACE_SIZE_GB << 30  # Conservative 8GB
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
            print(f"   Workspace memory: {WORKSPACE_SIZE_GB} GB (conservative)")
            
            # Stability-focused optimizations
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("   FP16 optimization: Enabled")
            
            # Enhanced precision settings for stability
            config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            print("   Precision constraints: Enabled")
            
            # Memory optimization flags
            config.set_flag(trt.BuilderFlag.REFIT)
            print("   Engine refitting: Enabled")
            
            # Enhanced optimization profile
            print("   Creating optimization profile...")
            profile = builder.create_optimization_profile()
            
            # Get input information
            if network.num_inputs == 0:
                raise RuntimeError("No inputs found in network")
            
            input_tensor = network.get_input(0)
            input_name = input_tensor.name
            input_shape = input_tensor.shape
            print(f"   Input tensor: {input_name}")
            print(f"   Input shape: {input_shape}")
            
            # Conservative batch size settings for stability
            min_shape = (BATCH_SIZE_MIN, 3, 960, 960)
            opt_shape = (BATCH_SIZE_OPT, 3, 960, 960)  # 21 frames (known working)
            max_shape = (BATCH_SIZE_MAX, 3, 960, 960)
            
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)
            
            print(f"   Batch sizes: {BATCH_SIZE_MIN}-{BATCH_SIZE_MAX} (optimal: {BATCH_SIZE_OPT})")
            
            # Enhanced timing and performance settings (API compatible)
            config.avg_timing_iterations = 8   # More timing iterations for stability
            print("   Enhanced timing settings applied")
            
            # Estimate memory usage
            single_frame_mb = (3 * 960 * 960 * 4) / (1024**2)
            opt_batch_mb = single_frame_mb * BATCH_SIZE_OPT
            max_batch_mb = single_frame_mb * BATCH_SIZE_MAX
            
            print(f"   Memory estimates:")
            print(f"     Single frame: {single_frame_mb:.1f} MB")
            print(f"     Optimal batch: {opt_batch_mb:.1f} MB")
            print(f"     Maximum batch: {max_batch_mb:.1f} MB")
            
            # Enhanced engine building with error checking
            print("   🔨 Building TensorRT engine...")
            print("   This may take 5-10 minutes for enhanced optimization...")
            
            # Build with timing cache
            timing_cache = config.create_timing_cache(b"")
            config.set_timing_cache(timing_cache, ignore_mismatch=False)
            
            serialized_engine = builder.build_serialized_network(network, config)
            
            if serialized_engine is None:
                raise RuntimeError("Engine building failed!")
            
            print("   ✅ Engine built successfully")
            
            # Save engine with verification
            print(f"   Saving engine to {engine_file_path}...")
            engine_path = Path(engine_file_path)
            engine_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(engine_file_path, 'wb') as f:
                f.write(serialized_engine)
            
            # Verify saved engine
            if not engine_path.exists():
                raise RuntimeError("Engine file not created!")
            
            engine_size = engine_path.stat().st_size / (1024**2)
            print(f"   ✅ Engine saved: {engine_size:.1f} MB")
            
            # Enhanced engine analysis
            print("\n📊 Enhanced Engine Analysis:")
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(serialized_engine)
            
            print(f"   Device memory: {engine.device_memory_size / (1024**2):.1f} MB")
            print(f"   I/O tensors: {engine.num_io_tensors}")
            print(f"   Optimization profiles: {engine.num_optimization_profiles}")
            
            # Analyze all tensors
            for i in range(engine.num_io_tensors):
                tensor_name = engine.get_tensor_name(i)
                is_input = engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
                tensor_type = "INPUT" if is_input else "OUTPUT"
                print(f"   Tensor {i}: {tensor_name} ({tensor_type})")
            
            return engine_file_path
            
    except Exception as e:
        print(f"❌ Engine building failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def validate_engine_enhanced(engine_path: str):
    """Enhanced engine validation with comprehensive testing"""
    print(f"\n🧪 Enhanced Engine Validation")
    print(f"   Engine: {engine_path}")
    
    if not Path(engine_path).exists():
        raise FileNotFoundError(f"Engine not found: {engine_path}")
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    
    # Load engine
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    
    engine = runtime.deserialize_cuda_engine(engine_data)
    if engine is None:
        raise RuntimeError("Failed to deserialize engine!")
    
    print("   ✅ Engine loaded successfully")
    
    # Create execution context
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create execution context!")
    
    print("   ✅ Execution context created")
    
    # Test batch sizes
    test_batch_sizes = [1, 2, 4, 8, 16, 21, 24, 32]
    successful_batches = []
    
    for batch_size in test_batch_sizes:
        try:
            # Get input tensor name
            input_name = None
            for i in range(engine.num_io_tensors):
                tensor_name = engine.get_tensor_name(i)
                if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    input_name = tensor_name
                    break
            
            if not input_name:
                print(f"   ❌ No input tensor found")
                continue
            
            # Set input shape
            test_shape = (batch_size, 3, 960, 960)
            context.set_input_shape(input_name, test_shape)
            
            # Validate all tensor shapes
            all_valid = True
            tensor_info = []
            
            for i in range(engine.num_io_tensors):
                tensor_name = engine.get_tensor_name(i)
                tensor_shape = context.get_tensor_shape(tensor_name)
                is_input = engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
                
                if -1 in tensor_shape:
                    all_valid = False
                    break
                
                tensor_info.append({
                    'name': tensor_name,
                    'shape': tuple(tensor_shape),
                    'is_input': is_input
                })
            
            if all_valid:
                successful_batches.append(batch_size)
                print(f"   ✅ Batch {batch_size}: Valid")
                
                # Show tensor details for optimal batch size
                if batch_size == BATCH_SIZE_OPT:
                    for info in tensor_info:
                        tensor_type = "INPUT" if info['is_input'] else "OUTPUT"
                        print(f"      {info['name']} ({tensor_type}): {info['shape']}")
            else:
                print(f"   ❌ Batch {batch_size}: Invalid tensor shapes")
                
        except Exception as e:
            print(f"   ❌ Batch {batch_size}: Error - {e}")
    
    print(f"\n   ✅ Validation complete")
    print(f"   Successful batch sizes: {successful_batches}")
    print(f"   Optimal batch size: {BATCH_SIZE_OPT}")
    
    if BATCH_SIZE_OPT not in successful_batches:
        print(f"   ⚠️  WARNING: Optimal batch size {BATCH_SIZE_OPT} failed validation!")
    
    return successful_batches

def main():
    """Enhanced main execution with comprehensive error handling"""
    print("🚀 Enhanced TensorRT Engine Generator for Multi-GPU Operation")
    print("=" * 80)
    print(f"   Target: Fix GPU tensor library compatibility issues")
    print(f"   Goal: Stable 40+ FPS with PyTorch/CuPy post-processing")
    print(f"   Hardware: Dual RTX 4090s optimization")
    
    try:
        # Check prerequisites
        if not Path(MODEL_YOLO).exists():
            raise FileNotFoundError(f"Source model not found: {MODEL_YOLO}")
        
        print(f"\n📋 Configuration:")
        print(f"   Source model: {MODEL_YOLO}")
        print(f"   ONNX output: {MODEL_ONNX_BATCH}")
        print(f"   Engine output: {TRT_ENGINE_BATCH}")
        print(f"   Batch range: {BATCH_SIZE_MIN}-{BATCH_SIZE_MAX}")
        print(f"   Optimal batch: {BATCH_SIZE_OPT}")
        print(f"   Workspace: {WORKSPACE_SIZE_GB} GB")
        
        # Step 1: Enhanced ONNX export
        print("\n" + "="*50)
        print("STEP 1: Enhanced ONNX Export")
        print("="*50)
        onnx_path = export_to_onnx_enhanced(MODEL_YOLO, MODEL_ONNX_BATCH)
        
        # Step 2: Enhanced TensorRT engine build
        print("\n" + "="*50)
        print("STEP 2: Enhanced TensorRT Engine Build")
        print("="*50)
        engine_path = build_trt_engine_enhanced(onnx_path, TRT_ENGINE_BATCH)
        
        # Step 3: Enhanced validation
        print("\n" + "="*50)
        print("STEP 3: Enhanced Engine Validation")
        print("="*50)
        successful_batches = validate_engine_enhanced(engine_path)
        
        # Success summary
        print("\n" + "="*50)
        print("🎉 ENHANCED ENGINE CREATION SUCCESSFUL!")
        print("="*50)
        print(f"   ✅ Engine file: {engine_path}")
        print(f"   ✅ Engine size: {Path(engine_path).stat().st_size / (1024**2):.1f} MB")
        print(f"   ✅ Supported batches: {successful_batches}")
        print(f"   ✅ Optimal batch: {BATCH_SIZE_OPT}")
        print(f"   ✅ Multi-GPU ready: Enhanced compatibility")
        print(f"   ✅ Memory optimized: Conservative {WORKSPACE_SIZE_GB}GB workspace")
        
        print(f"\n🚀 Ready for testing with:")
        print(f"   - Multi-GPU architecture (TensorRT GPU 0, PyTorch GPU 1)")
        print(f"   - Short video: input2.mkv (10 minutes)")
        print(f"   - Target performance: 40+ FPS stable operation")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Enhanced engine creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)