# export_model_onnx_batch.py
import torch
import tensorrt as trt
from pathlib import Path

# -------------------------
# CONFIG
# -------------------------
MODEL_YOLO = "models/auklab_model_xlarge_combined_4564_v1.pt"  # Path to your trained PyTorch model
MODEL_ONNX_BATCH = "models/auklab_model_xlarge_combined_4564_v1_batch.onnx"
TRT_ENGINE_BATCH = "models/auklab_model_xlarge_combined_4564_v1_batch.trt"

# Batch processing configuration for dual RTX 4090s (48GB total VRAM)
BATCH_SIZE_MIN = 1      # Minimum batch size for dynamic batching
BATCH_SIZE_OPT = 16     # Optimal batch size (sweet spot for dual RTX 4090s)
BATCH_SIZE_MAX = 32     # Maximum batch size (maximize GPU utilization)

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def export_to_onnx_batch(model_path, onnx_path, input_shape=(16, 3, 960, 960)):
    """Export a PyTorch YOLO model to ONNX with large batch support."""
    print(f"🚀 Loading model: {model_path}")
    model = torch.load(model_path, map_location="cpu", weights_only=False)['model'].float()
    model.eval()
    
    # Create dummy input with batch dimension
    dummy_input = torch.randn(*input_shape)
    print(f"   Input shape for export: {input_shape}")
    
    # Export with dynamic batch dimension
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        verbose=False,
        opset_version=17,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes={
            'images': {0: 'batch', 2: 'height', 3: 'width'},  # Dynamic batch and spatial dims
            'output': {0: 'batch'}  # Dynamic batch dimension for output
        }
    )
    print(f"✅ ONNX exported to {onnx_path}")
    print(f"   Batch dimension: Dynamic (1-32 frames)")
    print(f"   Spatial resolution: 960x960 (optimized for training data)")


def build_trt_engine_batch(onnx_file_path: str, engine_file_path: str):
    """Build TensorRT engine from ONNX with batch processing support."""
    print(f"🔧 Building TensorRT engine with batch support...")
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser, \
         builder.create_builder_config() as config:

        # Parse the ONNX file
        onnx_file_path = Path(onnx_file_path)
        if not onnx_file_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {onnx_file_path}")

        with open(onnx_file_path, 'rb') as f:
            if not parser.parse(f.read()):
                print("❌ Failed to parse ONNX model!")
                for i in range(parser.num_errors):
                    print(f"   Error {i}: {parser.get_error(i)}")
                return None

        # Enhanced builder config for large batch processing
        print("⚙️  Configuring TensorRT builder...")
        
        # Memory configuration - allocate more for large batch processing (dual RTX 4090s)
        workspace_size = 32 << 30  # 32 GB (increased for large batches on dual 4090s)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
        print(f"   Workspace memory: {workspace_size / (1024**3):.1f} GB")
        
        # Performance optimizations
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("   FP16 optimization: Enabled")
        
        # Remove STRICT_TYPES flag (TensorRT version compatibility issue)
        # config.set_flag(trt.BuilderFlag.STRICT_TYPES)  # Commented out for compatibility
        print("   Large batch optimization: Enabled")
        
        # Create optimization profile for batch processing
        profile = builder.create_optimization_profile()
        
        # Get input tensor information
        input_tensor = network.get_input(0)
        input_name = input_tensor.name
        input_shape = input_tensor.shape
        print(f"   Input tensor: {input_name}, shape: {input_shape}")
        
        # Configure dynamic batch processing
        # Format: (batch, channels, height, width) = (B, 3, 960, 960)
        min_shape = (BATCH_SIZE_MIN, 3, 960, 960)  # Minimum: 1 frame
        opt_shape = (BATCH_SIZE_OPT, 3, 960, 960)  # Optimal: 4 frames
        max_shape = (BATCH_SIZE_MAX, 3, 960, 960)  # Maximum: 8 frames
        
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        
        print(f"   Batch configuration:")
        print(f"     Minimum batch size: {BATCH_SIZE_MIN}")
        print(f"     Optimal batch size: {BATCH_SIZE_OPT}")
        print(f"     Maximum batch size: {BATCH_SIZE_MAX}")
        print(f"     Resolution: 960x960 (fixed for optimal performance)")
        print(f"     Total VRAM utilization target: ~40-45GB (dual RTX 4090s)")
        
        # Estimate memory usage
        single_frame_mb = (3 * 960 * 960 * 4) / (1024**2)  # FP32 input
        max_batch_mb = single_frame_mb * BATCH_SIZE_MAX
        print(f"     Estimated max batch input size: {max_batch_mb:.1f} MB")
        
        # Build engine with timing cache for faster subsequent builds
        print("🔨 Building TensorRT engine (this may take several minutes)...")
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            raise RuntimeError("❌ Failed to build the TensorRT engine!")

        # Save engine to file
        with open(engine_file_path, 'wb') as f:
            f.write(serialized_engine)
        
        # Get engine size from file
        engine_size = Path(engine_file_path).stat().st_size / (1024**2)
        print(f"✅ TensorRT engine saved to {engine_file_path}")
        print(f"   Engine size: {engine_size:.1f} MB")
        
        # Analyze the built engine (updated for modern TensorRT API)
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        
        print(f"\n📊 Engine Analysis:")
        print(f"   Device memory required: {engine.device_memory_size / (1024**2):.1f} MB")
        print(f"   Number of I/O tensors: {engine.num_io_tensors}")
        
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            tensor_shape = engine.get_tensor_shape(tensor_name)
            is_input = engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
            print(f"   Tensor {i}: {tensor_name} {'(input)' if is_input else '(output)'} - shape: {tensor_shape}")
        
        return serialized_engine


def validate_batch_engine(engine_path: str):
    """Validate the batch engine can handle different batch sizes."""
    print(f"\n🧪 Validating batch engine: {engine_path}")
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # Test different batch sizes up to 32
    test_batch_sizes = [1, 2, 4, 8, 16, 24, 32]
    
    for batch_size in test_batch_sizes:
        test_shape = (batch_size, 3, 960, 960)
        try:
            # Set input shape for this batch size (modern TensorRT API)
            input_name = engine.get_tensor_name(0)  # Assume first tensor is input
            context.set_input_shape(input_name, test_shape)
            
            # Check if all shapes are valid
            all_shapes_valid = True
            for i in range(engine.num_io_tensors):
                tensor_name = engine.get_tensor_name(i)
                tensor_shape = context.get_tensor_shape(tensor_name)
                if -1 in tensor_shape:
                    all_shapes_valid = False
                    break
            
            if all_shapes_valid:
                # Get output shape
                output_name = None
                for i in range(engine.num_io_tensors):
                    tensor_name = engine.get_tensor_name(i)
                    if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
                        output_name = tensor_name
                        break
                
                if output_name:
                    output_shape = context.get_tensor_shape(output_name)
                    print(f"   ✅ Batch size {batch_size}: Input {test_shape} → Output {output_shape}")
                else:
                    print(f"   ⚠️  Batch size {batch_size}: No output tensor found")
            else:
                print(f"   ❌ Batch size {batch_size}: Invalid tensor shapes")
                
        except Exception as e:
            print(f"   ❌ Batch size {batch_size}: Error - {e}")
    
    print("✅ Engine validation complete!")


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    print("🚀 Building Batch-Enabled TensorRT Engine for Dual-GPU Processing")
    print("=" * 70)
    
    try:
        # Step 1: Export to ONNX with batch support
        print("\n📦 Step 1: Exporting model to ONNX with batch support")
        export_to_onnx_batch(MODEL_YOLO, MODEL_ONNX_BATCH, input_shape=(BATCH_SIZE_OPT, 3, 960, 960))

        # Step 2: Build TensorRT engine with batch support
        print("\n🔧 Step 2: Building TensorRT engine with batch processing")
        build_trt_engine_batch(MODEL_ONNX_BATCH, TRT_ENGINE_BATCH)

        # Step 3: Validate the engine
        print("\n🧪 Step 3: Validating batch engine capabilities")
        validate_batch_engine(TRT_ENGINE_BATCH)
        
        print("\n🎉 Large Batch TensorRT Engine Successfully Created!")
        print(f"   Engine file: {TRT_ENGINE_BATCH}")
        print(f"   Supported batch sizes: {BATCH_SIZE_MIN} - {BATCH_SIZE_MAX}")
        print(f"   Optimal batch size: {BATCH_SIZE_OPT}")
        print(f"   Target performance: 80+ FPS with >80% GPU utilization")
        print(f"   Memory efficiency: Designed for dual RTX 4090s (48GB total)")
        print("\n🚀 Ready for high-throughput dual-GPU processing!")
        
    except Exception as e:
        print(f"\n❌ Error during engine creation: {e}")
        import traceback
        traceback.print_exc()