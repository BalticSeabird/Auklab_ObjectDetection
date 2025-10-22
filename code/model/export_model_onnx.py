# export_model_onnx.py
import torch
import tensorrt as trt
from pathlib import Path

# -------------------------
# CONFIG
# -------------------------
MODEL_YOLO = "models/auklab_model_xlarge_combined_4564_v1.pt"  # Path to your trained PyTorch model
MODEL_ONNX = "models/auklab_model_xlarge_combined_4564_v1.onnx"
TRT_ENGINE = "models/auklab_model_xlarge_combined_4564_v1.trt"

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def export_to_onnx(model_path, onnx_path, input_shape=(1, 3, 960, 960)):
    """Export a PyTorch YOLO model to ONNX."""
    model = torch.load(model_path, map_location="cpu", weights_only = False)['model'].float()  # Adjust if your checkpoint structure is different
    model.eval()
    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        verbose=False,
        opset_version=17,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes={
            'images': {0: 'batch', 2: 'height', 3: 'width'}, 
            'output': {0: 'batch'}
        }
    )
    print(f"ONNX exported to {onnx_path}")


def build_trt_engine(onnx_file_path: str, engine_file_path: str):
    """Build TensorRT engine from ONNX using modern TRT API."""
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
                print("Failed to parse ONNX model!")
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                return None

        # Builder config
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 << 30)  # 16 GB
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # Create optimization profile for dynamic input shapes
        profile = builder.create_optimization_profile()
        
        # Get input tensor name and current shape
        input_tensor = network.get_input(0)
        input_name = input_tensor.name
        input_shape = input_tensor.shape
        print(f"Input tensor: {input_name}, shape: {input_shape}")
        
        # Check if the model has dynamic input or fixed input
        if -1 in input_shape:
            # Dynamic input - set ranges
            min_shape = (1, 3, 640, 640)    # Minimum input size
            opt_shape = (1, 3, 960, 960)    # Optimal input size (training size)
            max_shape = (1, 3, 1280, 1280)  # Maximum input size
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)
            print("Using dynamic input shapes")
        else:
            # Fixed input - use the exact shape from ONNX
            fixed_shape = tuple(input_shape)
            profile.set_shape(input_name, fixed_shape, fixed_shape, fixed_shape)
            config.add_optimization_profile(profile)
            print(f"Using fixed input shape: {fixed_shape}")

        # Build engine
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build the TensorRT engine!")

        # Save engine to file
        with open(engine_file_path, 'wb') as f:
            f.write(serialized_engine)
        print(f"TensorRT engine saved to {engine_file_path}")
        return serialized_engine


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    # Step 1: Export to ONNX
    export_to_onnx(MODEL_YOLO, MODEL_ONNX)

    # Step 2: Build TensorRT engine
    build_trt_engine(MODEL_ONNX, TRT_ENGINE)
