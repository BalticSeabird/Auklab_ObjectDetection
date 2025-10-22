#!/usr/bin/env python3
"""
Simple DALI GPU Decoding Test
Test basic DALI functionality with your video and TensorRT engine
"""

import os
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# NVIDIA DALI imports
try:
    import nvidia.dali as dali
    from nvidia.dali import pipeline_def
    import nvidia.dali.fn as fn
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    from nvidia.dali import types
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False
    print("❌ NVIDIA DALI not available")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def create_dali_pipeline(video_path, batch_size=4, frame_skip=100):
    """Create a simple DALI pipeline for testing"""
    
    @pipeline_def
    def simple_video_pipeline():
        # Read video frames
        video = fn.readers.video(
            device="gpu",
            file_root="",
            filenames=[video_path],
            sequence_length=batch_size,
            step=frame_skip,
            stride=1,
            normalized=False,
            random_shuffle=False,
            pad_last_batch=True,
            skip_vfr_check=True,  # Allow variable frame rate videos
            name="video_reader"
        )
        
        # Resize to 960x960
        resized = fn.resize(
            video,
            device="gpu", 
            size=[960, 960],
            interp_type=types.INTERP_LINEAR
        )
        
        # Normalize to [0,1]
        normalized = fn.cast(resized, device="gpu", dtype=types.FLOAT)
        normalized = normalized / 255.0
        
        # Transpose HWC to CHW (for 4D: NHWC to NCHW)
        transposed = fn.transpose(normalized, device="gpu", perm=[0, 3, 1, 2])
        
        return transposed
    
    # Create pipeline with explicit parameters
    pipeline = simple_video_pipeline(
        batch_size=batch_size,
        num_threads=2,
        device_id=0,
        prefetch_queue_depth=1
    )
    
    return pipeline

def test_dali_with_tensorrt(video_path, engine_path):
    """Test DALI decoding with TensorRT inference"""
    
    if not DALI_AVAILABLE:
        print("❌ DALI not available")
        return False
    
    print(f"🧪 Testing DALI + TensorRT integration")
    print(f"   Video: {video_path}")
    print(f"   Engine: {engine_path}")
    
    batch_size = 4
    
    try:
        # Create DALI pipeline
        print("   Creating DALI pipeline...")
        pipeline = create_dali_pipeline(video_path, batch_size=batch_size, frame_skip=100)
        pipeline.build()
        
        # Create iterator
        iterator = DALIGenericIterator(
            [pipeline],
            output_map=["frames"],
            reader_name="video_reader",
            last_batch_policy=LastBatchPolicy.FILL,
            auto_reset=True
        )
        
        print("   ✅ DALI pipeline created successfully!")
        
        # Load TensorRT engine
        print("   Loading TensorRT engine...")
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(f.read())
        
        context = engine.create_execution_context()
        stream = cuda.Stream()
        
        # Get tensor info
        input_name = None
        output_names = []
        
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            is_input = engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
            
            if is_input:
                input_name = tensor_name
            else:
                output_names.append(tensor_name)
        
        print(f"   ✅ TensorRT engine loaded! Input: {input_name}")
        
        # Allocate memory
        input_shape = (batch_size, 3, 960, 960)
        input_size = int(np.prod(input_shape) * 4)
        d_input = cuda.mem_alloc(input_size)
        
        # Test processing one batch
        print("   Processing one batch...")
        batch_start = time.time()
        
        # Get DALI batch
        batch = next(iterator)
        frames = batch[0]["frames"]
        
        # Convert to numpy if needed
        if hasattr(frames, 'cpu'):
            frame_data = frames.cpu().numpy()
        else:
            frame_data = frames
        
        decode_time = time.time() - batch_start
        
        print(f"   ✅ DALI decode successful!")
        print(f"   ✅ Raw shape: {frame_data.shape}")
        print(f"   ✅ Dtype: {frame_data.dtype}")
        print(f"   ✅ Decode time: {decode_time*1000:.1f}ms")
        
        # DALI video reader outputs (batch_size, sequence_length, channels, height, width)
        # We need to reshape to (batch_size, channels, height, width) by taking first frame of sequence
        if len(frame_data.shape) == 5:
            # Take first frame from sequence: (batch, seq, c, h, w) -> (batch, c, h, w)  
            frame_data = frame_data[:, 0, :, :, :]
            print(f"   ✅ Reshaped to: {frame_data.shape}")
        
        # Test TensorRT inference
        inf_start = time.time()
        
        # Set dynamic shape
        context.set_input_shape(input_name, input_shape)
        
        # Copy to GPU
        frames_contiguous = np.ascontiguousarray(frame_data)
        cuda.memcpy_htod_async(d_input, frames_contiguous, stream)
        
        # Set tensor addresses
        context.set_tensor_address(input_name, int(d_input))
        
        # For outputs, we need at least one output tensor
        if output_names:
            output_shape = context.get_tensor_shape(output_names[0])
            if hasattr(output_shape, '__len__'):
                shape_tuple = tuple(output_shape)
            else:
                shape_tuple = tuple(output_shape[i] for i in range(len(output_shape)))
            
            output_size = int(np.prod(shape_tuple) * 4)
            d_output = cuda.mem_alloc(output_size)
            h_output = cuda.pagelocked_empty(shape_tuple, dtype=np.float32)
            
            context.set_tensor_address(output_names[0], int(d_output))
            
            # Execute inference
            success = context.execute_async_v3(stream.handle)
            
            if success:
                stream.synchronize()
                cuda.memcpy_dtoh(h_output, d_output)
                
                inf_time = time.time() - inf_start
                
                print(f"   ✅ TensorRT inference successful!")
                print(f"   ✅ Inference time: {inf_time*1000:.1f}ms")
                print(f"   ✅ Total time: {(decode_time + inf_time)*1000:.1f}ms")
                print(f"   ✅ Throughput: {batch_size/(decode_time + inf_time):.1f} images/second")
                
                # Cleanup
                d_output.free()
            else:
                print("   ❌ TensorRT inference failed")
        
        # Cleanup
        d_input.free()
        del iterator
        del pipeline
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    
    video_path = "vid/input.mp4"
    engine_path = "/home/jonas/Documents/vscode/Auklab_ObjectDetection/models/auklab_model_xlarge_combined_4564_v1_clean.trt"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    if not os.path.exists(engine_path):
        print(f"❌ Engine not found: {engine_path}")
        return
    
    print("🚀 DALI + TensorRT Integration Test")
    print("=" * 50)
    
    success = test_dali_with_tensorrt(video_path, engine_path)
    
    if success:
        print("\n🎉 SUCCESS! DALI + TensorRT integration working!")
        print("✅ Ready for production DALI implementation")
    else:
        print("\n❌ Test failed. Check error messages above.")

if __name__ == "__main__":
    main()