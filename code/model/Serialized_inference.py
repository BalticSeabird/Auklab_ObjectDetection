import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # initializes CUDA driver
import time

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
TRT_ENGINE = "models/auklab_model_xlarge_combined_4564_v1.trt"  # TensorRT engine path


def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


class TRTInference:
    def __init__(self, engine):
        self.engine = engine
        self.context = engine.create_execution_context()
        
        # Get input and output tensor names and shapes
        self.input_names = []
        self.output_names = []
        self.input_shapes = []
        self.output_shapes = []
        
        # Modern TensorRT API for getting tensor info
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.input_names.append(tensor_name)
                shape = engine.get_tensor_shape(tensor_name)
                self.input_shapes.append(shape)
            else:
                self.output_names.append(tensor_name)
                shape = engine.get_tensor_shape(tensor_name)
                self.output_shapes.append(shape)
        
        print(f"Input tensors: {list(zip(self.input_names, self.input_shapes))}")
        print(f"Output tensors: {list(zip(self.output_names, self.output_shapes))}")
        
        # Set up input shape for dynamic shapes (if needed)
        if -1 in self.input_shapes[0]:
            # Dynamic shape - set to 960x960 
            input_shape = (1, 3, 960, 960)
            self.context.set_input_shape(self.input_names[0], input_shape)
            self.actual_input_shape = input_shape
        else:
            # Fixed shape
            self.actual_input_shape = tuple(self.input_shapes[0])
            
        print(f"Using input shape: {self.actual_input_shape}")
        
        # Allocate memory
        self._allocate_memory()

    def _allocate_memory(self):
        """Allocate GPU memory for inputs and outputs"""
        self.inputs = []
        self.outputs = []
        
        # Allocate input memory
        input_size = int(np.prod(self.actual_input_shape))
        input_dtype = np.float32
        input_host = cuda.pagelocked_empty(input_size, input_dtype)
        input_device = cuda.mem_alloc(input_host.nbytes)
        self.inputs.append({'host': input_host, 'device': input_device, 'shape': self.actual_input_shape})
        
        # Allocate output memory
        for i, output_name in enumerate(self.output_names):
            output_shape = self.context.get_tensor_shape(output_name)
            output_size = int(np.prod(output_shape))
            output_dtype = np.float32
            output_host = cuda.pagelocked_empty(output_size, output_dtype)
            output_device = cuda.mem_alloc(output_host.nbytes)
            self.outputs.append({'host': output_host, 'device': output_device, 'shape': output_shape})
            print(f"Output {i} ({output_name}): shape {output_shape}")

    def infer(self, img):
        # Resize image to correct input size (960x960 for your model)
        height, width = self.actual_input_shape[2], self.actual_input_shape[3]
        img_resized = cv2.resize(img, (width, height))
        
        # Normalize and transpose
        img_input = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        
        # Copy to input buffer
        np.copyto(self.inputs[0]['host'], img_input.ravel())
        
        # Set tensor addresses for execution
        for i, input_name in enumerate(self.input_names):
            self.context.set_tensor_address(input_name, int(self.inputs[i]['device']))
        
        for i, output_name in enumerate(self.output_names):
            self.context.set_tensor_address(output_name, int(self.outputs[i]['device']))
        
        # Copy input to GPU
        cuda.memcpy_htod(self.inputs[0]['device'], self.inputs[0]['host'])
        
        # Execute inference
        stream = cuda.Stream()
        self.context.execute_async_v3(stream.handle)
        stream.synchronize()
        
        # Copy output back to CPU
        results = []
        for i, output in enumerate(self.outputs):
            cuda.memcpy_dtoh(output['host'], output['device'])
            result = output['host'].reshape(output['shape'])
            results.append(result)
        
        return results[0] if len(results) == 1 else results
    
    def __del__(self):
        """Clean up GPU memory"""
        try:
            if hasattr(self, 'inputs'):
                for inp in self.inputs:
                    if 'device' in inp:
                        inp['device'].free()
            if hasattr(self, 'outputs'):
                for out in self.outputs:
                    if 'device' in out:
                        out['device'].free()
        except:
            pass


def run_video_inference(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
        
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video info:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {duration/60:.1f} minutes")
    print(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = None

    engine = load_engine(TRT_ENGINE)
    trt_infer = TRTInference(engine)

    frame_count = 0
    start_time = time.time()
    inference_times = []
    
    print(f"\nStarting processing...")
    print(f"Monitor GPU usage with: watch -n 1 nvidia-smi")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Process frames in batches for better GPU utilization
        if frame_count % 4 == 0 and frame_count > 0:
            # TODO: Implement batch processing here
            pass
        
        # Time the inference
        inference_start = time.time()
        try:
            preds = trt_infer.infer(frame)
            inference_end = time.time()
            inference_time = inference_end - inference_start
            inference_times.append(inference_time)
            
            if isinstance(preds, list):
                # Use the main output (typically the first one for YOLO)
                main_pred = preds[0]
            else:
                main_pred = preds
            # TODO: decode main_pred, draw boxes if needed
        except Exception as e:
            print(f"Error during inference on frame {frame_count}: {e}")
            break

        if out_video is None:
            h, w = frame.shape[:2]
            out_video = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
        out_video.write(frame)
        
        # Print progress every 100 frames
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            processing_fps = frame_count / elapsed
            avg_inference_time = np.mean(inference_times[-100:]) * 1000  # Last 100 frames in ms
            
            print(f"Frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
            print(f"  Processing FPS: {processing_fps:.1f}")
            print(f"  Avg inference time: {avg_inference_time:.1f}ms")
            print(f"  Elapsed: {elapsed/60:.1f}min, ETA: {(elapsed/frame_count)*(total_frames-frame_count)/60:.1f}min")

    cap.release()
    if out_video:
        out_video.release()
    
    # Final statistics
    total_time = time.time() - start_time
    avg_processing_fps = frame_count / total_time
    avg_inference_time = np.mean(inference_times) * 1000
    
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Total frames processed: {frame_count}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Average processing FPS: {avg_processing_fps:.1f}")
    print(f"Average inference time: {avg_inference_time:.1f}ms")
    print(f"Speedup vs real-time: {avg_processing_fps/fps:.1f}x")
    print(f"Output saved to: {output_path}")
    
    return {
        'total_frames': frame_count,
        'total_time': total_time,
        'avg_fps': avg_processing_fps,
        'avg_inference_ms': avg_inference_time
    }


if __name__ == "__main__":
    # Check if input video exists
    import os
    video_path = "vid/input.mp4"
    if not os.path.exists(video_path):
        print(f"Error: Input video {video_path} not found")
        print("Available files in vid/ directory:")
        if os.path.exists("vid"):
            for f in os.listdir("vid"):
                print(f"  {f}")
    else:
        run_video_inference(video_path, "vid/output.mp4")
