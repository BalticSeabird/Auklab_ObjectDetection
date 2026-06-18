"""DALI video + YOLO inference test with CUDA device selection"""
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import torch
import torch.nn.functional as F
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class SimpleVideoPipe(Pipeline):
    def __init__(self, video_path, batch_size=1, seq_len=4, device_id=0):
        super().__init__(batch_size, num_threads=2, device_id=device_id, seed=42)
        self.video_path = video_path
        self.seq_len = seq_len

    def define_graph(self):
        video = fn.readers.video(
            device="gpu",
            filenames=[self.video_path],
            sequence_length=self.seq_len,
            shard_id=0,
            num_shards=1,
            random_shuffle=False,
        )
        return video


def load_yolo_model(model_path, device):
    """Load YOLO model"""
    logger.info(f"Loading YOLO model from: {model_path}")
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        model.to(device)
        logger.info(f"✓ Model loaded on {device}")
        return model
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)


def run_inference(video_path, model_path, cuda_device=0, batch_size=1, seq_len=4):
    """Run YOLO inference on video batches"""
    device = f"cuda:{cuda_device}"

    logger.info(f"Starting inference test")
    logger.info(f"Video: {video_path}")
    logger.info(f"Model: {model_path}")
    logger.info(f"CUDA device: {device}")
    logger.info(f"Batch size: {batch_size}, Seq length: {seq_len}")
    logger.info("-" * 60)

    # Load model
    model = load_yolo_model(model_path, device)

    # Build DALI pipeline
    try:
        logger.info("Building DALI pipeline...")
        pipe = SimpleVideoPipe(video_path, batch_size=batch_size, seq_len=seq_len, device_id=cuda_device)
        pipe.build()
        logger.info("✓ DALI pipeline built\n")
    except Exception as e:
        logger.error(f"Failed to build pipeline: {e}")
        return False

    batch_count = 0
    total_detections = 0

    try:
        while True:
            try:
                # Get video batch from DALI
                out = pipe.run()
                frames_gpu = out[0]
                batch_count += 1

                # Convert DALI tensor to torch tensor (keep on GPU)
                frames = torch.from_numpy(frames_gpu.as_cpu().as_array()).to(device).float()

                # frames shape: (batch, seq_len, height, width, channels) in BHWC format
                # Flatten batch and seq_len for inference
                b, s, h, w, c = frames.shape
                frames_flat = frames.view(b * s, h, w, c)  # (B*S, H, W, C)

                # Convert BHWC -> BCHW for model
                frames_bchw = frames_flat.permute(0, 3, 1, 2)  # (B*S, C, H, W)

                # Resize to model input size (640x640)
                frames_resized = F.interpolate(frames_bchw, size=(640, 640), mode='bilinear', align_corners=False)

                # Normalize to [0, 1]
                frames_norm = frames_resized / 255.0

                logger.info(f"Batch {batch_count}: {frames_norm.shape[0]} frames, shape {frames_norm.shape}")

                # Run model forward pass directly
                with torch.no_grad():
                    # YOLO model expects (B, 3, 640, 640)
                    preds = model.model(frames_norm)

                # Debug: show what the model returned
                if batch_count == 1:  # Log only first batch
                    if isinstance(preds, (list, tuple)):
                        logger.info(f"Model returned tuple/list with {len(preds)} elements")
                        for i, p in enumerate(preds):
                            if hasattr(p, 'shape'):
                                logger.info(f"  Element {i}: shape {p.shape}")
                    elif hasattr(preds, 'shape'):
                        logger.info(f"Model returned tensor: shape {preds.shape}")

                # preds is typically (B, num_detections, 6) where 6 = [x, y, w, h, conf, class]
                batch_detections = 0
                if isinstance(preds, (list, tuple)):
                    preds = preds[0]  # Get main output if tuple

                if hasattr(preds, 'shape') and len(preds.shape) >= 2:
                    # Filter by confidence 0.5
                    if preds.shape[-1] >= 5:  # Has confidence score at index 4
                        conf_mask = preds[..., 4] > 0.5
                        batch_detections = conf_mask.sum().item()
                    else:
                        logger.warning(f"Unexpected prediction shape: {preds.shape}")
                total_detections += batch_detections

                logger.info(f"  → {batch_detections} detections in this batch")

                if batch_count % 10 == 0:
                    logger.info(f"Progress: {batch_count} batches processed, {total_detections} total detections\n")

            except StopIteration:
                logger.info("\n" + "=" * 60)
                logger.info("✓ Reached end of video")
                break

    except Exception as e:
        logger.error(f"✗ Error during inference at batch {batch_count}: {e}", exc_info=True)
        return False

    # Summary
    logger.info(f"✓ Inference test PASSED")
    logger.info(f"  Total batches: {batch_count}")
    logger.info(f"  Total detections: {total_detections}")
    logger.info(f"  Avg detections per batch: {total_detections / max(1, batch_count):.1f}")

    return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inference_video.py <video_file> <model.pt> [cuda_device] [batch_size] [seq_len]")
        print("  video_file: path to video")
        print("  model.pt: path to YOLO model")
        print("  cuda_device: CUDA device index (default 0)")
        print("  batch_size: batch size (default 1)")
        print("  seq_len: sequence length (default 4)")
        print("\nExample:")
        print("  python inference_video.py ./vid/video.mkv ./models/yolov8n.pt 0 4 8")
        sys.exit(1)

    video_path = sys.argv[1]
    model_path = sys.argv[2]
    cuda_device = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    seq_len = int(sys.argv[5]) if len(sys.argv) > 5 else 4

    # Validate inputs
    if not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)

    if not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)

    success = run_inference(video_path, model_path, cuda_device, batch_size, seq_len)
    sys.exit(0 if success else 1)
