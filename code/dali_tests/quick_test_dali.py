"""Quick DALI video test - minimal setup"""
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn

class SimpleVideoPipe(Pipeline):
    def __init__(self, video_path, batch_size=1, seq_len=4):
        super().__init__(batch_size, num_threads=2, device_id=0, seed=42)
        self.video_path = video_path
        self.seq_len = seq_len

    def define_graph(self):
        # Simplest possible: just read frames, no fancy metadata
        video = fn.readers.video(
            device="gpu",
            filenames=[self.video_path],
            sequence_length=self.seq_len,
            shard_id=0,
            num_shards=1,
            random_shuffle=False,
        )
        return video

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python quick_test_dali.py <video_file>")
        sys.exit(1)

    video_path = sys.argv[1]
    print(f"Testing DALI on: {video_path}")

    try:
        pipe = SimpleVideoPipe(video_path, seq_len=4)
        pipe.build()
        print("✓ Pipeline built successfully")

        # Try to read a few frames
        for i in range(3):
            out = pipe.run()
            frames = out[0]
            print(f"✓ Frame batch {i+1}: shape {frames.shape()}")

        print("\n✓ DALI test PASSED")
    except Exception as e:
        print(f"\n✗ DALI test FAILED: {e}")
        sys.exit(1)
