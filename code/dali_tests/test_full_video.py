"""Full video DALI test - reads entire video to catch corruption"""
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import sys

class SimpleVideoPipe(Pipeline):
    def __init__(self, video_path, batch_size=1, seq_len=4):
        super().__init__(batch_size, num_threads=2, device_id=0, seed=42)
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

def test_full_video(video_path, seq_len=4, batch_size=1):
    """Read entire video with DALI to catch mid-file corruption"""
    print(f"Testing full video: {video_path}")
    print(f"Sequence length: {seq_len}, Batch size: {batch_size}")

    try:
        pipe = SimpleVideoPipe(video_path, batch_size=batch_size, seq_len=seq_len)
        pipe.build()
        print("✓ Pipeline built successfully\n")

        batch_count = 0

        # Run until pipeline exhausts the video
        while True:
            try:
                out = pipe.run()
                batch_count += 1

                # Print progress every 10 batches
                if batch_count % 10 == 0:
                    print(f"  Processed {batch_count} batches...")

            except StopIteration:
                print("\n✓ Reached end of video")
                break
            except Exception as e:
                print(f"\n✗ Error during playback at batch {batch_count}: {e}")
                return False

        print(f"✓ Full video test PASSED")
        print(f"  Total batches: {batch_count}")
        return True

    except Exception as e:
        print(f"✗ Full video test FAILED: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_full_video.py <video_file> [seq_len] [batch_size]")
        print("  seq_len: sequence length (default 4)")
        print("  batch_size: batch size (default 1)")
        sys.exit(1)

    video_path = sys.argv[1]
    seq_len = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    success = test_full_video(video_path, seq_len=seq_len, batch_size=batch_size)
    sys.exit(0 if success else 1)
