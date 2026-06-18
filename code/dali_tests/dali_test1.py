from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types

class VideoPipe(Pipeline):
    def __init__(self, video_path, batch_size=1, seq_len=8, num_threads=2, device_id=0):
        super().__init__(batch_size, num_threads, device_id, seed=42)
        self.video_path = video_path
        self.seq_len = seq_len

    def define_graph(self):
        video, frame_num, timestamps = fn.readers.video(
            device="gpu",
            filenames=[self.video_path],
            sequence_length=self.seq_len,
            shard_id=0,
            num_shards=1,
            random_shuffle=False,

            # --- critical knobs ---
            enable_frame_num=True,
            enable_timestamps=True,

            # Try toggling these:
            skip_vfr_check=False,
            # skip_vfr_check=True  # <-- test this

            # stride=1,
            # step=1,

            initial_fill=16
        )
        return video, frame_num, timestamps


# FAR3_20250718T034001.mkv

pipe = VideoPipe("./vid/FAR3_20250718T034001.mkv", seq_len=4)
pipe.build()

for i in range(10):
    out = pipe.run()
    frames, frame_nums, timestamps = out

    print("Frame nums:", frame_nums.as_cpu().as_array())
    print("Timestamps:", timestamps.as_cpu().as_array())