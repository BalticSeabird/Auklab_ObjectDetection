import subprocess
import shlex
from pathlib import Path


def build_ffmpeg_command(
    ffmpeg_path: str,
    rtsp_url: str,
    output_pattern: str,
    segment_time: int = 60,
    loglevel: str = "info",
):
    """
    Build an ffmpeg command for RTSP segmentation with improved timestamp handling.
    """

    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel", loglevel,

        # --- Input handling ---
        "-fflags", "+genpts",                     # Generate missing PTS
        "-rtsp_transport", "tcp",
        "-allowed_media_types", "video+audio",
        "-use_wallclock_as_timestamps", "1",
        "-max_delay", "100000",

        "-i", rtsp_url,

        # --- Stream mapping ---
        "-map", "0:v:0",
        "-map", "0:a:0?",                         # optional audio

        # --- Copy streams (fast, non-destructive) ---
        "-c:v", "copy",
        "-c:a", "copy",

        # --- Segmentation ---
        "-f", "segment",
        "-reset_timestamps", "1",
        "-segment_time", str(segment_time),
        "-segment_atclocktime", "1",
        "-segment_format", "mkv",

        # --- Output naming ---
        "-strftime", "1",
        output_pattern,
    ]

    return cmd


def run_ffmpeg(cmd):
    """
    Run ffmpeg command and stream logs in real-time.
    """
    print("Running command:")
    print(" ".join(shlex.quote(c) for c in cmd))

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        for line in process.stdout:
            print(line, end="")  # stream logs live

        process.wait()
    except KeyboardInterrupt:
        print("Stopping ffmpeg...")
        process.terminate()
        process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg exited with code {process.returncode}")


if __name__ == "__main__":
    ffmpeg_path = "ffmpeg"
    rtsp_url = "rtsp://admin:Auklab2008@169.254.28.178:554/Streaming/Channels/101"
    output_dir = Path("./segments")
    output_dir.mkdir(exist_ok=True)

    fname_pattern = str(output_dir / "%Y%m%d_%H%M%S.mkv")

    cmd = build_ffmpeg_command(
        ffmpeg_path=ffmpeg_path,
        rtsp_url=rtsp_url,
        output_pattern=fname_pattern,
        segment_time=60,
        loglevel="info",
    )

    run_ffmpeg(cmd)

