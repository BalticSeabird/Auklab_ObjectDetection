import subprocess
import shlex
from pathlib import Path


def build_ffmpeg_command(
    ffmpeg_path: str,
    rtsp_url: str,
    output_pattern: str,
    segment_time: int = 60,
    segment_format: str = "mp4",
    include_audio: bool = True,
    loglevel: str = "info",
):
    """
    Build an ffmpeg command for RTSP segmentation optimized for stable timestamps.

    The command avoids transcoding and keeps the original video bitstream,
    while preferring source packet timing over wallclock-derived timestamps.
    """

    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel", loglevel,

        # --- Input handling ---
        "-fflags", "+genpts+igndts+discardcorrupt",  # Stabilize/fix missing packet timing
        "-rtsp_transport", "tcp",
        "-allowed_media_types", "video+audio" if include_audio else "video",
        # Use camera/source timestamps; wallclock timestamps often introduce VFR jitter.
        "-max_delay", "100000",

        "-i", rtsp_url,

        # --- Stream mapping ---
        "-map", "0:v:0",

        # --- Copy streams (fast, non-destructive) ---
        "-c:v", "copy",
        "-copyts",
        "-copytb", "1",
        "-fps_mode", "passthrough",
        "-avoid_negative_ts", "make_zero",

        # --- Segmentation ---
        "-f", "segment",
        "-segment_time", str(segment_time),
        "-segment_time_delta", "0.05",
        "-segment_atclocktime", "0",
        "-segment_format", segment_format,
    ]

    if include_audio:
        cmd.extend(["-map", "0:a:0?", "-c:a", "copy"])
    else:
        cmd.extend(["-an"])

    if segment_format == "mp4":
        # Keep each segment self-contained and quickly readable.
        cmd.extend(["-segment_format_options", "movflags=+faststart"])

    cmd.extend([
        # --- Output naming ---
        "-strftime", "1",
        output_pattern,
    ])

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

    fname_pattern = str(output_dir / "%Y%m%d_%H%M%S.mp4")

    cmd = build_ffmpeg_command(
        ffmpeg_path=ffmpeg_path,
        rtsp_url=rtsp_url,
        output_pattern=fname_pattern,
        segment_time=60,
        segment_format="mp4",
        include_audio=True,
        loglevel="info",
    )

    run_ffmpeg(cmd)

