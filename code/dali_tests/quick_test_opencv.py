"""Quick OpenCV video test - fast pre-screening"""
import cv2
import sys

def test_video(video_path):
    """Quick test to check if video is readable"""
    print(f"Testing video: {video_path}")

    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("✗ Failed to open video")
            return False

        # Get basic info
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"  Frames: {frame_count}")
        print(f"  FPS: {fps}")
        print(f"  Resolution: {width}x{height}")

        # Try to read some frames
        frames_read = 0
        for i in range(min(10, frame_count)):
            ret, frame = cap.read()
            if not ret:
                print(f"✗ Failed to read frame {i}")
                cap.release()
                return False
            frames_read += 1

        cap.release()
        print(f"✓ Successfully read {frames_read} frames")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_test_opencv.py <video_file>")
        sys.exit(1)

    video_path = sys.argv[1]
    success = test_video(video_path)
    sys.exit(0 if success else 1)
