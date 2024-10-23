import subprocess
from datetime import timedelta

input_video = "howard_video.mp4"
timestamp_file = "output.txt"

def time_to_seconds(time_str):
    h, m, s = map(int, time_str.strip().split(':'))
    return h * 3600 + m * 60 + s

with open(timestamp_file, "r") as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    line = line.strip()
    if not line:
        continue  # Skip empty lines

    start_time_str, end_time_str = line.split("->")
    start_time_str = start_time_str.strip()
    end_time_str = end_time_str.strip()

    start_seconds = time_to_seconds(start_time_str)
    end_seconds = time_to_seconds(end_time_str)
    duration_seconds = end_seconds - start_seconds

    output_video = f"clip_{idx+1}.mp4"

    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-loglevel", "quiet",
        "-ss", start_time_str,
        "-i", input_video,
        "-t", str(duration_seconds),
        "-c:v", "libx264",
        "-c:a", "aac",
        output_video
    ]

    print(f"Extracting clip: {output_video} from {start_time_str} to {end_time_str}")
    subprocess.run(ffmpeg_command)
