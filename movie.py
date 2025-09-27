from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
import os

# !!! IMPORTANT: Replace 'your_video.mp4' with the actual name of a video file
# that you have placed inside the 'uploads' folder.
video_filename = 'your_video.mp4'

# Construct the full path
# This assumes 'movie.py' is in 'backend' and 'uploads' is also in 'backend'
video_file_path = os.path.join('uploads', video_filename) 

print(f"Loading video from: {video_file_path}")

try:
    # Attempt to load the video
    clip = VideoFileClip(video_file_path)
    print(f"Video loaded successfully! Duration: {clip.duration} seconds")

    # Always remember to close the clip to release the file handle
    clip.close() 
    print("Video clip closed.")
except OSError as e:
    print(f"Error: MoviePy error: the file {video_file_path} could not be found! {e}")
    print("Please check that you entered the correct path and the file exists.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

def add_watermark(input_path, output_path, logo_path="watermark.png", pos=("right", "bottom")):
    video = VideoFileClip(input_path)
    logo = (
        ImageClip(logo_path)
        .set_duration(video.duration)
        .resize(height=int(video.h * 0.12))  # 12% of video height, adjust as needed
        .margin(right=20, bottom=20, opacity=0)
        .set_pos(pos)
        .set_opacity(0.7)  # semi-transparent
    )
    final = CompositeVideoClip([video, logo])
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")
