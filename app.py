import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, ImageSequenceClip

class LaMaInpaintingModel:
    def __init__(self, model_path):
        # Load the actual LaMa model here
        self.model_path = model_path

    def inpaint(self, frame):
        original_height, original_width, _ = frame.shape
        new_width = int(original_height * 16 / 9)
        if new_width <= original_width:
            return frame

        left_padding = (new_width - original_width) // 2

        new_frame = np.zeros((original_height, new_width, 3), dtype=np.uint8)
        new_frame[:, left_padding:left_padding + original_width] = frame

        if left_padding > 0:
            new_frame[:, :left_padding] = np.flip(frame[:, :left_padding], axis=1)
            new_frame[:, left_padding + original_width:] = np.flip(frame[:, -left_padding:], axis=1)

        return new_frame

# Initialize the LaMa model
lama_model = LaMaInpaintingModel('D:\Placements\Assignments\kchbhi\big-lama')

def inpaint_frame(frame):
    if frame.shape[2] == 4:  # If the image has an alpha channel, remove it
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return lama_model.inpaint(frame)

def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_path, frame)
        frames.append(frame_path)
        frame_count += 1

    cap.release()
    return frames

def inpaint_frames(frame_paths, output_folder):
    inpainted_frames = []

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        inpainted_frame = inpaint_frame(frame)
        inpainted_frame_path = os.path.join(output_folder, os.path.basename(frame_path))
        cv2.imwrite(inpainted_frame_path, inpainted_frame)
        inpainted_frames.append(inpainted_frame_path)

    return inpainted_frames

def frames_to_video(frame_paths, output_video_path, fps):
    clip = ImageSequenceClip(frame_paths, fps=fps)
    clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

def convert_video(input_video_path, output_video_path):
    frames_folder = 'frames'
    inpainted_frames_folder = 'inpainted_frames'

    frame_paths = extract_frames(input_video_path, frames_folder)
    inpainted_frame_paths = inpaint_frames(frame_paths, inpainted_frames_folder)

    clip = VideoFileClip(input_video_path)
    fps = clip.fps
    frames_to_video(inpainted_frame_paths, output_video_path, fps)

if __name__ == "__main__":
    input_video_path = "test.mp4"
    output_video_path = "output_video_2.mp4"
    convert_video(input_video_path, output_video_path)
