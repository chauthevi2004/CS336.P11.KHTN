import os
import cv2
import pandas as pd
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import AdaptiveDetector
from scenedetect.frame_timecode import FrameTimecode
import matplotlib.pyplot as plt
import time


def process_all_videos_in_folder(folder_path):
    csv_paths = []
    total_time = 0

    for video_file in os.listdir(folder_path):
        video_path = os.path.join(folder_path, video_file)
        if os.path.isfile(video_path):
            start_time = time.time()
            csv_path = process_video(video_path)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Processed {video_file} in {elapsed_time:.2f} seconds")
            csv_paths.append(csv_path)
            total_time += elapsed_time

    print(f"Total processing time for all videos: {total_time:.2f} seconds")        
    return csv_paths

def process_video(video_path):
    # Extract the video name and create an output directory
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = f"{video_name}_frames"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize video manager and scene manager
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=0.7))
    
    # Start the video manager and perform scene detection
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    
    # Prepare to capture the frames
    cap = cv2.VideoCapture(video_path)
    frame_data = []

    # Iterate through each scene and save the first frame
    for scene in scene_list:
        start_frame = scene[0].get_frames()
        end_frame = scene[1].get_frames()
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = cap.read()
        if ret:
            frame_filename = f"{video_name}.{start_frame}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_data.append((frame_filename, FrameTimecode(start_frame, cap.get(cv2.CAP_PROP_FPS)).get_timecode()))
    
    # Save frame details to CSV
    df = pd.DataFrame(frame_data, columns=['Filename', 'Timestamp'])
    csv_path = os.path.join(output_dir, f"{video_name}_frames.csv")
    df.to_csv(csv_path, index=False)
    
    print(df.shape)
    return csv_path

folder_path = 'C:\\Users\\chaut\\Downloads\\Data\\L01_V001.mp4'
csv_paths = process_all_videos_in_folder(folder_path)