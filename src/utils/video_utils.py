import cv2

def extract_frames(video_path, output_folder, frame_interval=1):
    video = cv2.VideoCapture(video_path)
    count = 0
    frame_count = 0
    
    while True:
        success, frame = video.read()
        if not success:
            break
        
        if count % frame_interval == 0:
            cv2.imwrite(f"{output_folder}/frame_{frame_count:04d}.jpg", frame)
            frame_count += 1
        
        count += 1
    
    video.release()
    return frame_count

if __name__ == "__main__":
    video_path = "path/to/your/video.mp4"
    output_folder = "path/to/output/frames"
    total_frames = extract_frames(video_path, output_folder)
    print(f"Extracted {total_frames} frames")