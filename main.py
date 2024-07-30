import cv2
import numpy as np
from src.tracking.ball_tracker import BallTracker
from src.tracking.player_tracker import PlayerTracker
from src.analysis.football_analyzer import FootballAnalyzer

def main():
    video_path = "data/video.mp4"

    ball_tracker = BallTracker(force_reload=True)
    player_tracker = PlayerTracker()
    football_analyzer = FootballAnalyzer()

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    skip_frames = 2  # Process every 2nd frame
    scale_percent = 50  # Percent of original size

    while True:
        for _ in range(skip_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

        if not ret:
            break

        # Resize frame
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        player_boxes, ball_box = ball_tracker.detect(frame)
        tracked_players, team_colors = player_tracker.update(player_boxes, frame)

        ball_center = None
        if ball_box is not None:
            ball_center = ((ball_box[0] + ball_box[2]) // 2, (ball_box[1] + ball_box[3]) // 2)

        if ball_center and tracked_players:
            football_analyzer.update(frame_count, ball_center, tracked_players, team_colors)

        stats = football_analyzer.get_stats()
        
        print("Current stats:", stats)
        
        spain_passes = stats.get('passes', {}).get("Spain", 0)
        portugal_passes = stats.get('passes', {}).get("Portugal", 0)
        total_passes = spain_passes + portugal_passes
        

        # Display stats on frame
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display possession
        y_offset = 60
        for team, possession in stats['possession'].items():
            color = (255, 255, 255) if team == "Spain" else (0, 0, 255)
            cv2.putText(frame, f"{team} possession: {possession:.2%}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30

        # Display passes
        frame_width = frame.shape[1]
        spain_passes = stats['passes']["Spain"]
        portugal_passes = stats['passes']["Portugal"]
        total_passes = spain_passes + portugal_passes

        cv2.putText(frame, f"Spain: {spain_passes}", (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Total: {total_passes}", (frame_width // 2 - 50, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Portugal: {portugal_passes}", (frame_width - 160, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, f"Ball distance: {stats['ball_distance']:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw players and ball
        for (objectID, centroid) in tracked_players.items():
            color = team_colors[objectID]
            cv2.circle(frame, (centroid[0], centroid[1]), 4, color, -1)
            cv2.putText(frame, f"ID {objectID}", (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if ball_box is not None:
            cv2.rectangle(frame, (ball_box[0], ball_box[1]), (ball_box[2], ball_box[3]), (0, 165, 255), 2)
            cv2.circle(frame, ball_center, 5, (0, 165, 255), -1)

        cv2.imshow("Football Analysis", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Print final stats
    print("\nFinal Stats:")
    for stat, value in stats.items():
        print(f"{stat.capitalize()}: {value}")

if __name__ == "__main__":
    main()