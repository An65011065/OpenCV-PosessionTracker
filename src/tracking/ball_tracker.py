import torch

class BallTracker:
    def __init__(self, force_reload=True):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=force_reload)
        self.model.classes = [0, 32]  # 0 for person, 32 for sports ball

    def detect(self, frame):
        results = self.model(frame)
        
        player_boxes = []
        ball_box = None

        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det.tolist()
            if int(cls) == 0 and conf > 0.5:  # person
                player_boxes.append([int(x1), int(y1), int(x2), int(y2)])
            elif int(cls) == 32 and conf > 0.3:  # sports ball (lower threshold)
                ball_box = [int(x1), int(y1), int(x2), int(y2)]

        return player_boxes, ball_box