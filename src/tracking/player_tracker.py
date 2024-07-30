import cv2
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict

class PlayerTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.team_colors = OrderedDict()

    def register(self, centroid, frame):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.team_colors[self.nextObjectID] = self.assign_team_color(centroid, frame)
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.team_colors[objectID]

    def assign_team_color(self, centroid, frame):
        x, y = centroid
        roi = frame[max(0, y-20):min(frame.shape[0], y+20), max(0, x-20):min(frame.shape[1], x+20)]
        
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for white and red in HSV
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        
        white_mask = cv2.inRange(hsv_roi, white_lower, white_upper)
        red_mask1 = cv2.inRange(hsv_roi, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv_roi, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        white_count = cv2.countNonZero(white_mask)
        red_count = cv2.countNonZero(red_mask)
        
        if white_count > red_count:
            return (255, 255, 255)  # White
        else:
            return (0, 0, 255)  # Red

    def update(self, rects, frame):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects, self.team_colors

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], frame)
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                self.team_colors[objectID] = self.assign_team_color(inputCentroids[col], frame)

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], frame)

        return self.objects, self.team_colors