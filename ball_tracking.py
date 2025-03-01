import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math

@dataclass
class Point:
    x: float
    y: float
    timestamp: float

class BallTracker:
    def __init__(self):
        self.cap = cv2.VideoCapture('red_ball1.mp4')
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Playback speed controller
        self.playback_speed = 1  # 1 = Normal speed, >1 = Slow, <1 = Fast

        # Tracking parameters
        self.mode = "tracking"  
        self.is_tracking = False
        self.trajectory_points: List[Point] = []

        # Bat detection parameters
        self.bat_y = 400  
        self.bat_height = 40  
        self.hit_count = 0  
        self.last_hit_time = 0  
        self.min_hit_interval = 0.3  
        self.was_outside_bat = True  

        # Red ball HSV color range
        self.lower_red1 = np.array([0, 120, 70])  
        self.upper_red1 = np.array([10, 255, 255])  
        self.lower_red2 = np.array([170, 120, 70])  
        self.upper_red2 = np.array([180, 255, 255])  

        # Constants for prediction
        self.GRAVITY = 9.81  
        self.PIXELS_TO_METERS = 0.002  
        self.FPS = 30  

    def detect_ball(self, frame) -> Optional[Tuple[int, int]]:
        """Detect the red ball using HSV thresholding"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = mask1 + mask2  

        # Reduce noise
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 100:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy)
        return None

    def detect_hit(self):
        """Detects if the ball has successfully hit the bat."""
        if len(self.trajectory_points) < 2:
            return

        p1 = self.trajectory_points[-2]  
        p2 = self.trajectory_points[-1]  

        inside_bat_region = self.bat_y <= p2.y <= self.bat_y + self.bat_height

        if self.was_outside_bat and inside_bat_region and p1.y > p2.y:
            current_time = time.time()
            if current_time - self.last_hit_time > self.min_hit_interval:
                self.hit_count += 1
                self.last_hit_time = current_time  
                self.was_outside_bat = False  

        if not inside_bat_region:
            self.was_outside_bat = True  

    def predict_future_positions(self) -> List[Tuple[int, int]]:
        """Predicts future positions using physics (gravity + velocity)"""
        if len(self.trajectory_points) < 5:
            return []

        p1, p2 = self.trajectory_points[-2], self.trajectory_points[-1]
        dt = p2.timestamp - p1.timestamp
        if dt == 0:
            return []

        vx = (p2.x - p1.x) / dt  
        vy = (p2.y - p1.y) / dt  

        future_positions = []
        for i in range(1, 16):
            t = i / self.FPS  
            future_x = int(p2.x + vx * t)
            future_y = int(p2.y + vy * t + 0.5 * self.GRAVITY * (t ** 2) / self.PIXELS_TO_METERS)

            if future_y > 480 or future_x < 0 or future_x > 640:
                break
            
            future_positions.append((future_x, future_y))

        return future_positions

    def draw_interface(self, frame, ball_pos=None):
        """Draw tracking interface, hits, and bat region"""
        cv2.putText(frame, f"Mode: {self.mode}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        status = "Tracking ON" if self.is_tracking else "Tracking OFF"
        cv2.putText(frame, status, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(frame, f"Hits: {self.hit_count}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.rectangle(frame, (0, self.bat_y), (640, self.bat_y + self.bat_height), (0, 255, 255), 2)
        cv2.putText(frame, "Bat Region", (10, self.bat_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if ball_pos:
            cv2.circle(frame, ball_pos, 20, (0, 0, 255), 2)

        for point in self.trajectory_points:
            cv2.circle(frame, (int(point.x), int(point.y)), 5, (0, 255, 0), -1)

        if self.mode == "prediction":
            future_positions = self.predict_future_positions()
            for pos in future_positions:
                cv2.circle(frame, pos, 5, (255, 0, 0), -1)  

    def run(self):
        """Main loop"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                ball_pos = None
                if self.is_tracking:
                    ball_pos = self.detect_ball(frame)
                    if ball_pos:
                        self.trajectory_points.append(Point(ball_pos[0], ball_pos[1], time.time()))
                        self.trajectory_points = self.trajectory_points[-10:]  
                        self.detect_hit()

                self.draw_interface(frame, ball_pos)

                cv2.imshow('Frame', frame)

                # Playback speed control
                delay = max(1, int(33 * self.playback_speed))  # Adjust delay based on speed
                key = cv2.waitKey(delay) & 0xFF  

                if key == ord('q'):
                    break
                elif key == ord('m'):
                    self.mode = "prediction" if self.mode == "tracking" else "tracking"
                elif key == ord('t'):
                    self.is_tracking = not self.is_tracking
                elif key == ord('r'):
                    self.hit_count = 0
                    self.trajectory_points.clear()
                    self.was_outside_bat = True  
                elif key == ord('+'):
                    self.playback_speed = max(0.1, self.playback_speed - 0.1)  
                elif key == ord('-'):
                    self.playback_speed += 0.1  

        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = BallTracker()
    tracker.run()
