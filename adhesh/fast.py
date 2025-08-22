#!/usr/bin/env python3
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time

# Configuration
YOLO_MODEL_PATH = "best.pt"
CONF_THRESH = 0.25
TARGET_FPS = 15

class FPSMonitor:
    def __init__(self):
        self.times = []
        self.window_size = 10
        
    def update(self):
        self.times.append(time.time())
        if len(self.times) > self.window_size:
            self.times.pop(0)
            
    def get_fps(self):
        if len(self.times) < 2:
            return 0
        return (len(self.times)-1)/(self.times[-1]-self.times[0])

def main():
    fps_monitor = FPSMonitor()
    
    print("Loading YOLO model...")
    yolo = YOLO(YOLO_MODEL_PATH).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # RealSense setup with corrected scaling
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    
    # Get depth scale and align
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()  # Convert from mm to meters
    align = rs.align(rs.stream.color)
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    color_intrinsics = color_profile.get_intrinsics()
    
    try:
        while True:
            fps_monitor.update()
            
            # Get frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
                
            color_img = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())
            
            # YOLO detection
            results = yolo.predict(color_img, conf=CONF_THRESH, verbose=False)
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf.item()
                    
                    # Calculate center point
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    # Get depth in meters (matches ArUco code)
                    depth = depth_frame.get_distance(cx, cy)
                    if depth == 0:
                        continue
                    
                    # Convert to 3D coordinates (using color intrinsics like ArUco)
                    x, y, z = rs.rs2_deproject_pixel_to_point(
                        color_intrinsics,
                        [cx, cy],
                        depth
                    )
                    
                    # Print in ArUco-style format with meters
                    print(f"Object detected - X: {x:.6f}, Y: {y:.6f}, Z: {z:.6f}")
                    
                    # Visualization (optional)
                    cv2.rectangle(color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(color_img, f"Z: {z:.3f}m", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
            # Show FPS
            fps = fps_monitor.get_fps()
            cv2.putText(color_img, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            cv2.imshow("Detection", color_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import torch
    main()
