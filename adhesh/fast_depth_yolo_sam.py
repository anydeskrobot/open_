#!/usr/bin/env python3
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import time

# Configuration
MODEL_PATHS = {
    'yolo': "best.pt",
    'sam': "sam_vit_b_01ec64.pth"
}
MODEL_TYPE = "vit_b"
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
    
    print("Loading models...")
    start_time = time.time()
    yolo = YOLO(MODEL_PATHS['yolo']).to('cuda' if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_PATHS['sam']).to('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = SamPredictor(sam)
    print(f"Models loaded in {time.time()-start_time:.2f}s")
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align = rs.align(rs.stream.color)
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    
    for _ in range(5):
        pipeline.wait_for_frames()
    
    try:
        while True:
            fps_monitor.update()
            start_time = time.time()
            
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
                
            color_img = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_img, alpha=0.03),
                cv2.COLORMAP_JET
            )
            
            results = yolo.predict(
                color_img,
                conf=CONF_THRESH,
                imgsz=640,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                verbose=False,
                half=True
            )
            
            predictor.set_image(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
            display_img = color_img.copy()
            
            for result in results:
                if not result.boxes:
                    continue
                    
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf.item()  # Get confidence score
                    
                    center = np.array([[(x1+x2)//2, (y1+y2)//2]])
                    masks, _, _ = predictor.predict(
                        point_coords=center,
                        point_labels=np.array([1]),
                        multimask_output=False
                    )
                    
                    mask = masks[0]
                    cy, cx = np.median(np.argwhere(mask), axis=0).astype(int)
                    depth = depth_img[cy, cx] * depth_scale
                    
                    point_2d = [cx, cy]
                    point_3d = rs.rs2_deproject_pixel_to_point(
                        depth_intrinsics,
                        point_2d,
                        depth
                    )
                    x, y, z = point_3d
                    
                    # Print coordinates and confidence to console
                    print(f"Detection - X: {x:.3f}m, Y: {y:.3f}m, Z: {z:.3f}m, Confidence: {conf:.2f}")
                    
                    # Visualization
                    cv2.rectangle(display_img, (x1,y1), (x2,y2), (0,255,0), 2)
                    display_img[mask] = display_img[mask]*0.7 + np.array([0,0,255])*0.3
                    
                    # Display info on image
                    info_text = f"X:{x:.2f}m Y:{y:.2f}m Z:{z:.2f}m C:{conf:.2f}"
                    cv2.putText(display_img, info_text, 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (0,255,0), 2)
            
            fps = fps_monitor.get_fps()
            cv2.putText(display_img, f"FPS: {fps:.1f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            cv2.imshow("Detection", display_img)
            cv2.imshow("Depth", depth_colormap)
            
            elapsed = time.time() - start_time
            delay = max(1, int(1000*(1/TARGET_FPS - elapsed)))
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import torch
    main()
