#!/usr/bin/env python3
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import os

# Set OpenCV backend
os.environ["OPENCV_IMSHOW"] = "QT"

# --- Camera Intrinsics ---
CAMERA_MATRIX = np.array([[617.0, 0.0, 321.5],
                         [0.0, 617.0, 238.0],
                         [0.0, 0.0, 1.0]])
DIST_COEFFS = np.array([0.05, -0.1, 0.001, 0.002, 0.02])

# --- Model Paths ---
YOLO_MODEL_PATH = "/home/thunder/adhesh/opendroid/opendroid/detection_model/best.pt"
SAM_CHECKPOINT = "/home/thunder/adhesh/opendroid/opendroid/detection_model/sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"
CONFIDENCE_THRESHOLD = 0.25

def pixel_to_3d_point(u, v, depth, camera_matrix):
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    return np.array([X, Y, Z])

def initialize_camera():
    print("Initializing camera...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start pipeline and get depth scale
    cfg = pipeline.start(config)
    depth_sensor = cfg.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth scale: {depth_scale:.6f} meters/pixel")
    
    # Align depth to color
    align = rs.align(rs.stream.color)
    
    return pipeline, align, depth_scale  # Return depth_scale here

def main():
    print("Starting program...")
    
    # --- Load Models ---
    print("Loading YOLO model...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("Loading SAM model...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    predictor = SamPredictor(sam)
    
    # --- Initialize Camera ---
    pipeline, align, depth_scale = initialize_camera()  # Unpack depth_scale here
    
    try:
        while True:
            # Get frames
            frames = pipeline.wait_for_frames(5000)
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                print("Skipping incomplete frame")
                continue

            # Convert images
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image_bgr = np.asanyarray(color_frame.get_data())
            color_image_rgb = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)
            
            # Create display images
            display_image = color_image_bgr.copy()
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            
            # --- YOLO Detection ---
            yolo_results = yolo_model.predict(color_image_rgb, 
                                           conf=CONFIDENCE_THRESHOLD, 
                                           verbose=False)
            predictor.set_image(color_image_rgb)

            for result in yolo_results:
                if result.boxes and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = yolo_model.names[class_id]
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        # --- SAM Segmentation ---
                        input_point = np.array([[center_x, center_y]])
                        input_label = np.array([1])
                        masks, scores, _ = predictor.predict(
                            point_coords=input_point,
                            point_labels=input_label,
                            multimask_output=True,
                        )

                        best_mask_idx = np.argmax(scores)
                        best_mask = masks[best_mask_idx]

                        # Find centroid of mask
                        mask_indices = np.argwhere(best_mask)
                        if mask_indices.size > 0:
                            cy, cx = np.mean(mask_indices, axis=0).astype(int)

                            if 0 <= cy < depth_image.shape[0] and 0 <= cx < depth_image.shape[1]:
                                depth = depth_image[cy, cx] * depth_scale  # Now depth_scale is available
                                if depth > 0:
                                    center_3d = pixel_to_3d_point(cx, cy, depth, CAMERA_MATRIX)

                                    print(f"\nDetected: {class_name} ({confidence:.2f})")
                                    print(f"Pixel: ({cx}, {cy}) | Depth: {depth:.3f} m")
                                    print(f"3D Coordinates: X={center_3d[0]:.3f}, Y={center_3d[1]:.3f}, Z={center_3d[2]:.3f}")

                                    # Draw on image
                                    cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.circle(display_image, (center_x, center_y), 5, (0, 0, 255), -1)
                                    cv2.putText(display_image, f"{class_name} {confidence:.2f}",
                                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                               (0, 255, 0), 2)

                                    # Overlay mask
                                    alpha = 0.4
                                    color_mask = np.zeros_like(display_image, dtype=np.uint8)
                                    color_mask[best_mask] = [0, 0, 255]
                                    display_image = cv2.addWeighted(display_image, 1, color_mask, alpha, 0)

            # Display results
            cv2.imshow("YOLO + SAM + 3D", display_image)
            cv2.imshow("Depth View", depth_colormap)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error in main loop: {str(e)}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Pipeline stopped.")

if __name__ == "__main__":
    main()
