#!/usr/bin/env python3



# if aruco is detected get joint values from model if auco CONSISTENT_COUNT = 5  , publish the joint value to "subscribe_wth_gripper.py"  topic_name :/target_pose  and run the  "publish_move_with_gripper.py" for "BC IL "

import pyrealsense2 as rs
import numpy as np
import cv2
import json
import cv2.aruco as aruco
import joblib
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import os



import subprocess
import time
import signal

PID_FILE = "bg_script.pid"  # Path to store the background script PID


def run_script_in_background(script_path: str, timeout: float = 3.0):
    """
    Launch a Python script in a completely detached background process,
    and save its PID to a file.
    """
    try:
        process = subprocess.Popen(
            ['python3', script_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True  # Detach from parent
        )

        # Save PID to file
        with open(PID_FILE, "w") as f:
            f.write(str(process.pid))

        print(f"[...] Launched {script_path}, waiting {timeout}s to verify it's running...")
        time.sleep(timeout)

        if process.poll() is None:
            print(f"[✔] Script is running in background (PID: {process.pid})")
            return process
        else:
            print("[✘] Script exited prematurely.")
            return None

    except Exception as e:
        print(f"[!] Failed to launch script: {e}")
        return None

  # stop_background_script()

def stop_background_script(pid_file: str = PID_FILE):
    """
    Stop the background script using the PID stored in a file.
    """
    try:
        if not os.path.exists(pid_file):
            print(f"[!] PID file '{pid_file}' not found.")
            return

        with open(pid_file, "r") as f:
            pid = int(f.read().strip())

        os.kill(pid, signal.SIGTERM)  # Use SIGKILL for force kill
        print(f"[✔] Sent SIGTERM to background script (PID: {pid})")

        # Optionally remove the PID file
        os.remove(pid_file)

    except ProcessLookupError:
        print(f"[!] No process found with PID {pid} (may have already exited).")
    except Exception as e:
        print(f"[!] Error stopping script: {e}")



class ArmCalibrator:
    def __init__(self, model_path='arm_calibration_model.pkl'):
        saved_data = joblib.load(model_path)
        self.model = saved_data['model']
        self.scaler = saved_data['scaler']
    
    def predict_joint_angles(self, cam_x, cam_y, cam_z):
        """Predict joint angles from camera coordinates and return values rounded to 2 decimal places"""
        cam_coords = np.array([[cam_x, cam_y, cam_z]])
        cam_coords_scaled = self.scaler.transform(cam_coords)
        joint_angles = self.model.predict(cam_coords_scaled)[0]
        
        return {
            'joint_1': round(joint_angles[0], 2),
            'joint_2': round(joint_angles[1], 2),
            'joint_3': round(joint_angles[2], 2),
            'joint_4': round(joint_angles[3], 2),
            'joint_5': round(joint_angles[4], 2),
            'joint_6': round(joint_angles[5], 2)
        }

class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        self.publisher_ = self.create_publisher(Float64MultiArray, '/target_pose', 10)
        self.calibrator = ArmCalibrator()
        
        # Setup RealSense
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        
        # Aruco setup
        self.ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_7X7_250)
        self.parameters = aruco.DetectorParameters()
        
        # Variables for tracking consistent detections
        self.last_position = None
        self.consistent_count = 0
        self.REQUIRED_CONSISTENT_COUNT = 5  # Changed from 10 to 5 as requested
        self.publishing_enabled = True
        
    def is_position_same(self, current_pos, last_pos, tolerance=0.001):
        """Check if the current position is the same as the last position within a tolerance"""
        if last_pos is None:
            return False
        return (abs(current_pos[0] - last_pos[0]) < tolerance and
                abs(current_pos[1] - last_pos[1]) < tolerance and
                abs(current_pos[2] - last_pos[2]) < tolerance)
    
    def save_to_json_and_exit(self, position_data):
        """Save the position data to JSON and exit the program"""
        output_file = "aruco_position.json"
        with open(output_file, 'w') as f:
            json.dump(position_data, f, indent=4)
        self.get_logger().info(f"Saved position data to {output_file}")
        

        process = run_script_in_background("publish_move_with_gripper.py")

        if process:
            print("Script running in background with PID:")
        else:
            print("Script failed to launch.")

        time.sleep(20)
        # Clean up resources
        self.pipeline.stop()
        cv2.destroyAllWindows()
        self.destroy_node()
        rclpy.shutdown()
        exit(0)
    
    def detect_and_publish(self):
        try:
            while rclpy.ok():
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = aruco.detectMarkers(gray, self.ARUCO_DICT, parameters=self.parameters)

                if ids is not None:
                    aruco.drawDetectedMarkers(color_image, corners, ids)

                    for i in range(len(ids)):
                        c = corners[i][0]
                        cx = int(np.mean(c[:, 0]))
                        cy = int(np.mean(c[:, 1]))

                        depth = depth_frame.get_distance(cx, cy)
                        if depth == 0:
                            continue

                        cam_point = rs.rs2_deproject_pixel_to_point(self.intr, [cx, cy], depth)
                        cam_x, cam_y, cam_z = [round(p, 6) for p in cam_point]
                        current_position = (cam_x, cam_y, cam_z)

                        # Check if position is the same as last time
                        if self.is_position_same(current_position, self.last_position):
                            self.consistent_count += 1
                        else:
                            self.consistent_count = 0
                            self.publishing_enabled = True
                        
                        self.last_position = current_position

                        self.get_logger().info(
                            f"Marker ID: {ids[i][0]} | cam_x: {cam_x}, cam_y: {cam_y}, cam_z: {cam_z} | "
                            f"Consistent count: {self.consistent_count}"
                        )

                        # Predict joint angles using model
                        joints = self.calibrator.predict_joint_angles(cam_x, cam_y, cam_z)
                        self.get_logger().info(
                            f"Predicted Joint Angles -> J1: {joints['joint_1']}, J2: {joints['joint_2']}, "
                            f"J3: {joints['joint_3']}, J4: {joints['joint_4']}, "
                            f"J5: {joints['joint_5']}, J6: {joints['joint_6']}"
                        )

                        # Publish to ROS2 topic only if we haven't reached the consistent count threshold
                        if self.publishing_enabled:
                            msg = Float64MultiArray()
                            msg.data = [
                                joints['joint_1'], joints['joint_2'], joints['joint_3'],
                                joints['joint_4'], joints['joint_5'], joints['joint_6'], 0.8, 2.0
                            ]
                            self.publisher_.publish(msg)
                            self.get_logger().info(f"Published to /target_pose: {msg.data}")
                            
                            # Save to JSON and exit if we've had enough consistent detections
                            if self.consistent_count >= self.REQUIRED_CONSISTENT_COUNT:
                                position_data = {
                                    "marker_id": int(ids[i][0]),
                                    "camera_coordinates": {
                                        "x": cam_x,
                                        "y": cam_y,
                                        "z": cam_z
                                    },
                                    "joint_angles": joints,
                                    "consistent_detections": self.consistent_count
                                }
                                self.save_to_json_and_exit(position_data)



                        cv2.putText(color_image, f"ID:{ids[i][0]}", (cx - 20, cy - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    self.get_logger().info("No ArUco markers detected.")
                    # Reset counters when no marker is detected
                    self.last_position = None
                    self.consistent_count = 0
                    self.publishing_enabled = True

                cv2.imshow('ArUco Marker Detection + Prediction', color_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.get_logger().info("Quit signal received.")
                    break

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    aruco_detector = ArucoDetector()
    aruco_detector.detect_and_publish()
    aruco_detector.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
