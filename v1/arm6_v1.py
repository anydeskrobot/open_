#!/usr/bin/env python3

# get data from cam x,y,z and joint to train model for calibration

import pyrealsense2 as rs
import numpy as np
import cv2
from piper_control import piper_control, piper_connect
import time
import json
import cv2.aruco as aruco

def init_robot():
    ports = piper_connect.find_ports()
    available = [p[0] for p in ports]
    can_port = "can0"

    if can_port not in available:
        raise RuntimeError(f"{can_port} not found. Available: {available}")

    can_tuple = next(p for p in ports if p[0] == can_port)
    piper_connect.activate(ports=[can_tuple])
    robot = piper_control.PiperControl(can_port=can_port)

    robot.set_emergency_stop(piper_control.EmergencyStop.STOP)
    time.sleep(3)

    try:
        robot.reset(timeout_seconds=30)
    except:
        robot.enable(timeout_seconds=30)

    robot.set_joint_zero_positions(range(6))
    return robot


def detect_red_objects(robot):
    # ArUco dictionary and detector setup
    ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_7X7_250)
    parameters = aruco.DetectorParameters()

    # RealSense stream setup
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    detections = []

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # Detect ArUco markers
            corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT, parameters=parameters)

            if ids is not None:
                aruco.drawDetectedMarkers(color_image, corners, ids)

                for i in range(len(ids)):
                    c = corners[i][0]
                    cx = int(np.mean(c[:, 0]))
                    cy = int(np.mean(c[:, 1]))

                    depth = depth_frame.get_distance(cx, cy)
                    if depth == 0:
                        continue

                    cam_point = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], depth)
                    cam_x, cam_y, cam_z = [round(p, 6) for p in cam_point]

                    joint_values = robot.get_joint_positions()
                    joint_values = [round(j, 6) for j in joint_values]

                    detection = {
                        "id": int(ids[i]),
                        "cam_x": cam_x,
                        "cam_y": cam_y,
                        "cam_z": cam_z,
                        "joint_1": joint_values[0],
                        "joint_2": joint_values[1],
                        "joint_3": joint_values[2],
                        "joint_4": joint_values[3],
                        "joint_5": joint_values[4],
                        "joint_6": joint_values[5]
                    }

                    print("[INFO] ArUco Marker Detection:", detection)
                    detections.append(detection)

                    cv2.putText(color_image, f"ID:{ids[i][0]} Z:{cam_z:.2f}m", (cx - 20, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                print("[INFO] No ArUco markers detected.")

            cv2.imshow('ArUco Marker Detection', color_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] Quit signal received.")
                break
            elif key == 13:  # Enter key
                if detections:
                    with open("aruco_detections.json", "w") as f:
                        json.dump(detections, f, indent=2)
                    print("[SUCCESS] Saved detections to aruco_detections.json")
                else:
                    print("[WARNING] No detections yet to save.")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    robot = init_robot()
    detect_red_objects(robot)

