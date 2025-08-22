import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import json
import time
from datetime import datetime

json_file = '/home/thunder/adhesh/adhesh/src/arm/piper/jotst/final_code/v1/aruco_position.json'

def read_joint_values_from_json(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    joint_data = data.get("joint_angles", {})
    detected_joints = [
        joint_data.get("joint_1", 0.0),
        joint_data.get("joint_2", 0.0),
        joint_data.get("joint_3", 0.0),
        joint_data.get("joint_4", 0.0),
        joint_data.get("joint_5", 0.0),
        joint_data.get("joint_6", 0.0),
    ]
    
    return detected_joints

# Example usage:

detected_joints = read_joint_values_from_json(json_file)

def publish_target_pose_once(joint_gripper_data):
    rclpy.init()
    node = rclpy.create_node('target_pose_publisher')
    publisher = node.create_publisher(Float64MultiArray, '/target_pose', 10)

    msg = Float64MultiArray()
    msg.data = joint_gripper_data

    node.get_logger().info(f"Publishing target pose: {msg.data}")
    
    # Wait briefly to ensure publisher is ready
    rclpy.spin_once(node, timeout_sec=0)
    publisher.publish(msg)

    node.get_logger().info("Published!")
    node.destroy_node()
    rclpy.shutdown()

def read():
    with open("/home/thunder/adhesh/adhesh/src/arm/piper/jotst/final_v2.json", "r") as f:
        data = json.load(f)

    poses = data["poses"]

    trained_joints = []
    trained_gripper_pos = []
    timestamps = []
    for pose in poses:
        joints = pose["joints"]
        gripper = pose.get("gripper", {})
        gripper_position = gripper.get("position", 0.0)  # fallback 0.0 if absent

        time_obj = datetime.fromisoformat(pose["timestamp"])
        timestamps.append(time_obj)

        trained_joints.append([
            joints["joint_1"],
            joints["joint_2"],
            joints["joint_3"],
            joints["joint_4"],
            joints["joint_5"],
            joints["joint_6"],
        ])
        trained_gripper_pos.append(gripper_position)

    # Starting pose (detected pose)
    # (Use a starting gripper pos, up to you. Here we use the 1st frame's gripper as detected)
    
    #detected_joints = [0.06, 1.69, -1.15, -0.01, 1.19, 0.37]
    detected_joints = read_joint_values_from_json(json_file)
 
    detected_gripper_pos = trained_gripper_pos[0] if trained_gripper_pos else 0.0

    predicted = [detected_joints + [detected_gripper_pos, 2.0]]

    print("Predicted joint+gripper sequences:")
    for i in range(1, len(trained_joints)):
        # Compute delta for joints
        delta_joints = [trained_joints[i][j] - trained_joints[i-1][j] for j in range(6)]
        # Compute delta for gripper position
        delta_grip = trained_gripper_pos[i] - trained_gripper_pos[i-1]
        last = predicted[-1]
        # Apply deltas to last prediction
        next_joints = [round(last[j] + delta_joints[j], 4) for j in range(6)]
        next_gripper_position = round(last[6] + delta_grip, 4)
        next_gripper_effort = 2.0
        next_step = next_joints + [next_gripper_position, next_gripper_effort]
        predicted.append(next_step)

        # Delay based on recorded timestamps
        delay = (timestamps[i] - timestamps[i - 1]).total_seconds()
        delay = max(delay, 0.01)

        print(f"Step {i}: {next_step}")
        publish_target_pose_once(next_step)
        time.sleep(delay)

if __name__ == '__main__':
    read()
