#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import time
import subprocess
from typing import Sequence
from piper_control import piper_control, piper_connect


class PiperCartesianMover(Node):
    def __init__(self):
        super().__init__('piper_cartesian_mover')

        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/target_pose',
            self.listener_callback,
            10
        )

        # Gripper publisher
        self.gripper_pub = self.create_publisher(Float64MultiArray, 'gripper', 10)

        self.robot_initialized = False
        self.robot = None
        self.can_port = "can0"

        self.get_logger().info('Waiting for /target_pose...')
        self.init_robot()
        self.timer = self.create_timer(1.0, self.read_gripper_state)


    def reset_can_bus(self):
        self.get_logger().info(f"Resetting CAN bus: {self.can_port}")
        try:
            subprocess.run(["sudo", "ip", "link", "set", self.can_port, "down"], check=True)
            time.sleep(1)
            subprocess.run(["sudo", "ip", "link", "set", self.can_port, "type", "can", "bitrate", "1000000"], check=True)
            subprocess.run(["sudo", "ip", "link", "set", self.can_port, "up"], check=True)
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f"CAN reset error: {e}")
            raise

    def init_robot(self):
        ports = piper_connect.find_ports()
        available = [p[0] for p in ports]
        if self.can_port not in available:
            self.get_logger().error(f"{self.can_port} not found in ports: {available}")
            raise RuntimeError("CAN port not available")

        can_tuple = next(p for p in ports if p[0] == self.can_port)
        piper_connect.activate(ports=[can_tuple])
        self.robot = piper_control.PiperControl(can_port=self.can_port)

        self.get_logger().info("Performing emergency descent...")
        try:
            self.robot.set_emergency_stop(piper_control.EmergencyStop.STOP)
            time.sleep(5)
        except Exception as e:
            self.get_logger().error(f"Emergency stop failed: {e}")

        self.get_logger().info("Resetting the arm...")
        for attempt in range(3):
            try:
                self.robot.reset(timeout_seconds=30)
                break
            except TimeoutError as e:
                self.get_logger().warn(f"Reset failed (attempt {attempt + 1}): {e}")
                self.reset_can_bus()
                piper_connect.activate(ports=[can_tuple])
                time.sleep(2)
        else:
            self.get_logger().info("Using enable() as fallback")
            self.robot.enable(timeout_seconds=30)

        self.get_logger().info("Homing the arm...")
        home_pos = [0.0] * 6
        self.robot.set_joint_positions(home_pos, wait_for_completion=True)
        start = time.time()
        while self.robot.motion_status != piper_control.MotionStatus.REACHED_TARGET:
            if time.time() - start > 20:
                self.get_logger().warn("Homing timeout")
                break
            time.sleep(0.1)

        self.robot.set_joint_zero_positions(range(6))
        self.robot_initialized = True
        self.get_logger().info("Robot initialized and ready.")

    def listener_callback(self, msg: Float64MultiArray):
        if not self.robot_initialized:
            self.get_logger().warn("Robot not initialized yet.")
            return

        if len(msg.data) != 8:
            self.get_logger().error(
                "Received pose must be a list of 8 values: [joint1..joint6, gripper_pos, gripper_effort]")
            return

        joints = list(msg.data[:6])
        gripper_position = msg.data[6]
        gripper_effort = msg.data[7]

        self.get_logger().info(
            f"Moving to joints: {joints}; gripper position: {gripper_position}; effort: {gripper_effort}")

        self.move_to_pose(joints, gripper_position, gripper_effort)
    

    def read_gripper_state(self):
        angle, effort = self.robot.get_gripper_state()
        state = self.classify_gripper_state(angle, effort)
        self.get_logger().info(
            f"Gripper Angle: {angle:.4f} m, Effort: {effort:.3f} Nm → State: {state}")
        # Publish to /gripper topic
        msg = Float64MultiArray()
        msg.data = [angle, effort]
        self.gripper_pub.publish(msg)
        

    def classify_gripper_state(self, angle: float, effort: float) -> str:
        if angle > 0.025 and effort < 0.2:
            return "OPEN"
        elif angle < 0.0 and effort > 0.2:
            return "CLOSED"
        else:
            return "MOVING or PARTIALLY CLOSED"

    def move_to_pose(self, pose: Sequence[float], gripper_position: float = 0.0, gripper_effort: float = 2.0):
        self.robot.set_joint_positions(pose, wait_for_completion=True)
        print("Waiting for the arm to reach the target position")
        start_time = time.time()
        timeout = 20
        while self.robot.motion_status != piper_control.MotionStatus.REACHED_TARGET:
            current_joints = self.robot.get_joint_positions()
            print(f"Current joint positions: {current_joints}, Motion status: {self.robot.motion_status}")
            time.sleep(0.1)
            if time.time() - start_time > timeout:
                print("Timeout waiting for motion to complete.")
                break

        # --- GRIPPER CONTROL ---
        try:
            self.robot.set_gripper_ctrl(gripper_position, gripper_effort)
            print(f"Set gripper pos={gripper_position}, effort={gripper_effort}")

            # Publish to /gripper topic
            msg = Float64MultiArray()
            msg.data = [gripper_position, gripper_effort]
            self.gripper_pub.publish(msg)

        except AttributeError:
            try:
                self.robot.set_gripper_position(gripper_position)
                print(f"Set gripper pos={gripper_position} (effort not set)")

                # Still publish what we set
                msg = Float64MultiArray()
                msg.data = [gripper_position, gripper_effort]
                self.gripper_pub.publish(msg)

            except Exception:
                print("Gripper control not available for this robot API.")

def main(args=None):
    rclpy.init(args=args)
    node = PiperCartesianMover()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
