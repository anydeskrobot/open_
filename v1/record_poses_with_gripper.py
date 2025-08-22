#!/usr/bin/env python3
"""
Simple script to record robot joint positions and gripper state with timestamps.
Outputs valid JSON format.
"""
import time
import json
import signal
import sys
from datetime import datetime
from piper_control import piper_connect, piper_control

class SimplePoseRecorder:
    def __init__(self, output_file: str = "robot_poses.json", can_port: str = "can0"):
        self.output_file = output_file
        # Extract just the port name if a tuple is provided
        self.can_port = can_port[0] if isinstance(can_port, (tuple, list)) else str(can_port)
        self.running = False
        self.robot = None
        self.joint_names = [f"joint_{i+1}" for i in range(6)]
        self.poses = []  # Store poses in memory

        # Set up signal handler
        signal.signal(signal.SIGINT, self.signal_handler)

    def connect_to_robot(self) -> bool:
        """Connect to the robot."""
        try:
            print(f"Connecting to robot on {self.can_port}...")
            self.robot = piper_control.PiperControl(can_port=self.can_port)
            self.robot.enable(timeout_seconds=5)
            print("Connected and enabled.")
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def record_pose(self) -> dict:
        """Record current joint positions and gripper state (position, effort) with timestamp."""
        if not self.robot:
            return None
        try:
            gripper_position, gripper_effort = self.robot.get_gripper_state()
            pose = {
                "timestamp": datetime.now().isoformat(),
                "joints": {
                    name: float(pos)
                    for name, pos in zip(
                        self.joint_names,
                        self.robot.get_joint_positions()
                    )
                },
                "gripper": {
                    "position": float(gripper_position),
                    "effort": float(gripper_effort)
                }
            }
            return pose
        except Exception as e:
            print(f"Error recording pose: {e}")
            return None

    def save_poses(self) -> None:
        """Save all recorded poses to file as a valid JSON array."""
        try:
            with open(self.output_file, 'w') as f:
                json.dump({
                    "metadata": {
                        "recorded_at": datetime.now().isoformat(),
                        "can_port": self.can_port,
                        "num_poses": len(self.poses)
                    },
                    "poses": self.poses
                }, f, indent=2)
            print(f"Successfully saved {len(self.poses)} poses to {self.output_file}")
        except Exception as e:
            print(f"Error saving poses: {e}")

    def signal_handler(self, signum, frame) -> None:
        """Handle interrupt signals."""
        print("\nStopping...")
        self.running = False

    def run(self, interval: float = 0.1) -> None:
        """Run the recording loop."""
        if not self.connect_to_robot():
            return

        self.running = True
        print("Recording started. Press Ctrl+C to stop...")

        try:
            while self.running:
                start_time = time.time()

                pose = self.record_pose()
                if pose:
                    self.poses.append(pose)
                    print(
                        f"Recorded: {pose['timestamp']} - "
                        + " ".join([f"{k}:{v:.3f}" for k, v in pose['joints'].items()])
                        + f" | gripper pos: {pose['gripper']['position']:.4f}, effort: {pose['gripper']['effort']:.4f}"
                    )

                # Maintain consistent interval
                elapsed = time.time() - start_time
                time.sleep(max(0, interval - elapsed))

        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.save_poses()
            print("Recording stopped.")

def find_available_ports() -> list:
    """Find and list available CAN ports."""
    try:
        ports = piper_connect.find_ports()
        return [port[0] if isinstance(port, (tuple, list)) else port for port in ports]
    except Exception as e:
        print(f"Error finding CAN ports: {e}")
        return []

def select_can_port() -> str:
    """Prompt user to select a CAN port."""
    ports = find_available_ports()
    if not ports:
        print("No CAN ports found. Exiting.")
        sys.exit(1)

    print("\nAvailable CAN ports:")
    for i, port in enumerate(ports):
        print(f"{i}: {port}")

    while True:
        try:
            selection = input("\nSelect CAN port number (or 'q' to quit): ").strip()
            if selection.lower() == 'q':
                sys.exit(0)

            selection = int(selection)
            if 0 <= selection < len(ports):
                return ports[selection]
            print(f"Please enter a number between 0 and {len(ports)-1}")
        except ValueError:
            print("Please enter a valid number.")

if __name__ == "__main__":
    # Create output filename with timestamp
    output_file = f"robot_poses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Let user select CAN port
    can_port = select_can_port()

    # Start recording
    recorder = SimplePoseRecorder(output_file=output_file, can_port=can_port)
    recorder.run(interval=0.1)  # Record at 10Hz
