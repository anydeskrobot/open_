#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import re
import math

class PrecisionWallFollower(Node):
    def __init__(self):
        super().__init__('precision_wall_follower')
        
        # Create publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscribe to LiDAR data
        self.lidar_sub = self.create_subscription(
            String,
            '/lidar_data',
            self.lidar_callback,
            10)
        
        # Target parameters (using your exact specified speeds)
        self.target_speed = 0.11037467024297616  # m/s
        self.target_turn = 0.22074934048595232   # rad/s
        
        # Distance thresholds
        self.min_distance = 0.30  # meters
        self.max_distance = 0.40  # meters
        self.target_angle = 90.0   # degrees
        self.angle_tolerance = 2.0 # degrees (tighter tolerance)
        
        # Control variables
        self.current_distance = 0.0
        self.current_angle = 0.0
        
        # Create timer for continuous command publishing (20Hz)
        self.timer = self.create_timer(0.05, self.publish_velocity)
        
    def lidar_callback(self, msg):
        try:
            # Parse the LiDAR data message
            data = msg.data
            distance_match = re.search(r'Distance: ([\d.]+) m', data)
            angle_match = re.search(r'Angle: ([\d.]+)°', data)
            
            if distance_match and angle_match:
                self.current_distance = float(distance_match.group(1))
                self.current_angle = float(angle_match.group(1))
                
                self.get_logger().info(
                    f"Distance: {self.current_distance:.3f}m | "
                    f"Angle: {self.current_angle:.2f}°"
                )
        except Exception as e:
            self.get_logger().error(f"Error parsing LiDAR data: {e}")
    
    def publish_velocity(self):
        cmd = Twist()
        angle_ok = math.isclose(
            self.current_angle,
            self.target_angle,
            abs_tol=self.angle_tolerance
        )
        
        if not angle_ok:
            # Rotate to align with wall (at your specified turn rate)
            cmd.linear.x = 0.0
            cmd.angular.z = self.target_turn
            self.get_logger().info(f"Rotating to align: {cmd.angular.z:.5f}rad/s")
            
        elif self.current_distance < self.min_distance:
            # Too close - move backward at half speed
            cmd.linear.x = -self.target_speed * 0.5
            cmd.angular.z = 0.0
            self.get_logger().warning(
                f"Too close! Backing up: {cmd.linear.x:.5f}m/s | "
                f"Current: {self.current_distance:.3f}m"
            )
            
        elif self.current_distance > self.max_distance:
            # Too far - move forward at reduced speed
            speed_reduction = min(1.0, (self.current_distance - self.max_distance) * 2)
            cmd.linear.x = self.target_speed * speed_reduction
            cmd.angular.z = 0.0
            self.get_logger().warning(
                f"Too far! Approaching: {cmd.linear.x:.5f}m/s | "
                f"Current: {self.current_distance:.3f}m"
            )
            
        else:
            # Perfect distance - move forward at exact target speed
            cmd.linear.x = self.target_speed
            cmd.angular.z = 0.0
            self.get_logger().info(
                f"Optimal distance! Moving: {cmd.linear.x:.5f}m/s | "
                f"Current: {self.current_distance:.3f}m"
            )
        
        self.cmd_vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = PrecisionWallFollower()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the robot before exiting
        stop_cmd = Twist()
        node.cmd_vel_pub.publish(stop_cmd)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
