import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

def run_gripper_listener():
    rclpy.init()

    node = Node('gripper_listener_node')

    def callback(msg):
        if len(msg.data) >= 2:
            position = msg.data[0]
            effort = msg.data[1]
            print(f"[INFO] Gripper position: {position:.4f}, effort: {effort:.4f}")

            if position <= -0.0200:
                print("✅ Gripper is closed")
            else:
                print("🟢 Gripper is open")

            if effort <= -0.6110:
                print("📦 Object is picked")
        else:
            node.get_logger().warn("Received incomplete gripper data")

    node.create_subscription(Float64MultiArray, '/gripper', callback, 10)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down gripper listener.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    run_gripper_listener()

