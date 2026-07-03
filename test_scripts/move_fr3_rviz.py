import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class FR3RvizMover(Node):
    def __init__(self):
        super().__init__("fr3_rviz_mover")

        self.pub = self.create_publisher(JointState, "/joint_states", 10)

        self.t0 = time.time()

        # 20 Hz。速すぎない更新周期
        self.timer = self.create_timer(0.05, self.publish_joint_state)

        self.names = [
            "fr3_joint1",
            "fr3_joint2",
            "fr3_joint3",
            "fr3_joint4",
            "fr3_joint5",
            "fr3_joint6",
            "fr3_joint7",
        ]

    def publish_joint_state(self):
        t = time.time() - self.t0

        # ゆっくり小さく動かす
        q = [
            0.15 * math.sin(0.3 * t),   # joint1
            -0.7,                       # joint2
            0.10 * math.sin(0.3 * t),   # joint3
            -2.2,                       # joint4
            0.10 * math.sin(0.3 * t),   # joint5
            1.6,                        # joint6
            0.4,                        # joint7
        ]

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.names
        msg.position = q

        self.pub.publish(msg)


def main():
    rclpy.init()
    node = FR3RvizMover()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
