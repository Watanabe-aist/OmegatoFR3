import time
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from bc.utils import create_joint_state_msg
from python_utils.zmq_messenger import ZMQPublisher, ZMQSubscriber
from python_utils.global_configs import franka_right_real_zmq_addresses


class OmegaFrankaBridge(Node):
    def __init__(self):
        super().__init__("omega_franka_bridge")

        right_zmq_addresses = franka_right_real_zmq_addresses

        # Omega IK が出した q_goal を受ける
        self.q_goal_sub = self.create_subscription(
            JointState,
            "/omega_fr3/right/q_goal",
            self.q_goal_callback,
            10,
        )

        # Franka実機へ関節角目標を送るZMQ publisher
        self.franka_cmd_pub = ZMQPublisher(
            right_zmq_addresses["joint_pos_cmd_pub"]
        )

        # 実機状態を受けるZMQ subscriber
        self.franka_pos_sub = ZMQSubscriber(
            right_zmq_addresses["joint_state_sub"]
        )

        self.franka_torque_sub = ZMQSubscriber(
            right_zmq_addresses["joint_torque_sub"]
        )

        # ROS2へ観測値を流すpublisher
        self.franka_pos_pub = self.create_publisher(
            JointState,
            "/franka/right/obs_franka_state",
            10,
        )

        self.franka_torque_pub = self.create_publisher(
            JointState,
            "/franka/right/obs_franka_torque",
            10,
        )

        # 安全用：急激なq_goal変化を制限
        self.max_joint_vel = 0.5  # rad/s 最初は小さめ
        self.dt = 1.0 / 300.0
        self.prev_cmd = None

        self.timer = self.create_timer(self.dt, self.timer_callback)

        self.get_logger().info("OmegaFrankaBridge started")
        self.get_logger().info("subscribe: /omega_fr3/right/q_goal")
        self.get_logger().info("send to Franka right via ZMQ")

    def _rate_limit(self, cmd):
        if self.prev_cmd is None:
            self.prev_cmd = cmd.copy()
            return cmd

        max_step = self.max_joint_vel * self.dt
        step = np.clip(cmd - self.prev_cmd, -max_step, max_step)
        limited_cmd = self.prev_cmd + step

        self.prev_cmd = limited_cmd.copy()
        return limited_cmd

    def q_goal_callback(self, msg):
        if len(msg.position) < 7:
            self.get_logger().warn("Received q_goal with less than 7 joints")
            return

        q_goal = np.array(msg.position[:7], dtype=float)

        q_goal = self._rate_limit(q_goal)

        self.franka_cmd_pub.send_message(q_goal)

    def timer_callback(self):
        if self.franka_pos_sub.message is not None:
            q = np.array(self.franka_pos_sub.message[0:7])
            self.franka_pos_pub.publish(create_joint_state_msg(q))

        if self.franka_torque_sub.message is not None:
            tau = np.array(self.franka_torque_sub.message)
            self.franka_torque_pub.publish(create_joint_state_msg(tau))


def main(args=None):
    rclpy.init(args=args)
    node = OmegaFrankaBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
