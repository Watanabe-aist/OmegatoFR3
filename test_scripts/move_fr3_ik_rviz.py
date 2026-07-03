import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from pynput import keyboard

import pinocchio as pin

from ohrc_msgs.msg import State


# ===========================
# 冗長自由度設定
# 0: 固定なし
# 1〜7: その関節を固定
# ===========================
REDUNDANT_JOINT = 0


# ===========================
# Omega入力設定
# ===========================
OMEGA_TOPIC = "/right/state"
OMEGA_POS_SCALE = 2.0
OMEGA_ROT_SCALE = 1.0


# ===========================
# Omega位置座標 → FR3位置座標
# Omega xyz -> FR3 xyz
# ===========================
OMEGA_TO_FR3_POS = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])


OMEGA_ROT_SIGN = np.array([
    1.0,   # roll方向
   -1.0,   # pitch方向
   -1.0,   # yaw方向 
])


class FR3IKOmegaRvizMover(Node):
    def __init__(self):
        super().__init__("fr3_ik_omega_rviz_mover")

        if REDUNDANT_JOINT < 0 or REDUNDANT_JOINT > 7:
            raise ValueError("REDUNDANT_JOINT must be 0〜7")

        self.pub = self.create_publisher(JointState, "/joint_states", 10)

        self.joint_names = [
            "fr3_joint1",
            "fr3_joint2",
            "fr3_joint3",
            "fr3_joint4",
            "fr3_joint5",
            "fr3_joint6",
            "fr3_joint7",
            "fr3_finger_joint1",
            "fr3_finger_joint2",
        ]

        self.urdf_path = "/home/watanaberyuto/franka_ros2_ws/src/fr3_urdf/fr3.urdf"

        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()

        self.frame_name = "fr3_hand_tcp"
        if not self.model.existFrame(self.frame_name):
            self.frame_name = "fr3_link8"

        self.frame_id = self.model.getFrameId(self.frame_name)

        self.q = np.array([
            0.0,
            -0.7,
            0.0,
            -2.2,
            0.0,
            1.6,
            0.4,
            0.02,
            0.02,
        ])

        if REDUNDANT_JOINT == 0:
            self.control_joint_idx = None
            self.control_joint_value = None
            self.active_idx = list(range(7))
        else:
            self.control_joint_idx = REDUNDANT_JOINT - 1
            self.control_joint_value = self.q[self.control_joint_idx]
            self.active_idx = [
                i for i in range(7)
                if i != self.control_joint_idx
            ]

        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

        initial_se3 = self.data.oMf[self.frame_id]
        self.fr3_initial_pos = initial_se3.translation.copy()
        self.fr3_initial_rot = initial_se3.rotation.copy()

        self.target_pos = self.fr3_initial_pos.copy()
        self.target_rot = self.fr3_initial_rot.copy()

        self.omega_initialized = False
        self.omega_initial_pos = np.zeros(3)
        self.omega_pos = np.zeros(3)

        self.omega_initial_rot = np.eye(3)
        self.omega_rot = np.eye(3)

        self.create_subscription(
            State,
            OMEGA_TOPIC,
            self.omega_callback,
            10,
        )

        self.timer = self.create_timer(0.05, self.loop)

        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

        print("Omega control:")
        print(f"Omega topic: {OMEGA_TOPIC}")
        print("Omega position -> FR3 target position")
        print("Omega orientation -> FR3 target orientation")
        print("axis mapping: Omega xyz -> FR3 xyz")
        print(f"OMEGA_POS_SCALE = {OMEGA_POS_SCALE}")
        print(f"OMEGA_ROT_SCALE = {OMEGA_ROT_SCALE}")
        print(f"OMEGA_ROT_SIGN = {OMEGA_ROT_SIGN}")

        if self.control_joint_idx is None:
            print("Redundant joint: none, all 7 joints active")
        else:
            print("Redundant joint:")
            print("  z/x : redundant joint -/+")
            print(f"  fixed joint = fr3_joint{REDUNDANT_JOINT}")
            print("  initial value =", self.control_joint_value)

        print("initial FR3 pos =", self.fr3_initial_pos)
        print("waiting for Omega...")

    def quat_to_rot(self, qx, qy, qz, qw):
        norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)

        if norm < 1e-12:
            return np.eye(3)

        qx /= norm
        qy /= norm
        qz /= norm
        qw /= norm

        return np.array([
            [
                1.0 - 2.0 * (qy*qy + qz*qz),
                2.0 * (qx*qy - qz*qw),
                2.0 * (qx*qz + qy*qw),
            ],
            [
                2.0 * (qx*qy + qz*qw),
                1.0 - 2.0 * (qx*qx + qz*qz),
                2.0 * (qy*qz - qx*qw),
            ],
            [
                2.0 * (qx*qz - qy*qw),
                2.0 * (qy*qz + qx*qw),
                1.0 - 2.0 * (qx*qx + qy*qy),
            ],
        ])

    def omega_callback(self, msg):
        p = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])

        R = self.quat_to_rot(
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )

        if not self.omega_initialized:
            self.omega_initial_pos = p.copy()
            self.omega_initial_rot = R.copy()
            self.omega_initialized = True
            self.get_logger().info(f"Omega initialized: {self.omega_initial_pos}")

        self.omega_pos = p
        self.omega_rot = R

    def on_press(self, key):
        joint_step = 0.01

        try:
            if self.control_joint_idx is None:
                return

            if key.char == "z":
                self.control_joint_value -= joint_step
            elif key.char == "x":
                self.control_joint_value += joint_step
            else:
                return

            self.control_joint_value = np.clip(
                self.control_joint_value,
                self.model.lowerPositionLimit[self.control_joint_idx],
                self.model.upperPositionLimit[self.control_joint_idx],
            )

            print(
                f"fr3_joint{REDUNDANT_JOINT} =",
                self.control_joint_value,
            )

        except AttributeError:
            pass

    def update_target_from_omega(self):
        if not self.omega_initialized:
            return False

        delta_omega = self.omega_pos - self.omega_initial_pos
        delta_fr3 = OMEGA_TO_FR3_POS @ delta_omega

        self.target_pos = (
            self.fr3_initial_pos
            + OMEGA_POS_SCALE * delta_fr3
        )

        R_delta_omega = self.omega_initial_rot.T @ self.omega_rot

        rot_vec = pin.log3(R_delta_omega)

        # 回転量のスケールと符号反転
        rot_vec = OMEGA_ROT_SCALE * (OMEGA_ROT_SIGN * rot_vec)

        R_delta_mapped = pin.exp3(rot_vec)

        self.target_rot = self.fr3_initial_rot @ R_delta_mapped

        return True

    def damped_pinv(self, J, damping=0.05):
        return J.T @ np.linalg.inv(
            J @ J.T + damping * damping * np.eye(J.shape[0])
        )

    def solve_ik(self):
        max_iter = 80
        alpha = 0.2
        eps = 1e-4
        max_dq = 0.05

        for _ in range(max_iter):
            if self.control_joint_idx is not None:
                self.q[self.control_joint_idx] = self.control_joint_value

            self.q[7] = 0.02
            self.q[8] = 0.02

            pin.forwardKinematics(self.model, self.data, self.q)
            pin.updateFramePlacements(self.model, self.data)

            current_se3 = self.data.oMf[self.frame_id]

            pos_err = self.target_pos - current_se3.translation
            rot_err = pin.log3(self.target_rot @ current_se3.rotation.T)

            err = np.concatenate([pos_err, rot_err])

            if np.linalg.norm(err) < eps:
                break

            J6 = pin.computeFrameJacobian(
                self.model,
                self.data,
                self.q,
                self.frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )

            J_active = J6[:, self.active_idx]

            dq_active = self.damped_pinv(J_active) @ err

            dq_norm = np.linalg.norm(dq_active)
            if dq_norm > max_dq:
                dq_active = dq_active / dq_norm * max_dq

            dq = np.zeros(self.model.nv)

            for j, idx in enumerate(self.active_idx):
                dq[idx] = dq_active[j]

            self.q = self.q + alpha * dq

            if self.control_joint_idx is not None:
                self.q[self.control_joint_idx] = self.control_joint_value

            self.q[7] = 0.02
            self.q[8] = 0.02

            self.q = np.maximum(self.model.lowerPositionLimit, self.q)
            self.q = np.minimum(self.model.upperPositionLimit, self.q)

        return self.q

    def loop(self):
        if not self.update_target_from_omega():
            return

        q = self.solve_ik()

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = q.tolist()

        self.pub.publish(msg)

    def destroy_node(self):
        self.listener.stop()
        super().destroy_node()


def main():
    rclpy.init()
    node = FR3IKOmegaRvizMover()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
