import time
import numpy as np

import argparse
from python_utils.zmq_messenger import ZMQPublisher, ZMQSubscriber
from python_utils.global_configs import franka_left_real_zmq_addresses, franka_right_real_zmq_addresses

import sys

from pylibfranka import Robot, Torques
#add_watanabe
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


#___added

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="right", help="Robot name (left or right)")
    args = parser.parse_args()

    #add
    rclpy.init()
    node = Node("franka_cmd_bridge")

    obs_franka_state_pub = node.create_publisher(
    JointState,
    f"/franka/{args.name}/obs_franka_state",
    10
)
    cmd_franka_pos_pub = node.create_publisher(
    JointState,
    f"/factr_teleop/{args.name}/cmd_franka_pos",
    10
    )
    #wataanabe

    # Compliance parameters
    joint_stiffness = [90, 90, 90,30, 30, 30, 30]
    joint_damping = [2.5 * np.sqrt(k) for k in joint_stiffness]

    if args.name == "left":
        zmq_addresses = franka_left_real_zmq_addresses
        ip = "192.168.2.121"
    elif args.name == "right":
        zmq_addresses = franka_right_real_zmq_addresses
        ip = "192.168.2.122"
    else:
        raise ValueError(f"Invalid robot name '{args.name}'. Expected 'left' or 'right'.")

    # ZMQ通信
    franka_cmd_sub = ZMQSubscriber(zmq_addresses["joint_pos_cmd_pub"])
    franka_joint_state_pub = ZMQPublisher(zmq_addresses["joint_state_sub"])
    franka_force_pub = ZMQPublisher(zmq_addresses["force_ext_pub"])   # ★ ここで定義
    franka_torque_pub = ZMQPublisher(zmq_addresses["joint_torque_sub"])

    robot = None

    try:
        # Connect to robot
        robot = Robot(ip)

        robot.set_collision_behavior(
            [6000.0] * 7,
            [6000.0] * 7,
            [6000.0] * 6,
            [6000.0] * 6,
        )

        # Get initial state
        initial_state = robot.read_once()
        current_position = np.array(initial_state.q)

        # Start torque control
        active_control = robot.start_torque_control()

        # Create a model instance from robot
        model = robot.load_model()

        init_error = None
        prev_dq = np.zeros(7)

        while True:
            # Read robot state
            robot_state, _ = active_control.readOnce()

            # ★ 手先外力を publish（Franka API）
            F_ext = robot_state.K_F_ext_hat_K  # [Fx, Fy, Fz, Mx, My, Mz]
            franka_force_pub.send_message(np.array(F_ext))  # or F_ext.tolist()

            # joint state, torque も publish
            franka_joint_state_pub.send_message(np.array(robot_state.q))
            # print(robot_state.q)角度見るときはこれ
            franka_torque_pub.send_message(np.array(robot_state.tau_ext_hat_filtered))

            # Get state variables
            coriolis = np.array(model.coriolis(robot_state))
            q = np.array(robot_state.q)
            dq = np.array(robot_state.dq)

            # Get current target from teleop
            if franka_cmd_sub.message is None:
                continue
            q_goal = np.array(franka_cmd_sub.message)
            #add_watanabe
            msg = JointState()
            msg.position = q_goal.tolist()
            cmd_franka_pos_pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.0)
            #added__

            if init_error is None:
                init_error = q - q_goal

            # position error
            position_error = q - q_goal - init_error

            # joint-space impedance
            tau_task = np.zeros(7)
            for i in range(7):
                dq[i] = prev_dq[i] * 0.01 + dq[i] * 0.99
                prev_dq[i] = dq[i]
                tau_task[i] = -joint_stiffness[i] * position_error[i] - joint_damping[i] * dq[i]

            # Add coriolis compensation
            tau_d = tau_task + coriolis

            # Send torque command
            torque_command = Torques(tau_d.tolist())
            torque_command.motion_finished = False
            active_control.writeOnce(torque_command)


            state_msg = JointState()
            state_msg.position = q.tolist()
            obs_franka_state_pub.publish(state_msg)
    except Exception as e:
        print(f"\nError occurred: {e}")
        if robot is not None:
            robot.stop()
        return -1

    return 0


if __name__ == "__main__":
    sys.exit(main())