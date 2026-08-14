#!/usr/bin/env python3


# Copyright (c) 2025 Franka Robotics GmbH
# Use of this source code is governed by the Apache-2.0 license, see LICENSE

"""
Joint Impedance Target Example

This example shows a joint impedance controller that renders a spring damper system
to move the robot through a sequence of target joint configurations.
The controller will generate smooth trajectories between positions and run in a continuous loop.
"""

import argparse
import sys
import time

import numpy as np

from pylibfranka import Robot, Torques


class SimpleMotionGenerator:
    """Simple minimum jerk trajectory generator for smooth joint motion."""

    def __init__(self, start_position, end_position, duration=3.0):
        """Initialize the trajectory generator.

        Args:
            start_position: Starting joint positions (array of 7 values)
            end_position: Target joint positions (array of 7 values)
            duration: Duration of the trajectory in seconds
        """
        self.start_position = np.array(start_position)
        self.end_position = np.array(end_position)
        self.duration = duration
        self.start_time = None

    def start(self):
        """Start the trajectory."""
        self.start_time = time.time()

    def get_position(self):
        """Get the current target position along the trajectory."""
        if self.start_time is None:
            return self.start_position

        elapsed_time = time.time() - self.start_time
        s = self._minimum_jerk(min(elapsed_time / self.duration, 1.0))

        return self.start_position + s * (self.end_position - self.start_position)

    def is_finished(self):
        """Check if the trajectory is complete."""
        if self.start_time is None:
            return False

        elapsed_time = time.time() - self.start_time
        return elapsed_time >= self.duration

    def _minimum_jerk(self, t):
        """Minimum jerk trajectory profile (normalized [0,1])."""
        return 10 * (t**3) - 15 * (t**4) + 6 * (t**5)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument("--ip", type=str, default="192.168.2.122", help="Robot IP address")
    parser.add_argument("--name", type=str, default="right", help="Robot name (left or right)")
    args = parser.parse_args()

    if args.name == "left":
        ip = "192.168.2.121"
        target_joint_positions = [[0.1422, 0.03427, 0.4239, -2.134, -0.7413, 1.732, 1.544]]
        #[[
        # 0.4854744076728821,
        # -0.30415016412734985,
        # -0.10682374238967896,
        # -2.7122411727905273,
        # -0.19419655203819275,
        # 2.3623430728912354,
        # -0.31724807620048523,
        # ],]

        
        #[0.7651638388633728,0.22628208994865417,-0.41609451174736023,-1.9066907167434692,-0.7454635500907898,1.8872826099395752,1.5786808729171753]
        #[[.0, -0.7854, 0.0, -2.356, 0.0, 1.57, 0.8],]
        #[[0.23003636300563812, 0.506605863571167, 0.2930449843406677, -1.85, -1.2477757930755615, 1.1562508344650269, 0.12510937452316284],]
        #
        #
        #
        
    elif args.name == "right":
        ip = "192.168.2.122"
        target_joint_positions = [[-0.2683, 0.2595, -0.3499, -1.961, 0.7072, 1.673, -0.02815]]
        #[[
        # -0.5580952167510986,
        # -0.25113052129745483,
        # 0.08196945488452911,
        # -2.66996693611145,
        # 0.23759835958480835,
        # 2.3162217140197754,
        # 1.717193841934204,
        # ],]
        
        
        #[[.0, -0.7854, 0.0, -2.356, 0.0, 1.57, 0.8],]
        #[[-0.19939836859703064, 0.48379, -0.3693459928035736, -1.8857014179229736, 1.2677762508392334, 1.1561871767044067, -1.6482542753219604],]
        #
        #
        #
        
    else:
        raise ValueError(f"Invalid robot name '{args.name}'. Expected 'left' or 'right'.")

    # Define a sequence of target joint configurations
    # target_joint_positions = [
        # Home position (slightly bent arm)
        # [0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0],
        #start position basic
        #[.0, -0.7854, 0.0, -2.356, 0.0, 1.57, 0.0],
        #pre collision position
        # [0.21345369517803192, 0.46646371483802795, 0.3173743486404419, -1.929039716720581, -1.2516337633132935, 1.1626216173171997, 0.1598379909992218],
        # # Extended arm pointing forward
        # [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0],
        # # Arm pointing to the right
        # [0.5, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0],
        # # Arm pointing to the left
        # [-0.5, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0],
        # # Home position again
        # [0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0],
    #]

    # Compliance parameters
    joint_stiffness = [70.0, 120.0, 70.0, 120.0, 50.0, 50.0, 50.0]
    joint_damping = [2.0 * np.sqrt(k) for k in joint_stiffness]

    try:
        # Connect to robot
        robot = Robot(ip)

        # Get initial state
        initial_state = robot.read_once()
        current_position = np.array(initial_state.q)

        # Start torque control
        active_control = robot.start_torque_control()

        # Create a model instance from robot
        model = robot.load_model()

        # Main control variables
        wait_time = 0.5  # Time to wait at each position before moving to next

        # For loop over target positions
        for target_position in target_joint_positions:
            # Initialize trajectory for this target
            trajectory = SimpleMotionGenerator(
                current_position,
                target_position,
                duration=5.0,
            )
            trajectory.start()

            # Control until we reach target and wait time is complete
            target_reached = False
            wait_started = False
            wait_start_time = 0

            # Control loop for current trajectory
            while True:
                # Read robot state
                robot_state, _ = active_control.readOnce()

                # Get state variables
                coriolis = np.array(model.coriolis(robot_state))
                q = np.array(robot_state.q)
                dq = np.array(robot_state.dq)

                # Get current target from trajectory
                q_goal = trajectory.get_position()

                # Compute error to desired equilibrium joint configuration
                position_error = q - q_goal

                # Compute joint-space impedance control
                tau_task = np.zeros(7)
                for i in range(7):
                    tau_task[i] = -joint_stiffness[i] * position_error[i] - joint_damping[i] * dq[i]

                # Add coriolis compensation
                tau_d = tau_task + coriolis
                # print("tau_d:", tau_d)

                # Convert to array for Torques command
                torque_command = Torques(tau_d.tolist())
                torque_command.motion_finished = False
                active_control.writeOnce(torque_command)

                # Check if robot actually reached the target渡辺修正
                final_target = np.array(target_position)

                pos_error = q - final_target
                pos_error_norm = np.linalg.norm(pos_error)
                vel_norm = np.linalg.norm(dq)

                pos_tol = 0.1   # [rad] 関節角誤差の許容値。まずは0.03くらい
                vel_tol = 0.02   # [rad/s] 速度の許容値

                if pos_error_norm < pos_tol and vel_norm < vel_tol and not target_reached:
                    target_reached = True
                    wait_started = True
                    wait_start_time = time.time()
                    print(f"Target reached: pos_error={pos_error_norm:.4f}, vel={vel_norm:.4f}")

                # Check if we've waited long enough after reaching target
                if wait_started and (time.time() - wait_start_time >= wait_time):
                    current_position = final_target
                    break

                # # Check if trajectory is finished
                # if trajectory.is_finished() and not target_reached:
                #     target_reached = True
                #     wait_started = True
                #     wait_start_time = time.time()

                # # Check if we've waited long enough
                # if wait_started and (time.time() - wait_start_time >= wait_time):
                #     # Update current position for next trajectory
                #     current_position = q_goal
                #     break

    except Exception as e:
        print(f"\nError occurred: {e}")
        if robot is not None:
            robot.stop()
        return -1

    return 0


if __name__ == "__main__":
    sys.exit(main())
