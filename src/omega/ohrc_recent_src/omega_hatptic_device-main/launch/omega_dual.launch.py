from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="omega_haptic_device",
            executable="omega_driver_left",
            name="omega_driver_left",
            output="screen",
        ),
        Node(
            package="omega_haptic_device",
            executable="omega_driver_right",
            name="omega_driver_right",
            output="screen",
        ),
    ])
