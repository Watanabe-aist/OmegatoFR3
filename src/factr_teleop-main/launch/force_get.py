from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    force_get_right_node = Node(
        package='factr_teleop',
        executable='force_get_right',
        name='force_get_right',
        output='screen'
    )

    force_get_left_node = Node(
        package='factr_teleop',
        executable='force_get_left',
        name='force_get_left',
        output='screen'
    )

    return LaunchDescription([
        force_get_right_node,
        force_get_left_node,
    ])