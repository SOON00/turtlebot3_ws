from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_nav',
            executable='planner',
            output='screen'
        ),
        Node(
            package='my_nav',
            executable='costmap',
            output='screen'
        ),
        # RViz
        #Node(
        #    package='rviz2',
        #    executable='rviz2',
        #    output='screen'
        #),
    ])

