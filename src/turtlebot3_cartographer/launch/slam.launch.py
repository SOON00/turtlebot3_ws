# ~/turtlebot3_ws/src/my_nav/launch/slam_combined.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    turtlebot3_cartographer_dir = get_package_share_directory('turtlebot3_cartographer')

    cartographer_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turtlebot3_cartographer_dir, 'launch', 'cartographer.launch.py')
        )
    )

    occupancy_grid_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turtlebot3_cartographer_dir, 'launch', 'occupancy_grid.launch.py')
        )
    )

    return LaunchDescription([
        cartographer_launch,
        occupancy_grid_launch
    ])

