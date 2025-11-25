#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
from scipy.ndimage import maximum_filter

class CostmapGenerator(Node):
    def __init__(self):
        super().__init__('costmap_generator')

        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.publisher = self.create_publisher(OccupancyGrid, '/costmap', 10)

        # Inflation parameters
        self.inflation_radius = 0.25  # meters

    def map_callback(self, msg):
        grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        
        costmap = np.zeros_like(grid, dtype=np.int8)
        costmap[grid == -1] = 50
        costmap[grid == 100] = 100
        costmap[grid == 0] = 0

        # --- Inflation ---
        # inflation_radius를 grid 셀 단위로 변환
        cell_radius = int(self.inflation_radius / msg.info.resolution)
        # maximum_filter로 주변 영역을 확장
        inflated = maximum_filter(costmap, size=2*cell_radius+1, mode='constant')
        costmap = np.maximum(costmap, inflated)

        # OccupancyGrid로 다시 퍼블리시
        costmap_msg = OccupancyGrid()
        costmap_msg.header = msg.header
        costmap_msg.info = msg.info
        costmap_msg.data = costmap.flatten().tolist()

        self.publisher.publish(costmap_msg)
        self.get_logger().info(f'Costmap published with inflation radius {self.inflation_radius} m.')

def main(args=None):
    rclpy.init(args=args)
    node = CostmapGenerator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

