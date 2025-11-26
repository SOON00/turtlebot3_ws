#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
from scipy.ndimage import maximum_filter

class CostmapInflator(Node):
    def __init__(self):
        super().__init__('costmap_inflator')

        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.publisher = self.create_publisher(OccupancyGrid, '/costmap', 10)

        # Inflation 반경 (미터)
        self.inflation_radius = 0.25

    def map_callback(self, msg):
        # 원본 그리드 그대로 사용
        grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))

        # --- Inflation만 적용 ---
        cell_radius = int(self.inflation_radius / msg.info.resolution)
        inflated = maximum_filter(grid, size=2*cell_radius+1, mode='constant')

        # --- OccupancyGrid 메시지 생성 ---
        costmap_msg = OccupancyGrid()
        costmap_msg.header = msg.header
        costmap_msg.info = msg.info
        costmap_msg.data = inflated.flatten().tolist()

        self.publisher.publish(costmap_msg)
        self.get_logger().info(f'Costmap published with inflation={self.inflation_radius} m.')

def main(args=None):
    rclpy.init(args=args)
    node = CostmapInflator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
