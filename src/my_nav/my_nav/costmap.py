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

        # 사용자 조절 가능한 파라미터
        self.width_m = 20.0        # costmap 가로 (m)
        self.height_m = 20.0       # costmap 세로 (m)
        self.inflation_radius = 0.25  # m 단위

    def map_callback(self, msg):
        # 원본 그리드
        grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))

        # --- Inflation ---
        cell_radius = int(self.inflation_radius / msg.info.resolution)
        inflated = maximum_filter(grid, size=2*cell_radius+1, mode='constant')

        # --- 전체 costmap 크기 맞추기 ---
        width_cells = int(self.width_m / msg.info.resolution)
        height_cells = int(self.height_m / msg.info.resolution)

        padded_grid = np.zeros((height_cells, width_cells), dtype=np.int8)

        # 원본 맵을 중앙에 위치시키기
        start_x = (width_cells - grid.shape[1]) // 2
        start_y = (height_cells - grid.shape[0]) // 2
        padded_grid[start_y:start_y+grid.shape[0], start_x:start_x+grid.shape[1]] = inflated

        # --- OccupancyGrid 메시지 생성 ---
        costmap_msg = OccupancyGrid()
        costmap_msg.header = msg.header
        costmap_msg.info.resolution = msg.info.resolution
        costmap_msg.info.width = width_cells
        costmap_msg.info.height = height_cells

        # 원점을 중앙으로 설정
        costmap_msg.info.origin.position.x = -self.width_m / 2.0 +2
        costmap_msg.info.origin.position.y = -self.height_m / 2.0 +0.5
        costmap_msg.info.origin.position.z = 0.0
        costmap_msg.info.origin.orientation.w = 1.0  # 회전 없음

        costmap_msg.data = padded_grid.flatten().tolist()

        self.publisher.publish(costmap_msg)
        self.get_logger().info(f'Costmap published: {self.width_m}x{self.height_m} m, inflation={self.inflation_radius} m.')

def main(args=None):
    rclpy.init(args=args)
    node = CostmapInflator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
