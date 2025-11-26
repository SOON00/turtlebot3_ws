#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
from scipy.ndimage import maximum_filter

class CostmapInflator(Node):
    def __init__(self):
        super().__init__('costmap_inflator')

        # 구독
        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        # 퍼블리셔
        self.publisher = self.create_publisher(OccupancyGrid, '/costmap', 10)

        # Inflation 반경 (미터)
        self.inflation_radius = 0.75

        # 맵 padding (셀 단위)
        self.pad_cells = 500  # 약 25m 정도 확장, 필요에 맞게 조정

    def map_callback(self, msg):
        # 1️⃣ 원본 그리드 배열 변환
        grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))

        # 2️⃣ Inflation 적용
        cell_radius = int(self.inflation_radius / msg.info.resolution)
        inflated = maximum_filter(grid, size=2*cell_radius+1, mode='constant')

        # 3️⃣ 맵 padding
        padded = np.pad(
            inflated,
            self.pad_cells,
            mode='constant',
            constant_values=0
        )

        # 4️⃣ OccupancyGrid 메시지 생성
        costmap_msg = OccupancyGrid()
        costmap_msg.header = msg.header

        # info 수정 (크기, origin)
        costmap_msg.info.width = padded.shape[1]
        costmap_msg.info.height = padded.shape[0]
        costmap_msg.info.resolution = msg.info.resolution

        # origin.x, origin.y를 padding만큼 이동
        costmap_msg.info.origin.position.x = msg.info.origin.position.x - self.pad_cells * msg.info.resolution
        costmap_msg.info.origin.position.y = msg.info.origin.position.y - self.pad_cells * msg.info.resolution
        costmap_msg.info.origin.orientation = msg.info.origin.orientation

        # 데이터 flatten
        costmap_msg.data = padded.flatten().tolist()

        # Publish
        self.publisher.publish(costmap_msg)
        self.get_logger().info(f'Costmap published with inflation={self.inflation_radius} m, padding={self.pad_cells} cells.')

def main(args=None):
    rclpy.init(args=args)
    node = CostmapInflator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
