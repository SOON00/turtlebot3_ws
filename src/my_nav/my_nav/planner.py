#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Twist
import numpy as np
import math
import heapq
import tf2_ros
from tf2_ros import TransformException

class AStarNav(Node):
    def __init__(self):
        super().__init__('astar')

        # Subscribers
        self.create_subscription(OccupancyGrid, '/costmap', self.costmap_callback, 10)
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.goal_callback, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)

        # TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # State
        self.costmap = None
        self.resolution = 0.05
        self.origin = (0.0, 0.0)
        self.goal = None
        self.path = []

        # Control loop timer
        self.timer = self.create_timer(0.2, self.control_loop)

    def costmap_callback(self, msg):
        grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.costmap = grid
        self.origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.resolution = msg.info.resolution
        self.get_logger().info('Costmap received.')
        
        if self.goal is not None:
            self.get_logger().info("Replanning due to updated costmap.")
            self.plan_path()


    def goal_callback(self, msg):
        self.goal = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f'Goal received: x={self.goal[0]:.2f}, y={self.goal[1]:.2f}')
        if self.costmap is not None:
            self.plan_path()

    def plan_path(self):
        if self.costmap is None or self.goal is None:
            return

        # TF로 현재 로봇 위치 가져오기
        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
            start = (trans.transform.translation.x, trans.transform.translation.y)
        except TransformException:
            self.get_logger().warn('Cannot get robot position from TF.')
            return

        start_cell = self.world_to_grid(*start)
        goal_cell = self.world_to_grid(*self.goal)

        path_cells = self.a_star(self.costmap, start_cell, goal_cell)
        if path_cells is None:
            self.get_logger().warn('No path found!')
            return

        self.path = path_cells
        self.get_logger().info(f'Path planned with {len(self.path)} cells.')
        self.publish_path(path_cells)

    # World <-> Grid conversion
    def world_to_grid(self, x, y):
        gx = int((x - self.origin[0]) / self.resolution)
        gy = int((y - self.origin[1]) / self.resolution)
        return (gx, gy)

    def grid_to_world(self, gx, gy):
        x = gx * self.resolution + self.origin[0]
        y = gy * self.resolution + self.origin[1]
        return (x, y)

    # Simple 8-connected A*
    def a_star(self, grid, start, goal):
        h, w = grid.shape
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        def heuristic(a, b):
            return math.hypot(b[0]-a[0], b[1]-a[1])

        # 8-connected movement
        neighbors = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            for dx, dy in neighbors:
                nx, ny = current[0] + dx, current[1] + dy

                # bounds check
                if not (0 <= nx < w and 0 <= ny < h):
                    continue

                # obstacle check (inflation 포함)
                if grid[ny, nx] > 50:
                    continue

                # movement cost
                move_cost = math.hypot(dx, dy)  # 1 or √2

                tentative_g = g_score[current] + move_cost

                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = tentative_g
                    f_score = tentative_g + heuristic((nx, ny), goal)
                    heapq.heappush(open_set, (f_score, (nx, ny)))
                    came_from[(nx, ny)] = current

        return None


    # Publish path for RViz visualization
    def publish_path(self, path_cells):
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        for gx, gy in path_cells:
            x, y = self.grid_to_world(gx, gy)
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = path_msg.header.stamp
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)

    # Control loop
    def control_loop(self):
        if self.path is None or len(self.path) == 0:
            twist = Twist()
            self.get_logger().info('[No path]')
            return

        # Get robot position
        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
            robot_x = trans.transform.translation.x
            robot_y = trans.transform.translation.y
        except TransformException:
            self.get_logger().warn('Cannot get robot position from TF.')
            return

        # Next waypoint
        next_cell = self.path[0]
        goal_x, goal_y = self.grid_to_world(*next_cell)
        dx = goal_x - robot_x
        dy = goal_y - robot_y
        distance = math.hypot(dx, dy)
        angle_to_goal = math.atan2(dy, dx)

        twist = Twist()
        if distance > 0.05:
            twist.linear.x = min(0.2, distance)
            twist.angular.z = max(min(2.0*angle_to_goal, 1.0), -1.0)
            self.get_logger().info(f'[Control] dx={dx:.2f}, dy={dy:.2f}, dist={distance:.2f}, angle={angle_to_goal:.2f}')
        else:
            self.path.pop(0)
            self.get_logger().info('Waypoint reached, moving to next.')

        #self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = AStarNav()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

