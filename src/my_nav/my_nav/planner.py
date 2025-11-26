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
from scipy.interpolate import CubicSpline

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

        # Parameters
        self.obstacle_cost_threshold = 50  # cost > this considered non-traversable
        self.max_goal_search_radius_cells = 200  # max search radius for safe goal (in cells)

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

    # ---------------------- Planner ----------------------
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

        h, w = self.costmap.shape

        # clamp goal if outside map bounds
        gx = min(max(goal_cell[0], 0), w-1)
        gy = min(max(goal_cell[1], 0), h-1)
        if (gx, gy) != goal_cell:
            self.get_logger().info(f"Goal outside map bounds. Clamped {goal_cell} -> {(gx, gy)}")
            goal_cell = (gx, gy)

        # If start cell invalid (obstacle) try to find nearby valid start
        if not (0 <= start_cell[0] < w and 0 <= start_cell[1] < h) or self.costmap[start_cell[1], start_cell[0]] > self.obstacle_cost_threshold:
            found = False
            for r in range(1, 10):
                for dx in range(-r, r+1):
                    for dy in range(-r, r+1):
                        nx, ny = start_cell[0]+dx, start_cell[1]+dy
                        if 0 <= nx < w and 0 <= ny < h and self.costmap[ny, nx] <= self.obstacle_cost_threshold:
                            start_cell = (nx, ny)
                            found = True
                            self.get_logger().info(f"Start cell shifted to nearby valid cell: {start_cell}")
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                self.get_logger().warn("No valid start cell found near robot!")
                return

        # If goal cell is an obstacle/high cost, search nearest safe cell
        if self.costmap[goal_cell[1], goal_cell[0]] > self.obstacle_cost_threshold:
            safe = self.find_nearest_safe_cell(goal_cell, self.obstacle_cost_threshold, self.max_goal_search_radius_cells)
            if safe is None:
                self.get_logger().warn("Goal in obstacle and no nearby safe cell found. Aborting plan.")
                return
            else:
                self.get_logger().info(f"Goal at {goal_cell} is occupied/high-cost. Using nearby safe goal {safe} instead.")
                goal_cell = safe

        # A* 경로
        path_cells = self.a_star(self.costmap, start_cell, goal_cell)
        if path_cells is None:
            self.get_logger().warn('No path found!')
            return

        # Cubic Spline smoothing
        smooth_path_points = self.smooth_path(path_cells)

        self.path = smooth_path_points
        self.get_logger().info(f'Path planned with {len(self.path)} points (smoothed).')
        self.publish_path(smooth_path_points)

    def find_nearest_safe_cell(self, goal_cell, cost_threshold, max_radius):
        """목표 셀이 위험한 경우, 가장 가까운 안전한 셀을 우선순위 큐(거리)로 탐색해서 반환.
           실패하면 None 반환.
        """
        h, w = self.costmap.shape
        gx0, gy0 = goal_cell

        visited = set()
        heap = []
        # push initial neighbors including goal itself
        def push_cell(nx, ny):
            if (nx, ny) in visited:
                return
            if not (0 <= nx < w and 0 <= ny < h):
                return
            visited.add((nx, ny))
            dist = math.hypot(nx - gx0, ny - gy0)
            heapq.heappush(heap, (dist, nx, ny))

        push_cell(gx0, gy0)

        while heap:
            dist, nx, ny = heapq.heappop(heap)
            # radius limit
            if dist > max_radius:
                break
            # check cost
            if self.costmap[ny, nx] <= cost_threshold:
                return (nx, ny)
            # expand neighbors (4-connected to limit branching; could use 8)
            push_cell(nx+1, ny)
            push_cell(nx-1, ny)
            push_cell(nx, ny+1)
            push_cell(nx, ny-1)

        return None

    # ---------------------- World/Grid Conversion ----------------------
    def world_to_grid(self, x, y):
        gx = int((x - self.origin[0]) / self.resolution)
        gy = int((y - self.origin[1]) / self.resolution)
        return (gx, gy)

    def grid_to_world(self, gx, gy):
        x = gx * self.resolution + self.origin[0]
        y = gy * self.resolution + self.origin[1]
        return (x, y)

    # ---------------------- A* Algorithm ----------------------
    def a_star(self, grid, start, goal):
        h, w = grid.shape
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        def heuristic(a, b):
            return math.hypot(b[0]-a[0], b[1]-a[1])

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
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if grid[ny, nx] > self.obstacle_cost_threshold:
                    continue
                move_cost = math.hypot(dx, dy)
                tentative_g = g_score[current] + move_cost
                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = tentative_g
                    f_score = tentative_g + heuristic((nx, ny), goal)
                    heapq.heappush(open_set, (f_score, (nx, ny)))
                    came_from[(nx, ny)] = current

        return None

    # ---------------------- Cubic Spline Smoothing ----------------------
    def smooth_path(self, path_cells):
        if not path_cells or len(path_cells) < 3:
            # 너무 짧으면 smoothing 안함
            return [self.grid_to_world(gx, gy) for gx, gy in path_cells]

        # grid -> world
        xs = []
        ys = []
        for gx, gy in path_cells:
            x, y = self.grid_to_world(gx, gy)
            xs.append(x)
            ys.append(y)

        xs = np.array(xs)
        ys = np.array(ys)

        # Arc-length parameterization
        ds = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        s = np.insert(np.cumsum(ds), 0, 0)

        # Cubic spline
        cs_x = CubicSpline(s, xs)
        cs_y = CubicSpline(s, ys)

        # Sample along the spline
        # ensure at least a few samples; sample spacing = resolution
        n_samples = max(2, int(s[-1]/self.resolution) + 1)
        s_new = np.linspace(0, s[-1], n_samples)
        x_new = cs_x(s_new)
        y_new = cs_y(s_new)

        return list(zip(x_new, y_new))

    # ---------------------- Path Publisher ----------------------
    def publish_path(self, path_points):
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        # Current robot position as first point
        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
            robot_x = trans.transform.translation.x
            robot_y = trans.transform.translation.y
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = path_msg.header.stamp
            pose.pose.position.x = robot_x
            pose.pose.position.y = robot_y
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        except TransformException:
            self.get_logger().warn('Cannot get robot position from TF for path start.')

        # Add smooth path points
        for x, y in path_points:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = path_msg.header.stamp
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

    # ---------------------- Control Loop ----------------------
    def control_loop(self):
        if self.path is None or len(self.path) == 0:
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
        next_point = self.path[0]
        goal_x, goal_y = next_point
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
