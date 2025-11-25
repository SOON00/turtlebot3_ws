#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist
import tf2_ros
from tf2_ros import TransformException
import numpy as np
import math

class MPCTracker(Node):
    def __init__(self):
        super().__init__('MPC_tracker')

        self.create_subscription(Path, "/planned_path", self.path_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.path = []
        self.lookahead = 0.6  # 0.5~1.0m recommended for TB3
        self.timer = self.create_timer(0.05, self.control_loop)

        # MPC parameters
        self.dt = 0.05
        self.horizon = 10
        self.max_v = 0.25
        self.max_w = 1.2

    def path_callback(self, msg):
        self.path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.get_logger().info(f"[MPC] New path received, length={len(self.path)}")

    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                "map", "base_footprint", rclpy.time.Time())
            x = trans.transform.translation.x
            y = trans.transform.translation.y

            # orientation to yaw
            q = trans.transform.rotation
            yaw = math.atan2(2*(q.w*q.z + q.x*q.y),
                             1 - 2*(q.y*q.y + q.z*q.z))

            return x, y, yaw
        except TransformException:
            return None

    def find_target(self, x, y):
        if not self.path:
            return None

        # lookahead point search
        for px, py in self.path:
            d = math.hypot(px - x, py - y)
            if d > self.lookahead:
                return px, py

        return self.path[-1]

    def control_loop(self):
        if not self.path:
            return

        pose = self.get_robot_pose()
        if pose is None:
            return

        x, y, yaw = pose

        # --- Goal reach check ---
        goal_x, goal_y = self.path[-1]
        goal_dist = math.hypot(goal_x - x, goal_y - y)

        if goal_dist < 0.1:
            self.get_logger().info(f"[MPC] Goal reached! Stopping robot.")

            cmd = Twist()
            self.cmd_pub.publish(cmd)

            # path clear to stop tracking
            self.path = []
            return

        target = self.find_target(x, y)
        if target is None:
            return

        tx, ty = target

        dx = tx - x
        dy = ty - y

        # Convert to robot coordinate frame
        target_angle = math.atan2(dy, dx)
        angle_error = (target_angle - yaw + math.pi) % (2*math.pi) - math.pi
        dist = math.hypot(dx, dy)

        # --- Simple MPC-style control law ---
        # Linear velocity tries to reduce position error
        v = min(self.max_v, dist * 0.8)

        # Angular velocity tries to reduce angle error
        w = max(min(angle_error * 2.0, self.max_w), -self.max_w)

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f"[MPC] Tracking → dist={dist:.2f}, angle_err={angle_error:.2f}, v={v:.2f}, w={w:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = MPCTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
