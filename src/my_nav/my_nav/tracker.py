#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist
import tf2_ros
from tf2_ros import TransformException
import math

class PurePursuitTracker(Node):
    def __init__(self):
        super().__init__('pure_pursuit_tracker')

        # --- Subscribers & Publishers ---
        self.create_subscription(Path, "/planned_path", self.path_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # --- TF Listener ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Path tracking variables ---
        self.path = []
        self.lookahead = 0.6  # 0.5~1.0m recommended for TB3

        # --- Rotation flags ---
        self.rotating_to_target = False  # 초기에는 회전하지 않음
        self.rotation_done_once = False  # path마다 1회만 회전

        # --- Control timer ---
        self.timer = self.create_timer(0.05, self.control_loop)

        # --- Robot constraints ---
        self.max_v = 0.25  # m/s
        self.max_w = 1.2   # rad/s

    def path_callback(self, msg):
        self.path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.get_logger().info(f"[Pure Pursuit] New path received, length={len(self.path)}")
        if self.path:
            # path가 들어오면 1회만 회전하도록 설정
            self.rotating_to_target = True
            self.rotation_done_once = False

    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                "map", "base_footprint", rclpy.time.Time())
            x = trans.transform.translation.x
            y = trans.transform.translation.y

            # quaternion → yaw
            q = trans.transform.rotation
            yaw = math.atan2(2*(q.w*q.z + q.x*q.y),
                             1 - 2*(q.y*q.y + q.z*q.z))
            return x, y, yaw
        except TransformException:
            return None

    def find_lookahead_target(self, x, y):
        """Lookahead 기반 목표점 찾기"""
        if not self.path:
            return None
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

        # --- Goal reached check ---
        goal_x, goal_y = self.path[-1]
        goal_dist = math.hypot(goal_x - x, goal_y - y)
        if goal_dist < 0.1:
            self.get_logger().info("[Pure Pursuit] Goal reached! Stopping robot.")
            self.cmd_pub.publish(Twist())
            self.path = []
            return

        # --- Initial rotation (1회만) ---
        if self.rotating_to_target and not self.rotation_done_once:
            tx, ty = self.path[-1]
            dx = tx - x
            dy = ty - y
            target_angle = math.atan2(dy, dx)
            angle_error = (target_angle - yaw + math.pi) % (2*math.pi) - math.pi

            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = max(min(angle_error * 2.0, self.max_w), -self.max_w)
            self.cmd_pub.publish(cmd)

            # 회전 완료 판단
            if abs(angle_error) < 0.05:  # 약 3도 이내
                self.rotating_to_target = False
                self.rotation_done_once = True  # 1회 회전 완료
            return

        # --- Lookahead target tracking ---
        target = self.find_lookahead_target(x, y)
        if target is None:
            return

        tx, ty = target
        dx = tx - x
        dy = ty - y
        target_angle = math.atan2(dy, dx)
        angle_error = (target_angle - yaw + math.pi) % (2*math.pi) - math.pi
        dist = math.hypot(dx, dy)

        # Pure Pursuit control law
        v = min(self.max_v, dist * 0.8)
        w = max(min(angle_error * 2.0, self.max_w), -self.max_w)

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f"[Pure Pursuit] Tracking → dist={dist:.2f}, angle_err={angle_error:.2f}, v={v:.2f}, w={w:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
