#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, PoseStamped
import tf2_ros
from tf2_ros import TransformException
import numpy as np
import math
from scipy.optimize import minimize

class MPCTracker(Node):
    def __init__(self):
        super().__init__('mpc_tracker')

        # --- ROS topics ---
        self.create_subscription(Path, "/planned_path", self.path_callback, 10)
        self.create_subscription(PoseStamped, "/move_base_simple/goal", self.goal_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # --- TF buffer ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- MPC params ---
        self.dt = 0.1             # sampling time
        self.horizon = 10         # prediction horizon
        self.max_v = 0.25         # max linear velocity
        self.max_w = 1.2          # max angular velocity
        self.goal_reached_thresh = 0.1

        # --- internal states ---
        self.path = []
        self.goal_pose = None
        self.initial_rot_done = True  # 처음에는 회전할 필요 없음
        self.timer = self.create_timer(self.dt, self.control_loop)

    # --- Path callback ---
    def path_callback(self, msg):
        self.path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.get_logger().info(f"[MPC] New path received, length={len(self.path)}")
        # Path 자체로는 초기 회전 리셋하지 않음
        # 초기 회전은 goal이 들어왔을 때만

    # --- Goal callback ---
    def goal_callback(self, msg):
        self.goal_pose = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f"[MPC] New goal received: x={self.goal_pose[0]:.2f}, y={self.goal_pose[1]:.2f}")
        # Goal이 들어왔을 때 초기 회전 시작
        self.initial_rot_done = False

    # --- Get robot pose ---
    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform("map", "base_footprint", rclpy.time.Time())
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            q = trans.transform.rotation
            yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
            return x, y, yaw
        except TransformException:
            return None

    # --- Control loop ---
    def control_loop(self):
        if not self.path:
            return

        pose = self.get_robot_pose()
        if pose is None:
            return

        x, y, yaw = pose
        goal_x, goal_y = self.path[-1]
        dist_to_goal = math.hypot(goal_x - x, goal_y - y)

        # --- Goal reached check ---
        if dist_to_goal < self.goal_reached_thresh:
            self.get_logger().info("[MPC] Goal reached! Stopping robot.")
            self.cmd_pub.publish(Twist())
            self.path = []
            self.initial_rot_done = True
            return

        # --- Initial rotation (Goal-triggered) ---
        if not self.initial_rot_done and self.goal_pose is not None:
            target_yaw = math.atan2(goal_y - y, goal_x - x)
            yaw_err = (target_yaw - yaw + math.pi) % (2*math.pi) - math.pi
            cmd = Twist()
            if abs(yaw_err) > 0.05:
                cmd.angular.z = np.clip(yaw_err*1.5, -self.max_w, self.max_w)
            else:
                self.initial_rot_done = True
            self.cmd_pub.publish(cmd)
            return

        # --- MPC optimization ---
        u0 = np.zeros(self.horizon*2)  # [v0, w0, v1, w1, ...]
        bounds = []
        for _ in range(self.horizon):
            bounds.append((0, self.max_v))   # v bounds
            bounds.append((-self.max_w, self.max_w))  # w bounds

        def mpc_cost(u):
            x_pred, y_pred, yaw_pred = x, y, yaw
            cost = 0
            for t in range(self.horizon):
                v = u[2*t]
                w = u[2*t+1]
                x_pred += self.dt * v * math.cos(yaw_pred)
                y_pred += self.dt * v * math.sin(yaw_pred)
                yaw_pred += self.dt * w

                # 1. 경로 추종 항
                path_idx = min(t, len(self.path)-1)
                px, py = self.path[path_idx]
                cost += ((x_pred - px)**2 + (y_pred - py)**2) * 1.0

            return cost

        res = minimize(mpc_cost, u0, bounds=bounds, method='SLSQP')
        if res.success:
            v_cmd = res.x[0]
            w_cmd = res.x[1]
        else:
            v_cmd = 0.0
            w_cmd = 0.0
            self.get_logger().warn("[MPC] Optimization failed!")

        cmd = Twist()
        cmd.linear.x = v_cmd
        cmd.angular.z = w_cmd
        self.cmd_pub.publish(cmd)
        self.get_logger().info(f"[MPC] v={v_cmd:.2f}, w={w_cmd:.2f}, dist_goal={dist_to_goal:.2f}")


def main(args=None):
    rclpy.init(args=args)
    node = MPCTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
