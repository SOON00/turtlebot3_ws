#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Twist
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
        self.create_subscription(OccupancyGrid, "/costmap", self.costmap_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # --- TF buffer ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- MPC params ---
        self.dt = 0.1             # sampling time
        self.horizon = 20         # prediction horizon
        self.max_v = 0.5         # max linear velocity
        self.max_w = 2.0          # max angular velocity
        self.goal_reached_thresh = 0.2

        # --- internal states ---
        self.path = []
        self.initial_rot_done = True   # 기본값 True

        # costmap
        self.costmap = None
        self.resolution = 0.05
        self.origin = (0.0, 0.0)

        # For warm-start and previous command
        self.prev_u = np.zeros(self.horizon*2)   # warm start buffer
        self.last_v = 0.0
        self.last_w = 0.0

        # smoothing of v_ref
        self.v_ref_prev = 0.0
        self.v_ref_alpha = 0.6  # 0..1, higher = smoother (less reactive)

        # output low-pass filter
        self.cmd_alpha = 0.5  # 0..1, higher = smoother

        self.timer = self.create_timer(self.dt, self.control_loop)

    # ---------------- callbacks ----------------
    def path_callback(self, msg):
        self.path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.get_logger().info(f"[MPC] New path received, length={len(self.path)}")
        # do not reset initial_rot_done here

    def goal_callback(self, msg):
        self.initial_rot_done = False
        self.get_logger().info("[MPC] Goal received. Initial rotation enabled.")

    def costmap_callback(self, msg):
        grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.costmap = grid
        self.resolution = msg.info.resolution
        self.origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.get_logger().info("[MPC] Costmap updated.")

    # ---------------- helpers ----------------
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

    def world_to_grid(self, x, y):
        gx = int((x - self.origin[0]) / self.resolution)
        gy = int((y - self.origin[1]) / self.resolution)
        return gx, gy

    def compute_path_curvature(self, path, idx):
        if idx <= 0 or idx >= len(path)-1:
            return 0.0
        x1, y1 = path[idx-1]
        x2, y2 = path[idx]
        x3, y3 = path[idx+1]
        area = abs(0.5 * ((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)))
        a = math.hypot(x2-x1, y2-y1)
        b = math.hypot(x3-x2, y3-y2)
        c = math.hypot(x3-x1, y3-y1)
        if area == 0 or a==0 or b==0 or c==0:
            return 0.0
        R = (a*b*c)/(4*area)
        if R == 0:
            return 0.0
        return 1.0 / R

    # ---------------- control ----------------
    def control_loop(self):
        if not self.path:
            return

        pose = self.get_robot_pose()
        if pose is None:
            return

        x, y, yaw = pose
        goal_x, goal_y = self.path[-1]
        dist_to_goal = math.hypot(goal_x - x, goal_y - y)

        # goal reached
        if dist_to_goal < self.goal_reached_thresh:
            self.get_logger().info("[MPC] Goal reached! Stopping robot.")
            self.cmd_pub.publish(Twist())
            self.path = []
            self.initial_rot_done = True
            self.prev_u[:] = 0.0
            self.last_v = 0.0
            self.last_w = 0.0
            return

        # initial rotation
        if not self.initial_rot_done:
            target_yaw = math.atan2(goal_y - y, goal_x - x)
            yaw_err = (target_yaw - yaw + math.pi) % (2*math.pi) - math.pi
            cmd = Twist()
            if abs(yaw_err) > 0.05:
                cmd.angular.z = np.clip(yaw_err*1.5, -self.max_w, self.max_w)
                self.cmd_pub.publish(cmd)
                # keep last commands small during initial rotation
                self.last_v = 0.0
                self.last_w = cmd.angular.z
            else:
                self.initial_rot_done = True
            return

        # --- MPC optimization setup ---
        # warm start u0: shift previous solution, fill forward with last_v
        u0 = np.zeros(self.horizon*2)
        if np.any(self.prev_u):
            # shift previous control sequence
            u0[:-2] = self.prev_u[2:]
            u0[-2:] = self.prev_u[-2:]
        else:
            # if no previous, initialize forward with a reasonable forward speed
            for t in range(self.horizon):
                u0[2*t] = min(self.max_v, 0.6 * self.max_v)
                u0[2*t+1] = 0.0

        bounds = []
        for _ in range(self.horizon):
            bounds.append((0, self.max_v))
            bounds.append((-self.max_w, self.max_w))

        def mpc_cost(u):
            x_pred, y_pred, yaw_pred = x, y, yaw
            # start v_prev,w_prev from last applied to penalize large accel
            v_prev = self.last_v
            w_prev = self.last_w
            cost = 0.0

            for t in range(self.horizon):
                v = u[2*t]
                w = u[2*t+1]

                # forward simulate
                x_pred += self.dt * v * math.cos(yaw_pred)
                y_pred += self.dt * v * math.sin(yaw_pred)
                yaw_pred += self.dt * w

                # path tracking (smaller weight to avoid over-react)
                path_idx = min(t, len(self.path)-1)
                px, py = self.path[path_idx]
                cost += ((x_pred - px)**2 + (y_pred - py)**2) * 0.2

                # acceleration penalty (stronger)
                cost += 0.05 * ((v - v_prev)**2)  # penalize linear accel
                cost += 0.01 * ((w - w_prev)**2)  # penalize angular accel

                # curvature-based v_ref with smoothing and cap
                idx = min(path_idx, max(0, len(self.path)-2))
                kappa = self.compute_path_curvature(self.path, idx)
                if kappa < 0.01:
                    v_ref_new = self.max_v
                else:
                    # k param controls aggressiveness: tuned small for stable forward
                    k = 0.08
                    v_ref_new = min(self.max_v, k / (kappa + 1e-6))
                # smooth v_ref across MPC horizon
                v_ref = self.v_ref_alpha * self.v_ref_prev + (1.0 - self.v_ref_alpha) * v_ref_new
                # speed-tracking penalty (moderate)
                cost += 0.2 * ((v - v_ref)**2)

                # optional: costmap penalty (light)
                if self.costmap is not None:
                    gx, gy = self.world_to_grid(x_pred, y_pred)
                    h, wmap = self.costmap.shape
                    if 0 <= gx < wmap and 0 <= gy < h:
                        cost += float(self.costmap[gy, gx]) * 0.001

                # update prevs for next step
                v_prev = v
                w_prev = w

            return cost

        # run optimizer with bounds and warm start
        res = minimize(mpc_cost, u0, bounds=bounds, method='SLSQP', options={'maxiter': 50, 'ftol':1e-3})
        if res.success:
            u_opt = res.x
            # save for warm start next cycle
            self.prev_u = u_opt.copy()
            v_cmd = float(u_opt[0])
            w_cmd = float(u_opt[1])
            # update v_ref_prev for smoothing next loop (use horizon[0] curvature based v_ref)
            # recompute v_ref_new for current step to store
            idx0 = 0
            kappa0 = self.compute_path_curvature(self.path, min(idx0, max(0, len(self.path)-2)))
            if kappa0 < 0.01:
                v_ref_new0 = self.max_v
            else:
                k = 0.08
                v_ref_new0 = min(self.max_v, k / (kappa0 + 1e-6))
            self.v_ref_prev = self.v_ref_alpha * self.v_ref_prev + (1.0 - self.v_ref_alpha) * v_ref_new0
        else:
            self.get_logger().warn("[MPC] Optimization failed, using safe stop/hold")
            v_cmd = 0.0
            w_cmd = 0.0

        # apply low-pass filter to command to avoid jumps
        v_out = self.cmd_alpha * self.last_v + (1.0 - self.cmd_alpha) * v_cmd
        w_out = self.cmd_alpha * self.last_w + (1.0 - self.cmd_alpha) * w_cmd

        # publish
        cmd = Twist()
        cmd.linear.x = float(np.clip(v_out, 0.0, self.max_v))
        cmd.angular.z = float(np.clip(w_out, -self.max_w, self.max_w))
        self.cmd_pub.publish(cmd)

        # store last applied for next iteration
        self.last_v = cmd.linear.x
        self.last_w = cmd.angular.z

        self.get_logger().info(f"[MPC] v_cmd={cmd.linear.x:.3f}, w_cmd={cmd.angular.z:.3f}, dist_goal={dist_to_goal:.3f}")

def main(args=None):
    rclpy.init(args=args)
    node = MPCTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
