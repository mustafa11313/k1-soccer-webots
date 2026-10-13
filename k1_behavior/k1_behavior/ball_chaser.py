"""Ball chaser node: walks the K1 to the ball and stops with the ball in front of the right foot.

Subscribes /ball/position   geometry_msgs/PointStamped (camera frame, from ball_detector)
Publishes  /cmd_vel         geometry_msgs/Twist walking command
           /chaser/state    std_msgs/String  SEARCH, APPROACH, ALIGN or READY
Uses TF base_footprint <- head_camera (published by the Webots controller).
"""
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, Twist
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from k1_behavior.chaser_logic import BallChaser


def quat_to_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class BallChaserNode(Node):
    def __init__(self):
        super().__init__("ball_chaser")
        self.chaser = BallChaser()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.ball = None
        self.create_subscription(PointStamped, "/ball/position", self.on_ball, 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.state_pub = self.create_publisher(String, "/chaser/state", 10)
        self.create_timer(0.05, self.on_timer)
        self.last_state = None
        self.get_logger().info("Ball chaser ready")

    def on_ball(self, msg):
        try:
            tf = self.tf_buffer.lookup_transform("base_footprint", msg.header.frame_id, Time())
        except TransformException as e:
            self.get_logger().warn(f"No transform yet: {e}", throttle_duration_sec=2.0)
            return
        t, q = tf.transform.translation, tf.transform.rotation
        p = quat_to_matrix(q) @ np.array([msg.point.x, msg.point.y, msg.point.z])
        self.ball = (p[0] + t.x, p[1] + t.y)

    def on_timer(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        vx, vy, vyaw = self.chaser.update(self.ball, now)
        self.ball = None
        cmd = Twist()
        cmd.linear.x, cmd.linear.y, cmd.angular.z = vx, vy, vyaw
        self.cmd_pub.publish(cmd)
        self.state_pub.publish(String(data=self.chaser.state))
        if self.chaser.state != self.last_state:
            self.get_logger().info(f"State: {self.chaser.state}")
            self.last_state = self.chaser.state


def main():
    rclpy.init()
    rclpy.spin(BallChaserNode())


if __name__ == "__main__":
    main()
