"""Ball chaser node: finds the ball, walks to it and kicks it by walking through it.

Subscribes /ball/position   geometry_msgs/PointStamped (camera frame, from ball_detector)
Publishes  /cmd_vel         geometry_msgs/Twist walking command
           /chaser/state    std_msgs/String  SEARCH, APPROACH, TURN, KICK or DONE
Uses TF odom <- base_footprint <- head_camera (published by the Webots controller).

The ball is remembered in the odom frame, so the robot still knows where it is when
it drops out of view right in front of the feet.
"""
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, Twist
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from k1_behavior.chaser_logic import BallChaser

BALL_RADIUS = 0.11  # m, height of the ball centre above the floor
MEMORY = 10.0  # seconds a ball position stays usable after it was last seen


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
        self.ball_odom = None
        self.ball_time = -1e9
        self.create_subscription(PointStamped, "/ball/position", self.on_ball, 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.state_pub = self.create_publisher(String, "/chaser/state", 10)
        self.create_timer(0.05, self.on_timer)
        self.last_state = None
        self.get_logger().info("Ball chaser ready")

    def transform(self, target, source, point):
        """point (x, y, z) from frame source to frame target, or None if TF is missing."""
        try:
            tf = self.tf_buffer.lookup_transform(target, source, Time())
        except TransformException as e:
            self.get_logger().warn(f"No transform yet: {e}", throttle_duration_sec=2.0)
            return None
        t = tf.transform.translation
        return quat_to_matrix(tf.transform.rotation) @ np.array(point) + [t.x, t.y, t.z]

    def on_ball(self, msg):
        """Intersect the camera ray through the ball with the plane at ball-centre height.

        This only uses the ball's direction in the image, which is much more accurate
        than the distance the detector estimates from the box width.
        """
        camera = self.transform("odom", msg.header.frame_id, [0.0, 0.0, 0.0])
        p = self.transform("odom", msg.header.frame_id, [msg.point.x, msg.point.y, msg.point.z])
        if camera is None or p is None:
            return
        ray = p - camera
        if ray[2] > -1e-3:
            return  # ball at or above the horizon, cannot intersect the floor
        self.ball_odom = camera + ray * (BALL_RADIUS - camera[2]) / ray[2]
        self.ball_time = self.get_clock().now().nanoseconds * 1e-9

    def on_timer(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        ball = None
        if self.ball_odom is not None and now - self.ball_time < MEMORY:
            p = self.transform("base_footprint", "odom", self.ball_odom)
            if p is not None:
                ball = (p[0], p[1])
        vx, vy, vyaw = self.chaser.update(ball, now)
        if self.chaser.state == "DONE":
            self.ball_odom = None  # the kick moved the ball, wait for a fresh detection
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
