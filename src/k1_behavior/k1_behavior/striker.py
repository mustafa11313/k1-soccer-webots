"""Striker node: finds the ball and dribbles it into the goal.

Subscribes /ball/position   geometry_msgs/PointStamped (camera frame, from ball_detector)
Publishes  /cmd_vel         geometry_msgs/Twist walking command
           /head/command    std_msgs/Float64MultiArray [yaw, pitch] in rad, the head follows the ball
           /chaser/state    std_msgs/String  SEARCH, AROUND, BEHIND, DRIBBLE or SCORED
Parameters left_post, right_post: goal post positions [x, y] in odom (the Webots world
           frame here), as seen from the field: left post first.

The ball is tracked in odom (k1_behavior/tracking.py), so it is not lost when the
detector misses frames or when the ball is too close to the feet to be seen.

Run: ros2 run k1_behavior striker --ros-args -p use_sim_time:=true
"""
import math

import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty, Float64MultiArray, String

from k1_behavior.ball_chaser import BallChaserNode
from k1_behavior.striker_logic import SCORED, Striker
from k1_behavior.tracking import BallTracker, HeadTracker


class StrikerNode(BallChaserNode):
    def __init__(self):
        super().__init__("striker")
        self.chaser = Striker()
        self.tracker = BallTracker()
        self.head = HeadTracker()
        self.declare_parameter("left_post", [4.5, 1.05])
        self.declare_parameter("right_post", [4.5, -1.05])
        self.posts = [list(self.get_parameter(name).value) + [0.0] for name in ("left_post", "right_post")]
        self.head_pub = self.create_publisher(Float64MultiArray, "/head/command", 10)
        self.kick_pub = self.create_publisher(Empty, "/kick", 10)
        self.get_logger().info("Striker ready")

    def on_ball(self, msg):
        super().on_ball(msg)  # sets self.ball_odom from the camera ray
        if self.ball_odom is not None:
            self.tracker.observe((float(self.ball_odom[0]), float(self.ball_odom[1])),
                                 self.get_clock().now().nanoseconds * 1e-9, self.robot_pose())
            self.ball_odom = None

    def robot_pose(self):
        """(x, y, yaw) of base_footprint in odom, or None if TF is missing."""
        origin = self.transform("odom", "base_footprint", [0.0, 0.0, 0.0])
        ahead = self.transform("odom", "base_footprint", [1.0, 0.0, 0.0])
        if origin is None or ahead is None:
            return None
        return origin[0], origin[1], math.atan2(ahead[1] - origin[1], ahead[0] - origin[0])

    def on_timer(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        posts = [self.transform("base_footprint", "odom", post) for post in self.posts]
        pose = self.robot_pose()
        if posts[0] is None or posts[1] is None or pose is None:
            return
        ball = self.tracker.update(pose, now)
        vx, vy, vyaw = self.chaser.update(ball, tuple(posts[0][:2]), tuple(posts[1][:2]), now,
                                            fresh=self.tracker.age(now) < 0.3)
        if self.chaser.take_kick():
            self.kick_pub.publish(Empty())  # k1_walk.py kicks once it stands calmly
        cmd = Twist()
        cmd.linear.x, cmd.linear.y, cmd.angular.z = vx, vy, vyaw
        self.cmd_pub.publish(cmd)
        self.head_pub.publish(Float64MultiArray(data=list(self.head.update(ball, now))))
        self.state_pub.publish(String(data=self.chaser.state))
        if self.chaser.state != self.last_state:
            self.get_logger().info("GOAL!" if self.chaser.state == SCORED else f"State: {self.chaser.state}")
            self.last_state = self.chaser.state


def main():
    rclpy.init()
    rclpy.spin(StrikerNode())


if __name__ == "__main__":
    main()
