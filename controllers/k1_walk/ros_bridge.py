"""ROS 2 side of the K1 Webots controller.

Publishes  /clock               simulation time (use_sim_time:=true in other nodes)
           /camera/image_raw     head camera image (bgra8)
           /camera/camera_info   camera intrinsics
           /tf                  odom -> base_footprint (true pose from the Supervisor)
                                base_footprint -> head_camera
Subscribes /cmd_vel             geometry_msgs/Twist walking command
                                 (linear.x forward, linear.y left, angular.z turn)
           /head/command        std_msgs/Float64MultiArray [yaw, pitch] head angles in rad
"""
import math

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float64MultiArray
from tf2_ros import TransformBroadcaster

CMD_TIMEOUT = 0.5  # seconds without /cmd_vel before the robot stops


class RosBridge:
    def __init__(self, robot, camera):
        rclpy.init()
        self.node = rclpy.create_node("k1_sim")
        self.robot = robot
        self.camera = camera
        self.clock_pub = self.node.create_publisher(Clock, "/clock", 10)
        self.image_pub = self.node.create_publisher(Image, "/camera/image_raw", 1)
        self.info_pub = self.node.create_publisher(CameraInfo, "/camera/camera_info", 1)
        self.tf_pub = TransformBroadcaster(self.node)
        self.body = robot.getSelf()
        # getFromDevice wants the raw device tag in the R2025a Python API
        self.camera_node = robot.getFromDevice(camera._tag)
        self.node.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 1)
        self.node.create_subscription(Float64MultiArray, "/head/command", self._on_head, 1)
        self.head_cmd = None
        self.head_time = -1.0
        self.cmd_vel = None
        self.cmd_vel_time = -1.0
        self.camera_info = self._make_camera_info()
        self.node.get_logger().info("K1 ROS 2 bridge ready")

    def _make_camera_info(self):
        w, h = self.camera.getWidth(), self.camera.getHeight()
        f = (w / 2) / math.tan(self.camera.getFov() / 2)  # pinhole model, square pixels
        info = CameraInfo()
        info.header.frame_id = "head_camera"
        info.width, info.height = w, h
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        info.k = [f, 0.0, w / 2, 0.0, f, h / 2, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [f, 0.0, w / 2, 0.0, 0.0, f, h / 2, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def _on_cmd_vel(self, msg):
        self.cmd_vel = (msg.linear.x, msg.linear.y, msg.angular.z)
        self.cmd_vel_time = self.robot.getTime()

    def _on_head(self, msg):
        if len(msg.data) == 2:
            self.head_cmd = (msg.data[0], msg.data[1])
            self.head_time = self.robot.getTime()

    def head(self):
        """Latest /head/command as (yaw, pitch), or None if nothing arrived recently."""
        if self.head_cmd is None or self.robot.getTime() - self.head_time > CMD_TIMEOUT:
            return None
        return self.head_cmd

    def _stamp(self):
        t = self.robot.getTime()
        clock = Clock()
        clock.clock.sec = int(t)
        clock.clock.nanosec = int((t - int(t)) * 1e9)
        return clock.clock

    def spin(self):
        """Publish the clock and handle incoming messages. Call every policy step."""
        clock = Clock()
        clock.clock = self._stamp()
        self.clock_pub.publish(clock)
        rclpy.spin_once(self.node, timeout_sec=0)

    def publish_camera(self):
        stamp = self._stamp()
        img = Image()
        img.header.stamp = stamp
        img.header.frame_id = "head_camera"
        img.width, img.height = self.camera.getWidth(), self.camera.getHeight()
        img.encoding = "bgra8"
        img.step = 4 * img.width
        img.data = self.camera.getImage()
        self.image_pub.publish(img)
        self.camera_info.header.stamp = stamp
        self.info_pub.publish(self.camera_info)
        self._publish_tf(stamp)

    def _publish_tf(self, stamp):
        """base_footprint is the trunk projected on the ground, turned only by its yaw."""
        trunk_rot = np.array(self.body.getOrientation()).reshape(3, 3)
        yaw = math.atan2(trunk_rot[1, 0], trunk_rot[0, 0])
        c, s = math.cos(yaw), math.sin(yaw)
        footprint_rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        footprint_pos = np.array(self.body.getPosition()) * [1, 1, 0]
        cam_rot = np.array(self.camera_node.getOrientation()).reshape(3, 3)
        cam_pos = np.array(self.camera_node.getPosition())
        self.tf_pub.sendTransform([
            _transform(stamp, "odom", "base_footprint", footprint_pos, footprint_rot),
            _transform(stamp, "base_footprint", "head_camera",
                       footprint_rot.T @ (cam_pos - footprint_pos), footprint_rot.T @ cam_rot),
        ])

    def command(self):
        """Latest /cmd_vel as (vx, vy, vyaw), or None if nothing arrived recently."""
        if self.cmd_vel is None or self.robot.getTime() - self.cmd_vel_time > CMD_TIMEOUT:
            return None
        return self.cmd_vel


def _transform(stamp, parent, child, pos, rot):
    tf = TransformStamped()
    tf.header.stamp = stamp
    tf.header.frame_id = parent
    tf.child_frame_id = child
    tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, pos)
    q = tf.transform.rotation
    q.x, q.y, q.z, q.w = _quaternion(rot)
    return tf


def _quaternion(r):
    """Rotation matrix to quaternion (x, y, z, w)."""
    w = math.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    return x, y, z, w
