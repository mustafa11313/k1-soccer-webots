"""ROS 2 side of the K1 Webots controller.
Publishes  /clock, /camera/image_raw (bgra8), /camera/camera_info
Subscribes /cmd_vel (geometry_msgs/Twist: linear.x forward, linear.y left, angular.z turn)"""
import math

import rclpy
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image

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
        self.node.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 1)
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

    def command(self):
        """Latest /cmd_vel as (vx, vy, vyaw), or None if nothing arrived recently."""
        if self.cmd_vel is None or self.robot.getTime() - self.cmd_vel_time > CMD_TIMEOUT:
            return None
        return self.cmd_vel
