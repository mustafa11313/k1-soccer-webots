"""Ball detector node.
Subscribes /camera/image_raw, /camera/camera_info
Publishes  /ball/position     geometry_msgs/PointStamped, ball centre in the camera frame
                              (x forward, y left, z up), estimated from the known ball size
           /ball/debug_image  camera image with the detection drawn on it"""
import os

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from ultralytics import YOLO

OUR_MODEL = os.path.expanduser("~/k1_ws/src/models/ball_yolo11n_v4.pt")


class BallDetector(Node):
    def __init__(self):
        super().__init__("ball_detector")
        self.declare_parameter("model", OUR_MODEL)
        self.declare_parameter("ball_class", 0)  # 0 for our model, 32 (sports ball) for yolo11s.pt
        self.declare_parameter("confidence", 0.25)
        self.declare_parameter("ball_diameter", 0.22)  # metres
        self.model = YOLO(self.get_parameter("model").value)
        self.confidence = self.get_parameter("confidence").value
        self.ball_class = self.get_parameter("ball_class").value
        self.ball_diameter = self.get_parameter("ball_diameter").value
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.create_subscription(CameraInfo, "/camera/camera_info", self.on_camera_info, 1)
        self.create_subscription(Image, "/camera/image_raw", self.on_image, 1)
        self.position_pub = self.create_publisher(PointStamped, "/ball/position", 10)
        self.debug_pub = self.create_publisher(Image, "/ball/debug_image", 1)
        self.get_logger().info("Ball detector ready")

    def on_camera_info(self, msg):
        self.camera_matrix = np.array(msg.k).reshape(3, 3)

    def on_image(self, msg):
        if self.camera_matrix is None:
            return
        frame = cv2.cvtColor(self.bridge.imgmsg_to_cv2(msg), cv2.COLOR_BGRA2BGR)
        result = self.model(frame, classes=[self.ball_class], conf=self.confidence, device="cpu", verbose=False)[0]
        if len(result.boxes) > 0:
            best = int(result.boxes.conf.argmax())
            x1, y1, x2, y2 = result.boxes.xyxy[best].tolist()
            conf = float(result.boxes.conf[best])
            colour = (0, 255, 0)
            ball = self.full_ball(x1, y1, x2, y2, frame.shape[1], frame.shape[0])
            if ball is None:
                colour = (0, 165, 255)  # orange: cut off in a corner, size unknown, not published
            else:
                self.publish_position(msg.header, *ball)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)
            cv2.putText(frame, f"ball {conf:.2f}", (int(x1), max(int(y1) - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        debug = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        debug.header = msg.header
        self.debug_pub.publish(debug)

    @staticmethod
    def full_ball(x1, y1, x2, y2, width, height, margin=2):
        """Centre and diameter of the whole ball from a box that may be cut off by the
        image edge. The uncut side still shows the full diameter. Corner: None."""
        cut_x = x1 <= margin or x2 >= width - margin
        cut_y = y1 <= margin or y2 >= height - margin
        if cut_x and cut_y:
            return None
        size = max(x2 - x1, y2 - y1)
        u, v = (x1 + x2) / 2, (y1 + y2) / 2
        if cut_x:
            u = x2 - size / 2 if x1 <= margin else x1 + size / 2
        if cut_y:
            v = y2 - size / 2 if y1 <= margin else y1 + size / 2
        return u, v, size

    def publish_position(self, header, u, v, width_px):
        """Pinhole model: distance = focal length * real size / size in pixels."""
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        depth = fx * self.ball_diameter / width_px
        point = PointStamped()
        point.header = header
        point.point.x = depth                       # forward
        point.point.y = -(u - cx) * depth / fx      # left
        point.point.z = -(v - cy) * depth / fy      # up
        self.position_pub.publish(point)


def main():
    rclpy.init()
    rclpy.spin(BallDetector())


if __name__ == "__main__":
    main()
