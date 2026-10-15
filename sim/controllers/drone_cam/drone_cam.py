"""Camera drone and referee for the K1 field world.
Moves itself (Supervisor) behind and above the K1, camera aimed at robot and ball.
Publishes /drone/image_raw (sensor_msgs/Image) and /referee/goal (std_msgs/Bool),
and says "Goal!" when the true ball position crosses the goal line."""
import math

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from controller import Supervisor
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

BEHIND = 2.2      # m behind the robot
HEIGHT = 2.0      # m above the floor
SMOOTHING = 0.02  # 0..1 per step, how fast the drone follows (small = smooth)
GOAL_X = 4.5      # goal line of the kid-size RoboCup field
GOAL_HALF_WIDTH = 1.3
BALL_RADIUS = 0.11

robot = Supervisor()
dt_ms = int(robot.getBasicTimeStep())
me = robot.getSelf()
children = robot.getRoot().getField("children")
k1 = robot.getFromDef("K1") or next(children.getMFNode(i) for i in range(children.getCount())
                                    if children.getMFNode(i).getTypeName().startswith("K1"))
ball = robot.getFromDef("BALL")
camera = robot.getDevice("drone_camera")
camera.enable(33)
speaker = robot.getDevice("speaker")
translation, rotation = me.getField("translation"), me.getField("rotation")

rclpy.init()
node = rclpy.create_node("drone_cam")
image_pub = node.create_publisher(Image, "/drone/image_raw", 1)
goal_pub = node.create_publisher(Bool, "/referee/goal", 10)


def look_rotation(eye, target):
    """Axis-angle rotation that points the drone's x axis (camera) from eye at target."""
    d = np.array(target) - np.array(eye)
    yaw = math.atan2(d[1], d[0])
    pitch = math.atan2(-d[2], math.hypot(d[0], d[1]))
    w = math.cos(yaw / 2) * math.cos(pitch / 2)
    x = -math.sin(yaw / 2) * math.sin(pitch / 2)
    y = math.cos(yaw / 2) * math.sin(pitch / 2)
    z = math.sin(yaw / 2) * math.cos(pitch / 2)
    angle = 2 * math.acos(max(-1.0, min(1.0, w)))
    s = math.sqrt(max(1 - w * w, 1e-12))
    return [x / s, y / s, z / s, angle] if angle > 1e-6 else [0, 0, 1, 0]


eye = np.array(translation.getSFVec3f())
in_goal = False
last_image = -1.0
while robot.step(dt_ms) != -1:
    p = np.array(k1.getPosition())
    rot = np.array(k1.getOrientation()).reshape(3, 3)
    heading = math.atan2(rot[1, 0], rot[0, 0])
    b = np.array(ball.getPosition())
    wanted = np.array([p[0] - BEHIND * math.cos(heading), p[1] - BEHIND * math.sin(heading), HEIGHT])
    eye += SMOOTHING * (wanted - eye)
    target = (p + b) / 2
    target[2] = 0.2
    translation.setSFVec3f(eye.tolist())
    rotation.setSFRotation(look_rotation(eye, target))

    now = robot.getTime()
    if now - last_image >= 0.033:
        last_image = now
        msg = Image()
        msg.header.stamp = Time(sec=int(now), nanosec=int((now % 1) * 1e9))
        msg.header.frame_id = "drone_camera"
        msg.height, msg.width = camera.getHeight(), camera.getWidth()
        msg.encoding, msg.step = "bgra8", 4 * camera.getWidth()
        msg.data = camera.getImage()
        image_pub.publish(msg)

    scored = b[0] > GOAL_X + BALL_RADIUS and abs(b[1]) < GOAL_HALF_WIDTH
    if scored and not in_goal:
        print("REFEREE: GOAL!", flush=True)
        speaker.speak("Goal!", 1.0)
        goal_pub.publish(Bool(data=True))
    in_goal = scored
