"""Ball tracking and head control, without any ROS code so it can be tested on its own.

BallTracker keeps one ball estimate in the odom frame (fixed to the field), so the
robot still knows where the ball is when it turns, when the detector misses a frame,
or when the ball is too close to the feet for the camera to see. It
  - ignores single detections that jump far away from a ball it is tracking well,
  - keeps rolling the ball on with its measured speed (slowing down) when unseen,
  - pushes the ball ahead of the feet when the robot walks into it while unseen,
    so a ball that was touched never ends up "behind" the robot.

HeadTracker points the head at the ball, and sweeps it from side to side when the
ball is lost. The camera pose in TF follows the head, so detections stay correct.
"""
import math

JUMP = 1.0          # m, a detection this far from a well tracked ball is suspicious
JUMP_CONFIRM = 6    # detections at the same new place before believing them
SELF_X = 0.12       # m, detections not at least this far in front of the trunk are the robot itself
SMOOTHING = 0.5     # 0..1, weight of each new detection
FRICTION = 1.0      # 1/s, how fast an unseen ball slows down
FOOT_FRONT = 0.28   # m from the trunk, where a pushed ball sits in front of the feet
FOOT_HALF_WIDTH = 0.2


def _to_robot(robot, p):
    x, y, yaw = robot
    dx, dy = p[0] - x, p[1] - y
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def _to_odom(robot, p):
    x, y, yaw = robot
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * p[0] - s * p[1], y + s * p[0] + c * p[1]


class BallTracker:
    def __init__(self, memory=4.0):
        self.memory = memory      # s an estimate stays usable without detections
        self.ball = None          # (x, y) in odom
        self.vel = (0.0, 0.0)
        self.last_seen = -1e9
        self.last_update = None
        self.candidate = None     # a far-away detection waiting to be confirmed
        self.candidate_time = -1e9
        self.candidate_count = 0

    def age(self, now):
        return now - self.last_seen

    def observe(self, ball, now, robot=None):
        """A detection of the ball, (x, y) in odom. robot: the robot pose (x, y, yaw) in odom."""
        if robot is not None and _to_robot(robot, ball)[0] < SELF_X:
            return  # beside or behind the trunk: the camera is looking at the robot's own arm or leg
        near = self.ball is not None and math.hypot(ball[0] - self.ball[0], ball[1] - self.ball[1]) < JUMP
        if not near:
            # far from the tracked ball (or no ball yet): only believe it once it repeats
            # at the same place, so a single false detection cannot drag the track away
            if self.candidate is not None and now - self.candidate_time < 0.5 and \
                    math.hypot(ball[0] - self.candidate[0], ball[1] - self.candidate[1]) < 0.4:
                self.candidate_count += 1
            else:
                self.candidate_count = 1
            self.candidate, self.candidate_time = tuple(ball), now
            if self.candidate_count < JUMP_CONFIRM:
                return
            self.ball, self.vel = tuple(ball), (0.0, 0.0)
        else:
            dt = max(now - self.last_seen, 1e-3)
            new = (self.ball[0] + SMOOTHING * (ball[0] - self.ball[0]),
                   self.ball[1] + SMOOTHING * (ball[1] - self.ball[1]))
            if dt < 0.5:
                v = ((new[0] - self.ball[0]) / dt, (new[1] - self.ball[1]) / dt)
                self.vel = (self.vel[0] + 0.3 * (v[0] - self.vel[0]), self.vel[1] + 0.3 * (v[1] - self.vel[1]))
            self.ball = new
        self.candidate = None
        self.last_seen = now
        self.last_update = now

    def update(self, robot, now):
        """Ball position (x, y) in the robot's base_footprint frame, or None if unknown.

        robot is the robot pose (x, y, yaw) in odom. Call once per control step.
        """
        if self.ball is None or self.age(now) > self.memory:
            self.ball = None
            return None
        dt = 0.0 if self.last_update is None else now - self.last_update
        self.last_update = now
        if self.age(now) > 0.15 and dt > 0:
            # not seen right now: let it roll on and slow down
            self.ball = (self.ball[0] + self.vel[0] * dt, self.ball[1] + self.vel[1] * dt)
            k = math.exp(-FRICTION * dt)
            self.vel = (self.vel[0] * k, self.vel[1] * k)
            x, y = _to_robot(robot, self.ball)
            if x < FOOT_FRONT and x > -0.1 and abs(y) < FOOT_HALF_WIDTH:
                # the robot walked into the ball: it is pushed ahead, not left behind
                self.ball = _to_odom(robot, (FOOT_FRONT, y))
        return _to_robot(robot, self.ball)


HEAD_YAW_LIMIT = 0.7      # rad (joint limit 1.01); further out the camera sees the robot's own arm
HEAD_PITCH_MIN = 0.1      # rad, looking a little down
HEAD_PITCH_MAX = 0.6      # rad (joint limit 0.79); lower the camera sees arms and feet, which look like a ball,
                          # and the head sags about 0.1 rad below its target anyway
CAMERA_HEIGHT = 0.85      # m above the floor, roughly
CAMERA_FORWARD = 0.08     # m in front of the trunk centre
BALL_RADIUS = 0.11
SCAN_PERIOD = 4.0         # s for one sweep left-right-left while the ball is lost


class HeadTracker:
    def __init__(self):
        self.yaw, self.pitch = 0.0, 0.6
        self.scan_start = None

    def update(self, ball, now):
        """Head (yaw, pitch) targets. ball is from BallTracker (base_footprint) or None."""
        if ball is not None:
            # look at the ball, also at where it should be when it has not been seen for a
            # moment (behind the robot the head turns as far as it can to that side)
            self.scan_start = None
            x, y = ball[0] - CAMERA_FORWARD, ball[1]
            dist = max(math.hypot(x, y), 0.05)
            self.yaw = max(-HEAD_YAW_LIMIT, min(HEAD_YAW_LIMIT, math.atan2(y, x)))
            self.pitch = math.atan2(CAMERA_HEIGHT - BALL_RADIUS, dist)
        else:
            if self.scan_start is None:
                self.scan_start = now
            phase = 2 * math.pi * (now - self.scan_start) / SCAN_PERIOD
            self.yaw = HEAD_YAW_LIMIT * math.sin(phase)
            # alternate looking near and far on each sweep
            far = int((now - self.scan_start) / SCAN_PERIOD) % 2 == 1
            self.pitch = 0.3 if far else 0.6
        self.pitch = max(HEAD_PITCH_MIN, min(HEAD_PITCH_MAX, self.pitch))
        return self.yaw, self.pitch
