"""Ball chasing state machine, without any ROS code so it can be tested on its own.

Input:  ball position in the robot's base_footprint frame (x forward, y left), or None
Output: walking command (vx, vy, vyaw)
"""
import math

SEARCH, APPROACH, ALIGN, READY = "SEARCH", "APPROACH", "ALIGN", "READY"


def clip(v, lo, hi):
    return max(lo, min(hi, v))


class BallChaser:
    def __init__(self, kick_x=0.30, kick_y=-0.07, lost_after=1.5, smoothing=0.3):
        self.kick_x = kick_x          # where the ball should end up, ahead of the
        self.kick_y = kick_y          # right foot (metres, base_footprint frame)
        self.lost_after = lost_after  # seconds without a detection before searching
        self.smoothing = smoothing    # 0..1, weight of each new measurement
        self.state = SEARCH
        self.ball = None
        self.last_seen = -1e9
        self.search_dir = 1.0

    def update(self, ball, now):
        if ball is not None:
            if self.ball is None or now - self.last_seen > self.lost_after:
                self.ball = ball
            else:
                a = self.smoothing
                self.ball = (self.ball[0] + a * (ball[0] - self.ball[0]),
                             self.ball[1] + a * (ball[1] - self.ball[1]))
            self.last_seen = now
            self.search_dir = 1.0 if self.ball[1] >= 0 else -1.0
        if now - self.last_seen > self.lost_after:
            self.state = SEARCH
            return 0.0, 0.0, 0.5 * self.search_dir

        x, y = self.ball
        dist = math.hypot(x, y)
        bearing = math.atan2(y, x)
        if self.state == SEARCH:
            self.state = APPROACH
        if self.state == APPROACH:
            if dist < 0.6:
                self.state = ALIGN
            else:
                vyaw = clip(1.2 * bearing, -0.5, 0.5)
                vx = 0.0 if abs(bearing) > 0.5 else clip(0.5 * (dist - 0.4), 0.1, 0.4)
                return vx, 0.0, vyaw
        if self.state in (ALIGN, READY):
            if dist > 0.8:
                self.state = APPROACH
                return 0.0, 0.0, 0.0
            ex, ey = x - self.kick_x, y - self.kick_y
            if abs(ex) < 0.05 and abs(ey) < 0.03:
                self.state = READY
                return 0.0, 0.0, 0.0
            self.state = ALIGN
            vx = clip(0.6 * ex, -0.15, 0.2)
            vy = clip(1.0 * ey, -0.15, 0.15)
            vyaw = clip(0.8 * bearing, -0.3, 0.3) if abs(ey) > 0.15 else 0.0
            return vx, vy, vyaw
        return 0.0, 0.0, 0.0
