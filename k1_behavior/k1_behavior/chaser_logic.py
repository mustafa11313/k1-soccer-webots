"""Ball chasing state machine, without any ROS code so it can be tested on its own.

Input:  ball position in the robot's base_footprint frame (x forward, y left), or None
Output: walking command (vx, vy, vyaw)

The K1 walking policy in Webots does not track slow commands: below about 0.2 m/s
it stands still, and after a stop it keeps walking for about 0.7 m. So the robot
does not stop in front of the ball. It lines the ball up with its right foot from
a distance and walks through it (a walking kick), steering all the way.
"""
import math

SEARCH, APPROACH, TURN, KICK, DONE = "SEARCH", "APPROACH", "TURN", "KICK", "DONE"

WALK_SPEED = 0.3      # m/s, forward speed while approaching
KICK_SPEED = 0.3      # m/s, forward speed into the ball (faster trips the robot)
TURN_SPEED = 0.4      # rad/s, turning on the spot (slower turns barely move)
MAX_AIM_ERROR = 0.4   # rad, turn on the spot first if the ball is further off than this
KICK_DISTANCE = 0.8   # m, start the kick run-up from here
KICK_TIME = 4.0       # s, give up the run-up after this long
CONTACT_X = 0.3       # m, stop walking once the ball is this close; coasting does the kick


def clip(v, lo, hi):
    return max(lo, min(hi, v))


class BallChaser:
    def __init__(self, foot_y=-0.07, lost_after=1.5, smoothing=0.3):
        self.foot_y = foot_y          # the ball should pass this far left of the trunk
        self.lost_after = lost_after  # seconds without a ball before searching
        self.smoothing = smoothing    # 0..1, weight of each new measurement
        self.state = SEARCH
        self.ball = None
        self.last_seen = -1e9
        self.search_dir = 1.0
        self.kick_start = None

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

        if self.state == KICK:
            if now - self.kick_start > KICK_TIME or (self.ball and self.ball[0] < CONTACT_X):
                self.state = DONE
                return 0.0, 0.0, 0.0
            aim = math.atan2(self.ball[1] - self.foot_y, self.ball[0])
            return KICK_SPEED, 0.0, clip(0.8 * aim, -0.2, 0.2)
        if self.state == DONE:
            return 0.0, 0.0, 0.0

        if now - self.last_seen > self.lost_after:
            self.state = SEARCH
            return 0.0, 0.0, TURN_SPEED * self.search_dir

        x, y = self.ball
        dist = math.hypot(x, y)
        aim = math.atan2(y - self.foot_y, x)  # angle to put the ball in front of the right foot
        if abs(aim) > MAX_AIM_ERROR or (self.state == TURN and abs(aim) > 0.1):
            self.state = TURN
            return 0.0, 0.0, math.copysign(TURN_SPEED, aim)
        if dist < KICK_DISTANCE and abs(aim) < 0.15:
            self.state = KICK
            self.kick_start = now
            return KICK_SPEED, 0.0, 0.0
        self.state = APPROACH
        return WALK_SPEED, 0.0, clip(1.0 * aim, -0.3, 0.3)
