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
REST_TIME = 3.0       # s, stand still after a kick, then look for the ball again
CONTACT_X = 0.3       # m, the ball counts as reached once it is this close
FOLLOW_THROUGH = 0.6  # s, keep walking after reaching the ball


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
        self.contact_time = None
        self.done_time = None

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
            if self.contact_time is None and self.ball[0] < CONTACT_X:
                self.contact_time = now
            if self.contact_time is not None:
                if now - self.contact_time > FOLLOW_THROUGH:
                    self.state, self.done_time = DONE, now
                    return 0.0, 0.0, 0.0
                return KICK_SPEED, 0.0, 0.0  # walk straight through the ball
            if now - self.kick_start > KICK_TIME:
                self.state, self.done_time = DONE, now
                return 0.0, 0.0, 0.0
            aim = math.atan2(self.ball[1] - self.foot_y, self.ball[0])
            return KICK_SPEED, 0.0, clip(0.8 * aim, -0.2, 0.2)
        if self.state == DONE:
            if now - self.done_time < REST_TIME:
                return 0.0, 0.0, 0.0
            # forget the old position, the ball has moved
            self.state, self.ball, self.last_seen, self.contact_time = SEARCH, None, -1e9, None

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
