"""Offline ball-on-tilting-board simulator (pure numpy, no ROS).

Ball (solid sphere, rolling without slipping) on a tilted board:

    a_x = sign_x * (5/7) g sin(tilt_x) - c v_x
    a_y = sign_y * (5/7) g sin(tilt_y) - c v_y

The command is maze_interfaces/BoardTilt's (tilt_x, tilt_y), defined by its
effect on the ball (docs/interfaces.md, contract 4), so the true plant has
sign_x = sign_y = +1. The signs exist so a test can model an arm that gets
the convention wrong.

The board does not follow the command instantly: the command reaches the arm
after a pure delay ``delay`` and the tilt then follows a first-order lag with
time constant ``tau``. Optional Gaussian noise is added to the measured state.
"""

import collections

import numpy as np

G = 9.81


class BallSim:
    def __init__(self, c=0.3, tau=0.08, delay=0.05, sign_x=1.0, sign_y=1.0,
                 dt_phys=0.001, pos0=(0.0, 0.0), vel0=(0.0, 0.0),
                 noise_pos=0.0, noise_vel=0.0, seed=0, g=G):
        self.c, self.tau, self.delay = float(c), float(tau), float(delay)
        self.sign = np.array([float(sign_x), float(sign_y)])
        self.dt_phys = float(dt_phys)
        self.g = float(g)
        self.noise_pos, self.noise_vel = float(noise_pos), float(noise_vel)
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.pos = np.array(pos0, float)
        self.vel = np.array(vel0, float)
        self.tilt = np.zeros(2)            # actual (tilt_x, tilt_y)
        self._queue = collections.deque()  # (t_cmd, tilt)
        self._active = np.zeros(2)         # command the arm is following

    @classmethod
    def from_params(cls, p, **overrides):
        kw = dict(c=p['c'], tau=p['tau'], delay=p['delay'],
                  sign_x=p['sign']['tilt_x_to_ax'],
                  sign_y=p['sign']['tilt_y_to_ay'],
                  dt_phys=p['dt_phys'], noise_pos=p['noise_pos'],
                  noise_vel=p['noise_vel'])
        kw.update(overrides)
        return cls(**kw)

    def command(self, tilt):
        """Send (tilt_x, tilt_y) at the current sim time."""
        self._queue.append((self.t, np.array(tilt, float)))

    def _accel(self, vel, tilt):
        k = 5.0 / 7.0 * self.g
        return self.sign * k * np.sin(tilt) - self.c * vel

    def advance(self, t_end):
        """Integrate (semi-implicit Euler) up to time t_end."""
        while self.t < t_end - 1e-12:
            h = min(self.dt_phys, t_end - self.t)
            while self._queue and self._queue[0][0] <= self.t - self.delay + 1e-12:
                self._active = self._queue.popleft()[1]
            if self.tau > 0:
                self.tilt += (self._active - self.tilt) * (h / self.tau)
            else:
                self.tilt = self._active.copy()
            self.vel += self._accel(self.vel, self.tilt) * h
            self.pos += self.vel * h
            self.t += h

    def measure(self):
        """Measured (pos, vel), as the Kalman filter would publish them."""
        pos = self.pos + self.rng.normal(0.0, self.noise_pos, 2) \
            if self.noise_pos > 0 else self.pos.copy()
        vel = self.vel + self.rng.normal(0.0, self.noise_vel, 2) \
            if self.noise_vel > 0 else self.vel.copy()
        return pos, vel


def run_closed_loop(controller, sim, path, duration, state_until=None):
    """Run controller + sim at the controller rate.

    ``state_until``: stop delivering ball states after this time (tracking
    loss). Returns a dict of numpy arrays.
    """
    controller.on_path(path)
    dt = 1.0 / controller.rate_hz
    n = int(round(duration / dt))
    log = {k: [] for k in ('t', 'pos', 'vel', 'target', 'cmd', 'tilt',
                           'status', 'cross_track')}
    for k in range(n + 1):
        t = k * dt
        sim.advance(t)
        if state_until is None or t <= state_until:
            pos, vel = sim.measure()
            controller.on_state(t, pos, vel)
        cmd = controller.update(t)
        sim.command(cmd)
        tgt = controller.target
        log['t'].append(t)
        log['pos'].append(sim.pos.copy())
        log['vel'].append(sim.vel.copy())
        log['target'].append(np.full(2, np.nan) if tgt is None else tgt.copy())
        log['cmd'].append(cmd.copy())
        log['tilt'].append(sim.tilt.copy())
        log['status'].append(controller.status)
        log['cross_track'].append(controller.tracker.distance_to_path(sim.pos))
    return {k: np.array(v) for k, v in log.items()}
