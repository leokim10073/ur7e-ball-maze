"""Two-axis PID tilt controller (pure numpy, no ROS).

Per axis i in {x, y}:

    u_i = kp * e_i  -  kd * v_i  +  ki * integral(e_i)

* e = target - position. The D term uses the velocity from the Kalman
  filter (derivative on measurement), never a finite difference of position.
* u_i is the tilt that accelerates the ball towards +i, which is exactly
  ``tilt_x`` / ``tilt_y`` of maze_interfaces/BoardTilt (docs/interfaces.md,
  contract 4), so the output needs no sign or axis conversion. It is clipped
  to +-max_tilt first and then rate limited to +-max_rate.
* Anti-windup: the integrator only runs when |e_i| < i_zone (it is reset
  outside), its contribution is clipped to +-i_max, and it is frozen whenever
  the output is limited and integrating would push further into the limit.
"""

import numpy as np


class PIDController:
    def __init__(self, kp, ki, kd, i_zone, i_max, max_tilt, max_rate):
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.i_zone = float(i_zone)
        self.i_max = float(i_max)
        self.max_tilt = float(max_tilt)
        self.max_rate = float(max_rate)
        self.integral = np.zeros(2)
        self.u = np.zeros(2)          # last output (tilt_x, tilt_y)

    def reset(self):
        """Clear the integrator (the output keeps ramping from where it is)."""
        self.integral[:] = 0.0

    def _limit(self, u_des, dt):
        u_sat = np.clip(u_des, -self.max_tilt, self.max_tilt)
        step = self.max_rate * max(dt, 0.0)
        return self.u + np.clip(u_sat - self.u, -step, step)

    def step(self, target, pos, vel, dt):
        """One control step. Returns np.array([tilt_x, tilt_y]) in rad."""
        e = np.asarray(target, float) - np.asarray(pos, float)
        v = np.asarray(vel, float)

        in_zone = np.abs(e) < self.i_zone
        integral_new = np.where(in_zone, self.integral + e * dt, 0.0)
        if self.ki > 0.0:
            lim = self.i_max / self.ki
            integral_new = np.clip(integral_new, -lim, lim)

        u_des = self.kp * e - self.kd * v + self.ki * integral_new
        u_out = self._limit(u_des, dt)

        # Conditional integration: freeze an axis whose output is limited
        # when the error would drive it further into the limit.
        limited = np.abs(u_des - u_out) > 1e-12
        winding = limited & (np.sign(e) == np.sign(u_des - u_out))
        self.integral = np.where(winding & in_zone, self.integral,
                                 integral_new)

        self.u = u_out
        return u_out.copy()

    def step_zero(self, dt):
        """Ramp the tilt back to zero (rate limited) and clear the integrator."""
        self.reset()
        self.u = self._limit(np.zeros(2), dt)
        return self.u.copy()
