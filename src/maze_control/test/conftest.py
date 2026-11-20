import os
import sys

import pytest

# Import the package from source; no ROS or colcon build needed.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maze_control.common.ball_sim import BallSim, run_closed_loop  # noqa: E402
from maze_control.common.config import load_params  # noqa: E402
from maze_control.pid.controller import PidPathController  # noqa: E402
from maze_control.common.scenarios import SCENARIOS  # noqa: E402


@pytest.fixture
def pid_params():
    return load_params('pid.yaml')


@pytest.fixture
def sim_params():
    return load_params('sim.yaml')


@pytest.fixture
def run(pid_params, sim_params):
    """run(name, **sim_overrides) -> log for a named scenario."""
    def _run(name, ctrl_params=None, state_until=None, **sim_overrides):
        path, p0, T = SCENARIOS[name]
        ctrl = PidPathController.from_params(ctrl_params or pid_params)
        sim = BallSim.from_params(sim_params, pos0=p0, **sim_overrides)
        return run_closed_loop(ctrl, sim, path, T, state_until=state_until), path
    return _run
