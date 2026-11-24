import os
import sys

import pytest

# Import the package from source; no ROS or colcon build needed.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maze_control.common.ball_sim import BallSim, run_closed_loop  # noqa: E402
from maze_control.common.config import load_params  # noqa: E402
from maze_control.lqr.controller import LqrPathController  # noqa: E402
from maze_control.pid.controller import PidPathController  # noqa: E402
from maze_control.common.scenarios import SCENARIOS, corridors_for  # noqa: E402
from maze_control.mpc.controller import MpcPathController  # noqa: E402


@pytest.fixture
def pid_params():
    return load_params('pid.yaml')


@pytest.fixture
def sim_params():
    return load_params('ball_sim.yaml')


@pytest.fixture
def run(pid_params, sim_params):
    """run(name, **sim_overrides) -> log for a named scenario."""
    def _run(name, ctrl_params=None, state_until=None, **sim_overrides):
        path, p0, T = SCENARIOS[name]
        ctrl = PidPathController.from_params(ctrl_params or pid_params)
        sim = BallSim.from_params(sim_params, pos0=p0, **sim_overrides)
        return run_closed_loop(ctrl, sim, path, T, state_until=state_until), path
    return _run


@pytest.fixture
def lqr_params():
    return load_params('lqr.yaml')


@pytest.fixture
def run_lqr(lqr_params, sim_params):
    """run_lqr(name, ctrl_params=None, state_until=None, **sim_overrides)."""
    def _run(name, ctrl_params=None, state_until=None, **sim_overrides):
        path, p0, T = SCENARIOS[name]
        ctrl = LqrPathController.from_params(ctrl_params or lqr_params)
        sim = BallSim.from_params(sim_params, pos0=p0, **sim_overrides)
        return run_closed_loop(ctrl, sim, path, T, state_until=state_until), path
    return _run


@pytest.fixture
def mpc_params():
    return load_params('mpc.yaml')


@pytest.fixture
def run_mpc(mpc_params, sim_params):
    """run_mpc(name, half_width=None, ctrl_params=None, **sim_overrides) ->
    (log, path, ctrl). half_width: corridors from corridors_for; None = none."""
    def _run(name, half_width=None, ctrl_params=None, state_until=None,
             **sim_overrides):
        path, p0, T = SCENARIOS[name]
        ctrl = MpcPathController.from_params(ctrl_params or mpc_params)
        if half_width is not None:
            arc, corr, pose_corr = corridors_for(path, half_width)
            ctrl.on_route(path, arc, corr, pose_corr, plan_id=1)
        # run_closed_loop's on_path with the same points keeps the route.
        sim = BallSim.from_params(sim_params, pos0=p0, **sim_overrides)
        log = run_closed_loop(ctrl, sim, path, T, state_until=state_until)
        return log, path, ctrl
    return _run
