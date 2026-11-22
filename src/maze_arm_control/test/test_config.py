"""config/tilt_control.yaml agrees with the sim it drives and with maze_arm's FK model."""

import json
from pathlib import Path

import numpy as np
import yaml

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parents[1]


def _params():
    return yaml.safe_load((PKG / "config" / "tilt_control.yaml").read_text())["tilt_control"]["ros__parameters"]


def test_home_is_the_sim_start_pose():
    sim = yaml.safe_load((REPO / "src/maze_sim_description/config/sim.yaml").read_text())
    assert _params()["home_joints"] == sim["robot"]["initial_joints"]


def test_mount_is_the_sim_grasp_and_cad_handle():
    p = _params()["mount"]
    sim = yaml.safe_load((REPO / "src/maze_sim_description/config/sim.yaml").read_text())
    report = json.loads((REPO / "maze_CAD/levels/L1/sim/parts_report.json").read_text())
    assert p["grasp_xyz"] == sim["board"]["grasp_offset"]["xyz"]
    assert p["grasp_rpy"] == sim["board"]["grasp_offset"]["rpy"]
    assert np.allclose(p["handle_center"], np.asarray(report["handle"]["center_mm"]) / 1000.0, atol=1e-12)


def test_dh_is_maze_arms():
    import sys

    sys.path.insert(0, str(REPO / "src" / "maze_arm"))
    from maze_arm import kinematics

    k = _params()["kinematics"]
    assert np.allclose(k["d"], kinematics.D) and np.allclose(k["a"], kinematics.A)
    assert np.allclose(k["alpha"], kinematics.ALPHA)


def test_hard_limits_are_contract_4s():
    p = _params()
    assert p["limits"] == {"max_tilt": 0.087, "max_rate": 0.5}
    assert p["watchdog"]["command_timeout"] == 0.2 and p["watchdog"]["state_timeout"] == 0.02
