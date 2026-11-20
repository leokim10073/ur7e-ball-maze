import json
from pathlib import Path

import numpy as np
import pytest
import yaml

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parents[1]
SIM_YAML = REPO / "src" / "maze_sim_description" / "config" / "sim.yaml"
PARTS_REPORT = REPO / "maze_CAD" / "levels" / "L1" / "sim" / "parts_report.json"


@pytest.fixture(scope="session")
def sim_cfg():
    return yaml.safe_load(SIM_YAML.read_text())


@pytest.fixture(scope="session")
def initial_joints(sim_cfg):
    return sim_cfg["robot"]["initial_joints"]


@pytest.fixture(scope="session")
def grasp(sim_cfg):
    """(xyz, rpy) of the handle frame in tool0."""
    g = sim_cfg["board"]["grasp_offset"]
    return g["xyz"], g["rpy"]


@pytest.fixture(scope="session")
def handle_center():
    """Handle center in the board frame, m (from the L1 CAD)."""
    report = json.loads(PARTS_REPORT.read_text())
    return np.asarray(report["handle"]["center_mm"]) / 1000.0
