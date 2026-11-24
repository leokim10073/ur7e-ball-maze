"""Robotiq Hand-E STEP files -> visual meshes for the sim (maze_sim_description/meshes/hande/).

Each mesh keeps its STEP file's own frame, in metres; scene_builder places them:
  - hande_base.stl: the gripper housing with its two finger carriers, as in the STEP (carriers fixed:
    in the sim they only ever sit near the closed-on-the-tab position, so they stay visual).
    STEP frame: +y is the approach axis (flange at -y, fingers at +y), x is the closing axis, and the
    carriers' finger mounting face is at y = 14.6 mm.
  - hande_finger_vgroove.stl: one V-groove fingertip. STEP frame: +y along the finger (mounting face
    at y = 0, tip at 40 mm), z = 0 is the grip face (V-groove 2 mm deep), x across its 21 mm width.

Needs OCP (cadquery-ocp), which is not in the maze-sim image:
    python3 -m venv /tmp/ocp && /tmp/ocp/bin/pip install cadquery-ocp
    /tmp/ocp/bin/python maze_CAD/scripts/export_hande.py
"""

import argparse
from pathlib import Path

from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.gp import gp_Trsf
from OCP.StlAPI import StlAPI_Writer
from OCP.STEPControl import STEPControl_Reader

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "maze_CAD" / "gripper" / "robotiq_hande"
OUT = ROOT / "src" / "maze_sim_description" / "meshes" / "hande"
PARTS = {  # output name: (STEP file, linear deflection in mm)
    "hande_base.stl": ("ROBOTIQ_HAND-E_Base.step", 0.4),
    "hande_finger_vgroove.stl": ("HAND_E_FINGER_VGROOVE.step", 0.05),
}


def export(step, stl, deflection_mm):
    r = STEPControl_Reader()
    if r.ReadFile(str(step)) != 1:
        raise RuntimeError(f"cannot read {step}")
    r.TransferRoots()
    t = gp_Trsf()
    t.SetScaleFactor(0.001)  # mm -> m
    shape = BRepBuilderAPI_Transform(r.OneShape(), t, True).Shape()
    BRepMesh_IncrementalMesh(shape, deflection_mm * 0.001, False, 0.5, True)
    w = StlAPI_Writer()
    w.ASCIIMode = False
    if not w.Write(shape, str(stl)):
        raise RuntimeError(f"cannot write {stl}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, (step, defl) in PARTS.items():
        export(SRC / step, args.out / name, defl)
        print(f"{args.out / name}: {(args.out / name).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
