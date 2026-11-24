# Vendored MuJoCo Menagerie models

Copied unmodified from https://github.com/google-deepmind/mujoco_menagerie at commit
`98a70c569c4c1b6da467a20012f766ee1f8dc521` (2026-11-02):

- `universal_robots_ur5e/` (`ur5e.xml`, `assets/`, `LICENSE`)

Each model keeps its own LICENSE. (The Robotiq 2F-85 was here until the sim switched to the lab's Hand-E, built in `scene_builder.py` from Robotiq's STEP files.) The UR5e stands in for the UR7e: `ur_description`
ships identical kinematics, physical parameters and joint limits for both.
`maze_sim_description/scene_builder.py` changes actuators and adds the scene at build time;
these files are never edited. To update, copy the same paths from a newer commit and change the hash above.
