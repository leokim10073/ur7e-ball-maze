"""Test paths in the board frame (m). Shared by the tests and the plot script.

Each scenario: (path points N x 2, initial ball position, sim duration in s).
"""

import numpy as np

SCENARIOS = {
    'point': (np.array([[0.10, 0.06]]), (-0.10, -0.08), 6.0),
    'line': (np.array([[-0.20, -0.10], [0.20, 0.10]]), (-0.20, -0.10), 10.0),
    'L': (np.array([[-0.18, 0.15], [-0.18, -0.15], [0.18, -0.15]]),
          (-0.18, 0.15), 14.0),
    'U': (np.array([[-0.15, 0.18], [-0.15, -0.15], [0.0, -0.15],
                    [0.0, 0.18]]), (-0.15, 0.18), 16.0),
}



def corridors_for(path, half_width):
    """Contract-3-style corridors for a test path: one axis-aligned box per
    segment (its bounding box grown by half_width). Returns (arc_length (N,),
    corridors (M, 4) as [x_min, x_max, y_min, y_max], pose_corridor (N,)),
    pose i in corridor min(i, M - 1)."""
    pts = np.asarray(path, float).reshape(-1, 2)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    segs = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)] or [(pts[0], pts[0])]
    corr = np.array([[min(a[0], b[0]) - half_width, max(a[0], b[0]) + half_width,
                      min(a[1], b[1]) - half_width, max(a[1], b[1]) + half_width]
                     for a, b in segs])
    pose_corr = np.minimum(np.arange(len(pts)), len(corr) - 1)
    return arc, corr, pose_corr


__all__ = ['SCENARIOS', 'corridors_for']
