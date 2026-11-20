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

__all__ = ['SCENARIOS']
