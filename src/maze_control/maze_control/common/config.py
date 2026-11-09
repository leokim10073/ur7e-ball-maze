"""Load the ROS-style parameter YAML files without ROS."""

import os

import yaml

# Source tree only: <pkg>/maze_control/common/config.py -> <pkg>/config
PKG_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(PKG_DIR, 'config')


def load_params(filename, node_name=None):
    """Return the ``ros__parameters`` dict of the first (or named) node."""
    path = filename if os.path.isabs(filename) else \
        os.path.join(CONFIG_DIR, filename)
    with open(path) as f:
        data = yaml.safe_load(f)
    key = node_name or next(iter(data))
    return data[key]['ros__parameters']


def unflatten(flat):
    """{'a.b': 1} -> {'a': {'b': 1}} (ROS 2 reports nested params dotted)."""
    out = {}
    for k, v in flat.items():
        d = out
        *head, last = k.split('.')
        for h in head:
            d = d.setdefault(h, {})
        d[last] = v
    return out
