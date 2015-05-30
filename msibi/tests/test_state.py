import os
import pytest
import tempfile

import mdtraj as md
import numpy as np

from msibi.potentials import mie
from msibi.pair import Pair
from msibi.state import State


def test_init():
    pass

def test_reload_query_trajectory():
    state_dir = get_fn('state/')
    state = State(1.987e-3, 500.0, state_dir=state_dir, top_file='sys.hoomdxml',
        name='state0')
    state.reload_query_trajectory()
    assert(state.traj)
    assert(state.traj.top)

def test_save_runscript():
    pass
