import os
import pytest
import tempfile

import numpy as np

from msibi.potentials import mie
from msibi.pair import Pair


@pytest.mark.skipif(True, reason='This function used for setting up tests')
def init_state():
    pair = Pair('0', '1', potential=mie(r, 1.0, 1.0))
    t = md.load('./state/traj.dcd', top='./state/sys.hoomdxml')
    pair_list = t.top.select_pairs('name "0"', 'name "1"')
    alpha = 0.5
    state = (1.987e-3, 305.0, state_dir='./state/',
             traj_file='./state/traj.dcd', top_file='./state/sys.hoomdxml',
             name='state0')
    pair.add_state(state, rdf, alpha, pair_list)
    return pair, state

def test_pair_name():
    pair, state = init_state()
    assert(pair.name = '0-1')

def test_save_table_potential():
    dr = 0.1
    r = np.arange(0, 5.0+dr, dr)
    pair = Pair('A', 'B', potential=mie(r, 1.0, 1.0))
    pair.potential_file = tempfile.mkstemp()[1]
    pair.save_table_potential(r, dr)
    assert os.path.isfile(pair.potential_file)

def test_add_state():
    pair, state = init_state()
    assert(pair.states[state]['target_rdf'] == rdf)
    assert(pair.states[state]['current_rdf'] == None)
    assert(pair.states[state]['alpha'] == alpha)
    assert(len(pair.states[state]['pair_indices']) == 145152)
    assert(assert(pair.states[state]['f_fit']) == 0)

def test_calc_current_rdf_no_smooth():
    pair, state = init_state()
    pair.compute_current_rdf(state, [0.0, 2.0], 151, smooth=False, max_frames=1e3)
    assert(pair.states[state]['current_rdf']) != None
    assert(len(pair.states[state]['f_fit']) > 0)

def test_calc_current_rdf_smooth():
    pair, state = init_state()
    pair.compute_current_rdf(state, [0.0, 2.0], 151, smooth=True, max_frames=1e3)
    assert(pair.states[state]['current_rdf']) != None
    assert(len(pair.states[state]['f_fit']) > 0)

def test_save_current_rdf():
    pair, state = init_state()
    pair.compute_current_rdf(state, [0.0, 2.0], 151, smooth=True, max_frames=1e3)
    pair.save_current_rdf(state, 0, 0.1/6.0)
    assert os.path.isfile('./rdfs/pair_0-1-state-state-step0.txt')
