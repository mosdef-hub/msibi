import os
import pytest
import tempfile

import mdtraj as md
import numpy as np

from msibi.potentials import mie
from msibi.pair import Pair
from msibi.state import State


dr = 0.1/6.0
r = np.arange(0, 2.5+dr, dr)
pot_r = np.arange(0, 2.0+dr, dr)
r_range = np.asarray([0.0, 2.5+dr])
n_bins = 151

@pytest.mark.skipif(True, reason='This function used for setting up tests')
def init_state():
    pair = Pair('0', '1', potential=mie(r, 1.0, 1.0))
    topology_filename = os.path.join(os.getcwd(), 'final.hoomdxml')
    if not os.path.isfile(topology_filename):
        topology_filename = 'msibi/tests/state/sys.hoomdxml'
    traj_filename = 'state/query.dcd'
    if not os.path.isfile(traj_filename):
        traj_filename = 'msibi/tests/state/query.dcd'
    t = md.load(traj_filename, top=topology_filename)
    pair_list = t.top.select_pairs('name "0"', 'name "1"')
    rdf_filename = 'state/rdf.txt'
    if not os.path.isfile(rdf_filename):
        rdf_filename = 'msibi/tests/state/rdf.txt'
    rdf = np.loadtxt(rdf_filename)
    alpha = 0.5
    state_dir = 'state/'
    if not os.path.isdir(state_dir):
        state_dir = 'msibi/tests/state/'
    state = State(1.987e-3, 305.0, state_dir=state_dir, 
            top_file='sys.hoomdxml', name='state0')
    pair.add_state(state, rdf, alpha, pair_list)
    return pair, state, rdf

def test_pair_name():
    pair, state, rdf = init_state()
    assert(pair.name == '0-1')

def test_save_table_potential():
    pair = Pair('A', 'B', potential=mie(r, 1.0, 1.0))
    pair.potential_file = tempfile.mkstemp()[1]
    pair.save_table_potential(r, dr)
    assert os.path.isfile(pair.potential_file)

def test_add_state():
    pair, state, rdf = init_state()
    assert(np.array_equal(pair.states[state]['target_rdf'], rdf))
    assert(pair.states[state]['current_rdf'] == None)
    assert(pair.states[state]['alpha'] == 0.5)
    assert(len(pair.states[state]['pair_indices']) == 145152)
    assert(len(pair.states[state]['f_fit']) == 0)

def test_calc_current_rdf_no_smooth():
    pair, state, rdf = init_state()
    state.reload_query_trajectory()
    pair.compute_current_rdf(state, r_range, n_bins+1, smooth=False, max_frames=1e3)
    assert(pair.states[state]['current_rdf']) != None
    assert(len(pair.states[state]['f_fit']) > 0)

def test_calc_current_rdf_smooth():
    pair, state, rdf = init_state()
    state.reload_query_trajectory()
    pair.compute_current_rdf(state, r_range, n_bins+1, smooth=True, max_frames=1e3)
    assert(pair.states[state]['current_rdf']) != None
    assert(len(pair.states[state]['f_fit']) > 0)

def test_save_current_rdf():
    pair, state, rdf = init_state()
    state.reload_query_trajectory()
    pair.compute_current_rdf(state, r_range, n_bins+1, smooth=True, max_frames=1e3)
    pair.save_current_rdf(state, 0, 0.1/6.0)
    if not os.path.isdir('rdfs'):
        os.system('mkdir rdfs')
    assert os.path.isfile('rdfs/pair_0-1-state_state0-step0.txt')

def test_update_potential():
    pair, state, rdf = init_state()
    state.reload_query_trajectory()
    pair.compute_current_rdf(state, r_range, n_bins+1, smooth=True, max_frames=1e3)
    pair.update_potential(np.arange(0, 2.5+dr, dr), r_switch=1.8)
    assert(not np.array_equal(pair.potential, pair.previous_potential))
