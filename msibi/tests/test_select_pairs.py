import mdtraj as md
import networkx as nx
import numpy as np

from msibi.utils.general import get_fn
from msibi.utils.find_exclusions import find_1_n_exclusions
from msibi.utils.find_exclusions import is_1_n


def test_select_pair_no_exclusions():
    """Test pair selection without exclusions"""
    top = md.load(get_fn("2chains.hoomdxml")).top
    pairs = top.select_pairs("name 'tail'", "name 'tail'")
    assert pairs.shape[0] == 190


def test_find_1_n_exclusions():
    top = md.load(get_fn("2chains.hoomdxml")).top
    pairs = top.select_pairs("name 'tail'", "name 'tail'")
    to_delete = find_1_n_exclusions(top, pairs, 3)
    assert to_delete.shape[0] == 28


def test_select_pair_with_exclusions():
    traj = md.load(get_fn("2chains.hoomdxml"))
    pairs = traj.top.select_pairs("name 'tail'", "name 'tail'")
    to_delete = find_1_n_exclusions(traj.top, pairs, 3)
    pairs = np.delete(pairs, to_delete, axis=0)
    assert pairs.shape[0] == 162


def test_is_exclusion():
    top = md.load(get_fn("2chains.hoomdxml")).top
    G = nx.Graph()
    G.add_nodes_from([a.index for a in top.atoms])
    bonds = [b for b in top.bonds]
    bonds_by_index = [(b[0].index, b[1].index) for b in bonds]
    G.add_edges_from(bonds_by_index)
    tail_tail = top.select_pairs("name 'tail'", "name 'tail'")
    assert is_1_n(tail_tail[0], 3, G)
    assert is_1_n(tail_tail[0], 2, G)
    assert is_1_n(tail_tail[0], 4, G)
    assert is_1_n(tail_tail[2], 4, G)
    assert not is_1_n(tail_tail[2], 3, G)
