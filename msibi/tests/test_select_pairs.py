import pytest

import mdtraj as md
import numpy as np

from msibi.testing import get_fn
from msibi.utils.find_exclusions import find_1_n_exclusions


def test_select_pair_no_exclusions():
    """Test pair selection without exclusions"""
    top = md.load(get_fn('2chains.hoomdxml')).top
    pairs = top.select_pairs("name 'tail'", "name 'tail'")
    assert(pairs.shape[0] == 190)

def test_find_1_n_exclusions():
    top = md.load(get_fn('2chains.hoomdxml')).top
    pairs = top.select_pairs("name 'tail'", "name 'tail'")
    to_delete = find_1_n_exclusions(top, pairs, 3)
    assert(to_delete.shape[0] == 28)

def test_select_pair_with_exclusions():
    """Does this just test np.delete()?"""
    top = md.load(get_fn('2chains.hoomdxml')).top
    pairs = top.select_pairs("name 'tail'", "name 'tail'")
    to_delete = find_1_n_exclusions(top, pairs, 3)
    pairs = np.delete(pairs, to_delete, axis=0)
    assert(pairs.shape[0] == 162)
