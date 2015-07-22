import os
import pytest

from msibi.tests.test_pair import init_state
from msibi.utils.exceptions import UnsupportedEngine
from msibi.workers import run_query_simulations
from msibi.workers import _post_query


def test_unsupported_engine():
    engine = 'crammps'
    with pytest.raises(UnsupportedEngine):
        run_query_simulations(['margaret', 'thatcher'], engine=engine)


def test__post_query():
    pair, state0, rdf = init_state(0)
    _post_query(state0)
    assert state0.traj is not None
    assert os.path.isfile(os.path.join(state0.state_dir, '_.0.log.txt'))
    assert os.path.isfile(os.path.join(state0.state_dir, '_.0.err.txt'))
