import pytest

from msibi.state import State
from msibi.utils.general import get_fn


@pytest.mark.skipif(True, reason='Needs implementing!')
def test_init():
    pass


def test_reload_query_trajectory():
    state_dir = get_fn('state0/')
    state = State(1.987e-3, 500.0, state_dir=state_dir, top_file='sys.hoomdxml',
                  name='state0')
    state.reload_query_trajectory()
    assert(state.traj)
    assert(state.traj.top)

@pytest.mark.skipif(True, reason='Needs implementing!')
def test_save_runscript():
    pass
