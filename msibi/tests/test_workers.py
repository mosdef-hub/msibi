import os

import pytest

from msibi.utils.exceptions import UnsupportedEngine
from msibi.workers import _post_query, run_query_simulations

from .base_test import BaseTest


class TestWorkers(BaseTest):
    def test_unsupported_engine(self):
        engine = "crammps"
        with pytest.raises(UnsupportedEngine):
            run_query_simulations(["margaret", "thatcher"], engine=engine)

    def test_post_query(self, state0):
        pair, state, rdf = state0
        _post_query(state)
        assert state.traj is not None
        assert os.path.isfile(os.path.join(state.state_dir, "_.0.log.txt"))
        assert os.path.isfile(os.path.join(state.state_dir, "_.0.err.txt"))
