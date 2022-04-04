import os
import pytest
from pathlib import Path

from msibi.utils.exceptions import UnsupportedEngine
from msibi.workers import _post_query, run_query_simulations

from .base_test import BaseTest


class TestWorkers(BaseTest):
    def test_post_query(self, state0):
        log_file = os.path.join(state0.dir, "log.txt")
        err_file = os.path.join(state0.dir, "err.txt")
        Path(log_file).touch()
        Path(err_file).touch()

        _post_query(state0)
        assert state0.traj_file is not None
        assert state0.query_traj is not None
        assert os.path.isfile(os.path.join(state0.dir, "_.0.log.txt"))
        assert os.path.isfile(os.path.join(state0.dir, "_.0.err.txt"))
