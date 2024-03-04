import os

import numpy as np
import pytest

from msibi import MSIBI, State, Bond, Angle 

from .base_test import BaseTest


class TestState(BaseTest):
    def test_state_init(self, tmp_path, stateX):
        assert stateX.name == "X"
        assert stateX.alpha == 1.0
        assert stateX.kT == 1.0
        assert os.path.exists(os.path.join(tmp_path, "states/X_1.0/"))

    def test_n_frames(self, stateX):
        stateX.nframes = 50
        assert stateX.nframes == 50
