import os

import numpy as np
import pytest

from msibi import MSIBI, State, Bond, Angle 
from msibi.potentials import save_table_potential

from .base_test import BaseTest


class TestAngle(BaseTest):
    def test_angle_name(self, angle):
        assert angle.name == "0-1-2"

    def test_set_harmonic(self):
        angle = Angle("0", "1", "2")
        angle.set_harmonic(k=500, theta0=2)
        assert angle.angle_type == "static"
        assert "k=500" in angle.angle_entry
        assert "t0=2" in angle.angle_entry

    def test_set_quadratic(self):
        angle = Angle("0", "1", "2")
        angle.set_quadratic(1, 1, 1, 1)
        assert angle.angle_type == "table"

    def test_save_table_potential(self, tmp_path):
        angle = Angle("0", "1", "2")
        angle.set_quadratic(1, 1, 1, 1)
        angle.potential_file = os.path.join(tmp_path, "pot.txt")
        save_table_potential(
                angle.potential,
                angle.theta_range,
                angle.dtheta,
                None,
                angle.potential_file
        )
        assert os.path.isfile(angle.potential_file)

