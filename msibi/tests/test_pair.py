import os
import tempfile

import numpy as np

from msibi.potentials import mie
from msibi.pair import Pair


def test_save_table_potential():
    dr = 0.1
    r = np.arange(0, 5.0+dr, dr)
    pair = Pair('A', 'B', potential=mie(r, 1.0, 1.0))
    pair.potential_file = tempfile.mkstemp()[1]
    pair.save_table_potential(r, dr)
    assert os.path.isfile(pair.potential_file)
