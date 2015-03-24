from msibi.potentials import mie
from msibi.pair import Pair


def test_save_table_potential():
    pair = Pair('A', 'B', potential=mie(1.0, 1.0))
    filename = ''