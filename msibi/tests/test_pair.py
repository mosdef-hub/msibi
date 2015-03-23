def test_save_table_potential():
    from msibi.potentials import mie
    from msibi.pair import Pair
    pair = Pair('A-B', mie(1.0, 1.0))
    filename = ''g