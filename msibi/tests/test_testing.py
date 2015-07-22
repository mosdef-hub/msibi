import pytest
from msibi.utils.general import get_fn


with pytest.raises(ValueError):
    filename = get_fn('MargaretThatcheris110%SEXY.txt')
