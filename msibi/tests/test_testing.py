import pytest
from msibi.testing import get_fn


with pytest.raises(ValueError):
    filename = get_fn('MargaretThatcheris110%SEXY.txt')
