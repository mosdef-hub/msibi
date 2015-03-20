import sys

import memory_profiler
import numpy as np
import mdtraj as md

@profile
def test_dcd():
    traj = md.load('query.dcd', top='start.pdb')

if __name__ == "__main__":
    test_dcd()
