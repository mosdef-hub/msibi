import itertools
import string
import os

import numpy as np

from msibi import MSIBI, State, Pair, mie


# Set up global parameters.
rdf_cutoff = 5.0
opt = MSIBI(rdf_cutoff=rdf_cutoff, n_rdf_points=201, pot_cutoff=3.0,
        smooth_rdfs=True)

# Specify states.
stateA = State(kT=0.5, state_dir='./state_A', top_file='start.hoomdxml',
               name='stateA', backup_trajectory=True)
stateB = State(kT=1.5, state_dir='./state_B', top_file='start.hoomdxml',
               name='stateB', backup_trajectory=True)
stateC = State(kT=2.0, state_dir='./state_C', top_file='start.hoomdxml',
               name='stateC', backup_trajectory=True)
states = [stateA, stateB, stateC]

# Specify pairs.
indices = list(itertools.combinations(range(1024), 2))  # all-all for 1024 atoms

initial_guess = mie(opt.pot_r, 1.0, 1.0)  # 1-D array of potential values.
alphabet = ['A', 'B', 'C']
rdf_targets = [np.loadtxt('rdfs/C3-C3-state_{0}.txt'.format(i))
                for i in alphabet]

pair0 = Pair('C3', 'C3', initial_guess)
alphas = [1.0, 1.0, 1.0]

# Add targets to pair.
for state, target, alpha in zip(states, rdf_targets, alphas):
    pair0.add_state(state, target, alpha, indices)
pairs = [pair0]  # optimize() expects a list of pairs

# Do magic.
opt.optimize(states, pairs, n_iterations=5, engine='hoomd')
