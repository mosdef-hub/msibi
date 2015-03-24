import itertools
import pdb

import numpy as np

from msibi import *

# Set up global parameters.
rdf_cutoff = 5.0
dr = 0.05
r = np.arange(0.0, rdf_cutoff, dr)
opt = MSIBI(rdf_cutoff=rdf_cutoff, dr=dr)

# Specify states.
state0 = State(k=1, T=0.5, state_dir='./state0', top_file='target.pdb')
state1 = State(k=1, T=1.5, state_dir='./state1', top_file='target.pdb')
state2 = State(k=1, T=2.0, state_dir='./state2', top_file='target.pdb')
states = [state0, state1, state2]

# Specify pairs.
indices = list(itertools.combinations(range(1468), 2))  # all-all for 1468 atoms
initial_guess = mie(opt.pot_r, 1.0, 1.0)  # 1-D array of potential values.
rdf_targets = [np.loadtxt('rdfs/rdf.target{0:d}.t1t1.txt'.format(i))
               for i in range(3)]
pair0 = Pair('1', '1', initial_guess)
alphas = [1.0, 1.0, 1.0]

# Add targets to pair.
for state, target, alpha in zip(states, rdf_targets, alphas):
    pair0.add_state(state, target, alpha, indices)
pairs = [pair0]

# Do magic.
opt.optimize(states, pairs, n_iterations=5, engine='hoomd')
opt.plot()