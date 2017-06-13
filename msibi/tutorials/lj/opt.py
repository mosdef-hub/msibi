import itertools


import numpy as np
import os

from msibi import MSIBI, State, Pair, mie


# **** 2017_05_31 Notes ******
# Clear out the temp files
# back up, not hard coded, not file-based (development dependent)
# later - storing data (backups potentially), some fiels may not exist in future
os.system('rm state*/_* rdfs/pair* potentials/* f_fits.log state*/log.txt')
os.system('rm state*/err.txt')
os.system('rm state*/query.dcd')

# Set up global parameters.
rdf_cutoff = 5.0
opt = MSIBI(rdf_cutoff=rdf_cutoff, n_rdf_points=101, pot_cutoff=3.0,
        smooth_rdfs=True)

# Specify states.
state0 = State(kT=0.5, state_dir='./state0', top_file='start.hoomdxml',
               name='state0', backup_trajectory=True)
state1 = State(kT=1.5, state_dir='./state1', top_file='start.hoomdxml',
               name='state1', backup_trajectory=True)
state2 = State(kT=2.0, state_dir='./state2', top_file='start.hoomdxml',
               name='state2', backup_trajectory=True)
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
pairs = [pair0]  # optimize() expects a list of pairs

# Do magic.
opt.optimize(states, pairs, n_iterations=5, engine='hoomd')
