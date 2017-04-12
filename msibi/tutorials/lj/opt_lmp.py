import itertools


import numpy as np
import os

from msibi import MSIBI, State, Pair, mie


# Clear out the temp files
os.system('rm state*/_* rdfs/pair* potentials/* f_fits.log state*/log.txt')
os.system('rm state*/err.txt')

# Set up global parameters.
rdf_cutoff = 1.0
opt = MSIBI(engine='lammps', rdf_cutoff=rdf_cutoff, n_rdf_points=100, pot_cutoff=1.0,
        smooth_rdfs=True, max_frames=50)

# Specify states.
state0 = State(k=1, T=0.5, state_dir='./state_lmp', top_file='target.pdb',
               traj_file='query.dcd', name='state0', backup_trajectory=True)
states = [state0]

# Specify pairs.
indices = list(itertools.combinations(range(3000), 2))  # all-all for 1468 atoms
initial_guess = mie(opt.pot_r, 0.01, 0.2)  # 1-D array of potential values.
#rdf_target = np.loadtxt('rdfs/lmp.rdf')
rdf_target = np.loadtxt('rdfs/lmp.target.txt')

pair0 = Pair('1', '1', initial_guess, head_correction_form='linear')
alphas = [1.0]

# Add targets to pair.
#for state, target, alpha in zip(states, rdf_targets, alphas):
pair0.add_state(states[0], rdf_target, alphas[0], indices)
pairs = [pair0]  # optimize() expects a list of pairs

# Do magic.
opt.optimize(states, pairs, n_iterations=10)
