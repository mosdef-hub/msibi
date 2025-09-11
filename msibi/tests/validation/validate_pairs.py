import os
import shutil
import warnings

import hoomd

from msibi import MSIBI, Angle, Bond, Pair, State

warnings.filterwarnings("ignore")


def harmonic(x, k, x0):
    return 0.5 * k * (x - x0) ** 2


optimizer = MSIBI(
    nlist=hoomd.md.nlist.Cell,
    integrator_method=hoomd.md.methods.ConstantVolume,
    thermostat=hoomd.md.methods.thermostats.MTTK,
    thermostat_kwargs={"tau": 0.1},
    method_kwargs={},
    dt=0.0001,
    gsd_period=5000,
)

kT = 3.0
state = State(
    name=f"{kT}kT",
    kT=kT,
    traj_file=f"target_data/chains/{kT}kT.gsd",
    n_frames=100,
    sampling_stride=2,
    alpha0=0.7,
    exclude_bonded=True,
)

bond = Bond(type1="A", type2="A", optimize=False)
bond.set_harmonic(k=500, r0=1.1)
angle = Angle(type1="A", type2="A", type3="A", optimize=False)
angle.set_from_file(file_path="AAA_angle_potential.csv")

pair = Pair(
    type1="A",
    type2="A",
    optimize=True,
    r_cut=2.5,
    nbins=80,
    smoothing_window=15,
    r_switch=1.5,
    correction_fit_window=7,
)
pair.set_lj(r_min=0.1, r_cut=2.5, epsilon=0.8, sigma=1.2)

optimizer.add_state(state)
optimizer.add_force(bond)
optimizer.add_force(angle)
optimizer.add_force(pair)

optimizer.run_optimization(n_iterations=8, n_steps=1e6, backup_trajectories=False)
pair.smooth_potential()

scores = pair._states[state]["f_fit"]
assert scores[-1] >= 0.98

print("Finished")
print(f"Fit score = {scores[-1]}")

# Clean-up
shutil.rmtree("states")
os.remove("AAA_angle_potential.csv")
