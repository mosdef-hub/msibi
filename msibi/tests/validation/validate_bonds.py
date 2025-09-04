import shutil
import warnings

import hoomd
import numpy as np
from scipy.optimize import curve_fit

from msibi import MSIBI, Bond, State

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

state = State(
    name="single-chain",
    kT=4.0,
    traj_file="target_data/chains/single-chain.gsd",
    n_frames=100,
    sampling_stride=2,
    alpha0=0.5,
)

bond = Bond(
    type1="A",
    type2="A",
    optimize=True,
    nbins=60,
    smoothing_window=5,
    correction_fit_window=7,
)
bond.set_polynomial(x_min=0.0, x_max=2.5, x0=1.3, k2=200, k3=0, k4=0)

optimizer.add_state(state)
optimizer.add_force(bond)

optimizer.run_optimization(n_iterations=7, n_steps=1e6, backup_trajectories=False)
bond.smooth_potential()

scores = bond._states[state]["f_fit"]
assert scores[-1] >= 0.97

# Target data simulations used k = 500 and x0 = 1.1
params, params_covariance = curve_fit(harmonic, bond.x_range, bond.potential)
k_fit, x0_fit = params
assert np.allclose(k_fit, 500, atol=10)
assert np.allclose(x0_fit, 1.1, atol=0.05)

print("Finished")
print(f"Fit score = {scores[-1]}, k fit = {k_fit}, x0 fit = {x0_fit}")

# Clean-up
shutil.rmtree("states")
