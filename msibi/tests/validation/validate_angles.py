import shutil
import warnings

import hoomd
import numpy as np
from scipy.optimize import curve_fit

from msibi import MSIBI, Angle, Bond, State

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

bond = Bond(type1="A", type2="A", optimize=False)
bond.set_harmonic(k=500, r0=1.1)

angle = Angle(
    type1="A",
    type2="A",
    type3="A",
    optimize=True,
    nbins=60,
    smoothing_window=5,
    correction_fit_window=7,
)
angle.set_polynomial(x_min=0.0, x_max=np.pi, x0=2.3, k2=80, k3=0, k4=0)

optimizer.add_state(state)
optimizer.add_force(bond)
optimizer.add_force(angle)

optimizer.run_optimization(n_iterations=8, n_steps=1e6, backup_trajectories=False)
angle.smooth_potential()
angle.save_potential("AAA_angle_potential.csv")

scores = angle._states[state]["f_fit"]
assert scores[-1] >= 0.96

# Target data simulations used k = 250 and x0 = 2.0
# Try to fit in range of potnetial that used IBI, instead of including large head/tail correction regions
indices = np.where((angle.x_range <= 2.5) & (angle.x_range >= 1.5))
params, params_covariance = curve_fit(
    harmonic, angle.x_range[indices], angle.potential[indices], p0=[250, 2.0]
)
k_fit, x0_fit = params

print("Finished")
print(f"Fit score = {scores[-1]}, k fit = {k_fit}, x0 fit = {x0_fit}")
assert np.allclose(k_fit, 250, atol=25)
assert np.allclose(x0_fit, 2.0, atol=0.05)

# Clean-up
shutil.rmtree("states")
