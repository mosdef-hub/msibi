import os
import shutil
import warnings

import hoomd
import numpy as np
from scipy.optimize import curve_fit

from msibi import MSIBI, Angle, Bond, Pair, State

warnings.filterwarnings("ignore")
_dir = os.path.dirname(os.path.abspath(__file__))
single_chain_path = os.path.join(_dir, "target_data", "chains", "single-chain.gsd")

# Check if the states dir is still here from a previous run
if os.path.exists(os.path.join(_dir, "states")):
    shutil.rmtree(os.path.join(_dir, "states"))


def harmonic(x, k, x0):
    return 0.5 * k * (x - x0) ** 2


def validate_bonds():
    optimizer = MSIBI(
        nlist=hoomd.md.nlist.Cell,
        integrator_method=hoomd.md.methods.ConstantVolume,
        thermostat=hoomd.md.methods.thermostats.MTTK,
        thermostat_kwargs={"tau": 0.1},
        method_kwargs={},
        dt=0.0001,
        gsd_period=500,
    )

    state = State(
        name="single-chain",
        kT=4.0,
        traj_file=single_chain_path,
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

    optimizer.run_optimization(n_iterations=12, n_steps=5e5, backup_trajectories=False)
    bond.smooth_potential()

    scores = bond._states[state]["f_fit"]
    assert any(np.array(scores) > 0.97)

    # Target data simulations used k = 500 and x0 = 1.1
    params, params_covariance = curve_fit(harmonic, bond.x_range, bond.potential)
    k_fit, x0_fit = params
    assert np.allclose(k_fit, 500, atol=20)
    assert np.allclose(x0_fit, 1.1, atol=0.05)

    print("Finished validating bonds.")
    print(f"Fit score = {scores[-1]}, k fit = {k_fit}, x0 fit = {x0_fit}")

    # Clean-up
    shutil.rmtree(os.path.join(_dir, "states"))


def validate_angles():
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
        traj_file=single_chain_path,
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

    optimizer.run_optimization(n_iterations=12, n_steps=5e5, backup_trajectories=False)
    angle.smooth_potential()
    angle.save_potential("AAA_angle_potential.csv")

    scores = angle._states[state]["f_fit"]
    assert any(np.array(scores) > 0.97)

    # Target data simulations used k = 250 and x0 = 2.0
    # Try to fit in range of potnetial that used IBI, instead of including large head/tail correction regions
    indices = np.where((angle.x_range <= 2.5) & (angle.x_range >= 1.5))
    params, params_covariance = curve_fit(
        harmonic, angle.x_range[indices], angle.potential[indices], p0=[250, 2.0]
    )
    k_fit, x0_fit = params

    print("Finished")
    print(f"Fit score = {scores[-1]}, k fit = {k_fit}, x0 fit = {x0_fit}")
    assert np.allclose(k_fit, 250, atol=26)
    assert np.allclose(x0_fit, 2.0, atol=0.05)

    # Clean-up
    shutil.rmtree(os.path.join(_dir, "states"))


def validate_pairs():
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
        traj_file=os.path.join(_dir, "target_data", "lj-fluid", f"{kT}kT.gsd"),
        n_frames=100,
        sampling_stride=2,
        alpha0=0.7,
        exclude_bonded=True,
    )

    bond = Bond(type1="A", type2="A", optimize=False)
    bond.set_harmonic(k=500, r0=1.1)
    angle = Angle(type1="A", type2="A", type3="A", optimize=False)
    angle.set_from_file(file_path=os.path.join(_dir, "AAA_angle_potential.csv"))

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

    optimizer.run_optimization(n_iterations=12, n_steps=1e6, backup_trajectories=False)
    pair.smooth_potential()

    scores = pair._states[state]["f_fit"]
    assert any(np.array(scores) > 0.97)

    print("Finished validating pairs.")
    print(f"Fit score = {scores[-1]}")

    # Clean-up
    shutil.rmtree("states")
    os.remove(os.path.join(_dir, "AAA_angle_potential.csv"))


if __name__ == "__main__":
    validate_bonds()
    validate_angles()
    validate_pairs()
