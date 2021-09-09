import os
from distutils.spawn import find_executable
import subprocess as sp

from msibi.utils.exceptions import UnsupportedEngine
from msibi.utils.general import backup_file


def run_query_simulations(states, engine="hoomd"):
    """Run all query simulations for a single iteration."""

    # Gather hardware info.
    gpu = _has_gpu()

    if engine.lower() == "hoomd":
        worker = _hoomd_worker
    else:
        raise UnsupportedEngine(engine)

    for state in states:
        _hoomd_worker(state, gpu=gpu)


def _hoomd_worker(state, gpu):
    """Worker for managing a single HOOMD-blue simulation."""

    log_file = os.path.join(state.dir, "log.txt")
    err_file = os.path.join(state.dir, "err.txt")

    executable = "python"
    with open(log_file, "w") as log, open(err_file, "w") as err:
        if gpu:
            print(f"Running state {state.name} on GPU")
            cmds = [executable, "run.py", "--mode=gpu"]
        else:
            print(f"Running state {state.name} on CPU")
            cmds = [executable, "run.py"]

        print(f"Launched HOOMD in {state.dir}")
        sp.run(
            cmds, cwd=state.dir, stdout=log, stderr=err,
            universal_newlines=True
        )
        print(f"Finished in {state.dir}.")
    _post_query(state)


def _post_query(state):
    """Reload the query trajectory and make backups."""
    backup_file(os.path.join(state.dir, "log.txt"))
    backup_file(os.path.join(state.dir, "err.txt"))
    if state.backup_trajectory:
        backup_file(state.traj_path)


def _has_gpu():
    nvidia_smi = find_executable("nvidia-smi")
    if not nvidia_smi:
        return False
    else:
        return True
