import itertools
import os
from distutils.spawn import find_executable
from math import ceil
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool
from subprocess import Popen

from msibi.utils.exceptions import UnsupportedEngine
from msibi.utils.general import backup_file


def run_query_simulations(states, engine="hoomd"):
    """Run all query simulations for a single iteration. """

    # Gather hardware info.
    gpus = _get_gpu_info()
    if gpus is None:
        n_procs = cpu_count()
        gpus = []
        print("Launching {n_procs} CPU threads...".format(**locals()))
    else:
        n_procs = len(gpus)
        print("Launching {n_procs} GPU threads...".format(**locals()))

    if engine.lower() == "hoomd":
        worker = _hoomd_worker
    else:
        raise UnsupportedEngine(engine)

    n_states = len(states)
    worker_args = zip(states, range(n_states), itertools.repeat(gpus))
    chunk_size = ceil(n_states / n_procs)

    pool = Pool(n_procs)
    pool.imap(worker, worker_args, chunk_size)
    pool.close()
    pool.join()


def _hoomd_worker(args):
    """Worker for managing a single HOOMD-blue simulation. """

    state, idx, gpus = args
    log_file = os.path.join(state.dir, "log.txt")
    err_file = os.path.join(state.dir, "err.txt")

    executable = "python"
    with open(log_file, "w") as log, open(err_file, "w") as err:
        if gpus:
            card = gpus[idx % len(gpus)]
            cmds = [executable, "run.py", "--gpu={card}".format(**locals())]
        else:
            print("Running state {state.name} on CPU".format(**locals()))
            cmds = [executable, "run.py"]

        proc = Popen(
            cmds, cwd=state.dir, stdout=log, stderr=err,
            universal_newlines=True
        )
        print("Launched HOOMD in {state.state_dir}".format(**locals()))
        proc.communicate()
        print("Finished in {state.state_dir}.".format(**locals()))
    _post_query(state)


def _post_query(state):
    """Reload the query trajectory and make backups. """

    state.reload_query_trajectory()
    backup_file(os.path.join(state.state_dir, "log.txt"))
    backup_file(os.path.join(state.state_dir, "err.txt"))
    if state.backup_trajectory:
        backup_file(state.traj_path)


def _get_gpu_info():
    """ """
    nvidia_smi = find_executable("nvidia-smi")
    if not nvidia_smi:
        return
    else:
        gpus = [
            line.split()[1].replace(":", "")
            for line in os.popen("nvidia-smi -L").readlines()
        ]
        return gpus
