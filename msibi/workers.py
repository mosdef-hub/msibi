from __future__ import print_function

from distutils.spawn import find_executable
from math import ceil
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool
import os
from subprocess import Popen

from msibi.utils.general import backup_file
from msibi.utils.exceptions import UnsupportedEngine

N_PROCS = 0
USE_GPU = True


def run_query_simulations(states, engine='hoomd'):
    """Run all query simulations for a single iteration. """
    # TODO: GPU count and proper "cluster management"
    n_gpus = _get_gpu_info()

    global N_PROCS
    if not n_gpus:
        N_PROCS = cpu_count()
        global USE_GPU
        USE_GPU = False
        print("Launching {0:d} CPU threads...".format(cpu_count()))
    else:
        N_PROCS = n_gpus
        print("Launching {0:d} GPU threads...".format(n_gpus))
    pool = Pool(N_PROCS)

    if engine.lower() == 'hoomd':
        worker = _hoomd_worker
    else:
        raise UnsupportedEngine(engine)

    pool.imap(worker, zip(states, range(len(states))), ceil(len(states) / N_PROCS))
    pool.close()
    pool.join()


def _hoomd_worker(args):
    """Worker for managing a single HOOMD-blue simulation. """
    state = args[0]
    idx = args[1]
    log_file = os.path.join(state.state_dir, 'log.txt')
    err_file = os.path.join(state.state_dir, 'err.txt')
    with open(log_file, 'w') as log, open(err_file, 'w') as err:
        if USE_GPU:
            print('Running state {0} on GPU {1:d}'.format(state.name, idx % N_PROCS))
            card = idx % N_PROCS
            proc = Popen(['hoomd', 'run.py', '--gpu=%d' % (card)],
                         cwd=state.state_dir, stdout=log, stderr=err,
                         universal_newlines=True)
        else:
            proc = Popen(['hoomd', 'run.py'],
                         cwd=state.state_dir, stdout=log, stderr=err,
                         universal_newlines=True)

        print("    Launched HOOMD in {0}...".format(state.state_dir))
        proc.communicate()
        print("    Finished in {0}.".format(state.state_dir))
    _post_query(state)


def _post_query(state):
    """Reload the query trajectory and make backups. """
    state.reload_query_trajectory()
    backup_file(os.path.join(state.state_dir, 'log.txt'))
    backup_file(os.path.join(state.state_dir, 'err.txt'))
    if state.backup_trajectory:
        backup_file(state.traj_path)


def _get_gpu_info():
    nvidia_settings = find_executable('nvidia-settings')
    if not nvidia_settings:
        return
    else:
        print('FIGURE OUT GPU NUMBERS')

