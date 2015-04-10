from __future__ import print_function, division

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

import logging
logging.basicConfig(level=logging.DEBUG,
        format='[%(levelname)s] %(message)s')


def run_query_simulations(states, engine='hoomd'):
    """Run all query simulations for a single iteration. """
    # TODO: GPU count and proper "cluster management"
    gpus = _get_gpu_info()

    global N_PROCS
    if gpus is None:
        N_PROCS = cpu_count()
        global USE_GPU
        USE_GPU = False
        gpus = []
        print("Launching {0:d} CPU threads...".format(cpu_count()))
    else:
        N_PROCS = len(gpus)
        print("Launching {0:d} GPU threads...".format(gpus))
    pool = Pool(N_PROCS)

    if engine.lower() == 'hoomd':
        worker = _hoomd_worker
    else:
        raise UnsupportedEngine(engine)

    L = len(states)
    print(ceil(L / N_PROCS))
    logging.debug('pool.imap')
    pool.imap(worker, zip(states, range(L), L * list(gpus)), ceil(L / N_PROCS))
    logging.debug(L * list(gpus))
    #pool.imap(worker, zip(states, range(len(states))), ceil(len(states) / N_PROCS))
    pool.close()
    pool.join()


def _hoomd_worker(args):
    """Worker for managing a single HOOMD-blue simulation. """
    state = args[0]
    idx = args[1]
    gpus = args[2][0]  # a list of the gpus available
    logging.debug(gpus)
    log_file = os.path.join(state.state_dir, 'log.txt')
    err_file = os.path.join(state.state_dir, 'err.txt')
    with open(log_file, 'w') as log, open(err_file, 'w') as err:
        if USE_GPU:
            card = gpus[idx % len(gpus)]
            print('Running state {0} on GPU {1:d}'.format(state.name, card_no))
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
    nvidia_smi = find_executable('nvidia-smi')
    if not nvidia_smi:
        return
    else:
        smi_out = os.popen('nvidia-smi').readlines()
        card_numbers = []
        for i, line in enumerate(smi_out[7:]):
            if not line.strip():
                break
            if i % 3 == 0:
                card_numbers.append(line.split()[1])
        return card_numbers
