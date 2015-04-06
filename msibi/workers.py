import math
import multiprocessing as mp
from multiprocessing.dummy import Pool
import os
from subprocess import Popen

from msibi.utils.general import backup_file
from msibi.utils.exceptions import UnsupportedEngine


HARDCODE_N_GPUS = 4
def run_query_simulations(states, engine='hoomd'):
    """Run all query simulations for a single iteration. """
    # TODO: GPU count and proper "cluster management"
    pool = Pool(HARDCODE_N_GPUS)   # should be max number concurrent simulations
    print('Changed pool to Pool(4) for ACCRE. See https://github.com/ctk3b/msibi/issues/5')
    print("Launching {0:d} threads...".format(HARDCOE_N_GPUS))
    if engine.lower() == 'hoomd':
        worker = _hoomd_worker
    else:
        raise UnsupportedEngine(engine)
    pool.imap(worker, zip(states, range(len(states))), len(states) // HARDCODE_N_GPUS)
    print('Also added chunksize into imap with hard-coded values')
    pool.close()
    pool.join()


def _hoomd_worker(state):
    """Worker for managing a single HOOMD-blue simulation. """
    idx = state[1]
    state = state[0]  # so i don't have to rename below
    log_file = os.path.join(state.state_dir, 'log.txt')
    err_file = os.path.join(state.state_dir, 'err.txt')
    with open(log_file, 'w') as log, open(err_file, 'w') as err:
        print('running state %s on gpu %d' % (state.name, idx % HARDCODE_N_GPUS))
        card = idx % HARDCODE_N_GPUS
        proc = Popen(['hoomd', 'run.py', '--gpu=%d' % (card)],
                cwd=state.state_dir, stdout=log, stderr=err,
                universal_newlines=True)
        print("    Launched HOOMD in {0}...".format(state.state_dir))
        proc.communicate()
        print("    Finished in {0}.".format(state.state_dir))
    _post_query(state)


def _post_query(state):
    state.reload_query_trajectory()
    backup_file(os.path.join(state.state_dir, 'log.txt'))
    backup_file(os.path.join(state.state_dir, 'err.txt'))
    if state.backup_trajectory:
        backup_file(state.traj_path)

