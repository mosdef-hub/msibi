import glob
import multiprocessing as mp
from multiprocessing.dummy import Pool
import os
import shutil
from subprocess import Popen


from msibi.utils.general import *
from msibi.utils.general import _backup
from msibi.utils.exceptions import UnsupportedEngine


def run_query_simulations(states, engine='hoomd'):
    """Run all query simulations for a single iteration. """
    # TODO: GPU count and proper "cluster management"
    pool = Pool(mp.cpu_count())
    print("Launching {0:d} threads...".format(mp.cpu_count()))
    if engine.lower() == 'hoomd':
        worker = _hoomd_worker
    else:
        raise UnsupportedEngine(engine)
    pool.imap(worker, states)
    pool.close()
    pool.join()


def _hoomd_worker(state):
    """Worker for managing a single HOOMD-blue simulation. """
    log_file = os.path.join(state.state_dir, 'log.txt')
    err_file = os.path.join(state.state_dir, 'err.txt')
    with open(log_file, 'w') as log, open(err_file, 'w') as err:
        proc = Popen(['hoomd', 'run.py'], cwd=state.state_dir, stdout=log,
                     stderr=err, universal_newlines=True)
        print("    Launched HOOMD in {0}...".format(state.state_dir))
        proc.communicate()
        print("    Finished in {0}.".format(state.state_dir))
    _post_query(state)


def _post_query(state):
    state.reload_query_trajectory()
    _backup(os.path.join(state.state_dir, 'log.txt'))
    _backup(os.path.join(state.state_dir, 'err.txt'))
    if state.save_trajectory: # backup trajectory if True
        print('backing up in %s' % state.state_dir)
        _backup(state.traj_path)

