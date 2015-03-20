import mdtraj as md


class State(object):
    """A single state used as part of a multistate optimization.

    Attributes
    ----------
    traj : md.Trajectory
        The trajectory associated with this state.
    kT : float

    """
    def __init__(self, kT, traj_file, top_file=None):
        if top_file:
            self.traj = md.load(traj_file, topology=top_file)
        else:
            self.traj = md.load(traj_file)
        self.kT = kT

