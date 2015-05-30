import os
from pkg_resources import resource_filename


def get_fn(name):
    """Get the full path to one of the reference files shopped for testing

    This function is taken straight from MDTraj (see https://github.com/mdtraj/mdtraj).
    In the source distribution, these files are in ``msibi/testing/reference``,
    but on istallation, they're moved to somewhere in the user's python 
    site-packages directory.

    Parameters
    ----------
    name : str
        Name of the file ot load (with respecto to the reference/ directory).

    Examples
    ________
    >>> import mdtraj as md
    >>> t = md.load(get_fun('final.hoomdxml'))
    """

    fn = resource_filename('msibi', os.path.join('testing', 'reference', name))

    if not os.path.exists(fn):
        raise ValueError('Sorry! %s does not exist. If you just '
                'added it, you\'ll have to re install' % fn)

    return fn
