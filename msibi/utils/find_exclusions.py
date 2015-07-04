import networkx as nx
from networkx import NetworkXNoPath

def find_1_n_exclusions(top, pairs, n):
    """Find exclusions in a trajectory based on an exculsion principle

    Parameters
    ----------
    top : mdtraj.Topology
        Topology object containing the types and list of bonds
    pairs : array-like, shape=(n_pairs, 2), dtype=int
        Each row gives the indices of two atoms.
    n : int
        Exclude particles in pairs separated by n or fewer bonds
    """
    G = nx.Graph()
    G.add_nodes_from([a.index for a in top.atoms])
    bonds = [b for b in top.bonds]
    bonds_by_index = [(b[0].index, b[1].index) for b in bonds]
    G.add_edges_from(bonds_by_index)
    to_exclude = []
    for pair in pairs:
        if is_1_n(pair, n, G) == True:
            to_exclude.append(i)
    return np.asarray(to_exclude)


def is_1_n(pair, n, G):
    """Find if atoms in a pair are separated by n or less bonds

    Parameters
    ----------
    n : int
        Return false atoms in pair are separated by n or fewer bonds
    pair : [int, int]
        Pair of atom indices
    G : networkx.Graph
        A graph with atoms and nodes and bonds as edges

    Returns
    -------
    answer : bool
        answer == True if atoms are separated by n or fewer bonds, False otherwise

    The graph is expected to have atom indices as nodes, and tuples of atom indices as
    edges. Ideally, the nodes would be MDTraj.Atom-s and edges a list of tuples of
    MDTraj.Atom-s 
    try:
        return n > len(nx.shortest_path(G, pair[0], pair[1])) - 1
    except:  # no path between pair[0] and pair[1]
        return False
    """
    try:
        return n > len(nx.shortest_path(G, pair[0], pair[1])) - 1
    except NetworkXNoPath:
        return False
