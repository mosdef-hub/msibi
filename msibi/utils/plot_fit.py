import os.path


def plot_pair_fits(pair, fits, use_agg=False):
    if use_agg:
        import matplotlib as mpl
        mpl.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for state, fit in fits[pair].iteritems():
        ax.plot(fit, label=state)
    ax.set_xlabel('step')
    ax.set_ylabel('relative fit')
    ax.legend(loc='best')
    ax.suptitle(pair)
    fig.tight_layout()
    fig.savefig('figures/%s-fit.pdf' % pair)
    plt.close('all')

def find_fits(filename): 
    # errs is a dict with keys 'type1-type2' for the pairs
    # the values are dicts, with the keys the state name and the values the 
    # list of fit values at that state
    fits = {}
    for line in open(filename, 'r'):
        try: 
            keyword = line.split()[1]
        except IndexError:
            pass
        if keyword == 'pair':
            try:
                fits[line.split()[2][:-1]][line.split()[4][:-1]].append(line.split()[-1])
            except KeyError:  # pair not in fits
                try:
                    fits[line.split()[2][:-1]][line.split()[4][:-1]] = [line.split()[-1]]
                except KeyError:  # state not in pairs in fits
                    fits[line.split()[2][:-1]] = {line.split()[4][:-1]: [line.split()[-1]]}
    return fits

def plot_all(filename, use_agg=False):
    """Plot fitness function vs. iteration for each pair at each state

    Args
    ----
    filename : str
        Name of file from which to read.
    use_agg : bool
        Use Agg backend if True - may be useful on clusters with no display

    Returns
    -------
    Nothing is returned, but plots are made for each pair.

    If the directory './figures' does not exist, it is created, and the figures 
    are saved in that directory with the name 'type1-type2-fit.pdf'.
    The filename should where the optimization output was redirected, as the 
    format is determined by the MSIBI.optimize() function.
    """
    fits = find_fits(filename)
    if not os.path.exists('figures'):
        os.makedirs('figures')
    for pair in fits:
        plot_pair_fits(pair, fits)
