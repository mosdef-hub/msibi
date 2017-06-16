from msibi_utils.plot_fit import plot_all_fits
from msibi_utils.plot_rdfs import plot_all_rdfs
from msibi_utils.animate_rdf import animate_all_pairs_states

plot_all_fits('opt.out', ylims=(0.5, 1))
plot_all_rdfs('opt.out', 'rdfs', step=4)
animate_all_pairs_states('opt.out', 'rdfs', step=4, n_skip=0)
