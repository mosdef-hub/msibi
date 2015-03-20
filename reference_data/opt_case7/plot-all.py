# say wuuuut
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from matplotlib import rc
import pdb
import math
rc('text.latex', preamble=r'\usepackage{cmbright}')
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('font', family='sans-serif')
#rc('text', usetex=True)
import sys

######################################################################
# first parse the command options to the types and step to be used
######################################################################
argc = len(sys.argv)

# make sure both types and the iteration are given
if argc < 4:
	print "Give: type1 type2 iteration [no_]anim"
	print "Passing no_anim will cause no animation to be made"
	exit(1)
mov = True
if argc == 5:
	if sys.argv[4] == 'no_anim':
		mov = False

# make sure t1 < t2
if int(sys.argv[1]) > int(sys.argv[2]):
	print "type1 must be <= type2"
	exit(1)
######################################################################


# define some stuff
# look at one pair at a time, so that the 4 states will be shown in a nice 2x2 grid
t1 = int(sys.argv[1])
t2 = int(sys.argv[2])
iter = int(sys.argv[3])
if mov == True:
	smov = "On"
elif mov == False:
	smov = "Off"
print "Showing results for:"
print "Type 1: %d" % t1
print "Type 2: %d" % t2
print "Iterations: %d" % iter
print "Animation: %s" % smov
states = [0, 1, 2]
epsilon = 1.0
sigma = 1.0
plot_pu = False
marker_skip = 1

def calc_LJ(x, eps, sig):
	try:
		y = 4*eps*((sig/x)**12-(sig/x)**6)
	except:
		y = 99999.0
	return y

def calc_spacing(axes):
	a = axes.get_xlim()[0]
	b = axes.get_xlim()[1]
	c = axes.get_ylim()[0]
	d = axes.get_ylim()[1]
	xpos = a+(b-a)/20.0
	ypos = d-(d-c)/20.0
	pos = []
	pos.append(xpos)
	pos.append(ypos)
	return pos

def calc_spacing_bot_r(axes):
	a = axes.get_xlim()[0]
	b = axes.get_xlim()[1]
	c = axes.get_ylim()[0]
	d = axes.get_ylim()[1]
	xpos = b-(b-a)/20.0
	ypos = c+(d-c)/20.0
	pos = []
	pos.append(xpos)
	pos.append(ypos)
	return pos

# make an array of the filenames to plot
#file = []
#for i in xrange(state,state+1):
#	file.append("rdfs/rdf."+str(iter)+".query"+str(i)+".t"+str(t1)+"t"+str(t2)+".txt")
#target = []
#for i in xrange(state,state+1):
#	target.append("rdfs/rdf.target"+str(i)+".t"+str(t1)+"t"+str(t2)+".txt")

# read files to plot
r, tgr0 = np.loadtxt("rdfs/rdf.target%d.t%dt%d.txt" % (states[0], t1, t2), unpack=True)
r, gr0 = np.loadtxt("rdfs/rdf.%d.query%d.t%dt%d.txt" % (iter, states[0], t1, t2), unpack=True)

r, tgr1 = np.loadtxt("rdfs/rdf.target%d.t%dt%d.txt" % (states[1], t1, t2), unpack=True)
r, gr1 = np.loadtxt("rdfs/rdf.%d.query%d.t%dt%d.txt" % (iter, states[1], t1, t2), unpack=True)

r, tgr2 = np.loadtxt("rdfs/rdf.target%d.t%dt%d.txt" % (states[2], t1, t2), unpack=True)
r, gr2 = np.loadtxt("rdfs/rdf.%d.query%d.t%dt%d.txt" % (iter, states[2], t1, t2), unpack=True)

rv, V0, F0 = np.loadtxt("potentials/pot_full.time%d.t%dt%d.txt" % (iter, t1, t2), unpack=True)
#rv, V2, F2 = np.loadtxt("potentials/pot.t1t1.txt", unpack=True)
#rv, V3, F3 = np.loadtxt("potentials/pot.t1t1.txt", unpack=True)
#V_true = calc_LJ(rv, epsilon, sigma)

# set up plots as a 2-d array
fig, axarr = plt.subplots(2,2)

# add data to plot
axarr[0,0].plot(r,tgr0,linewidth=2.0,color='k')
axarr[0,0].plot(r[::marker_skip],gr0[::marker_skip], 'o', markerfacecolor='none', markeredgecolor='k', markeredgewidth=1.0)

axarr[0,1].plot(r,tgr1,linewidth=2.0,color='k')
axarr[0,1].plot(r[::marker_skip],gr1[::marker_skip], 'o', markerfacecolor='none', markeredgecolor='r', markeredgewidth=1.0)

axarr[1,0].plot(r,tgr2,linewidth=2.0,color='k')
axarr[1,0].plot(r[::marker_skip],gr2[::marker_skip], 'o', markerfacecolor='none', markeredgecolor='b', markeredgewidth=1.0)

axarr[1,1].plot(rv, V0, '-x', label="Derived potential", linewidth=1.5, color='orange')
axarr[1,1].plot(rv, calc_LJ(rv, 1.0, 1.0), '-', mew=2.0, label="Actual potential", color='k')
#axarr[1,1].plot(rv, V3, label="State 2", linewidth=1.5, color='r')
if plot_pu == True:
	axarr[1,1].plot(rv, V_true, label="Pu et al. 2007", linewidth=2.0, color='k')

# axes labels
axarr[0,0].set_ylabel('g(r)', size='large')
axarr[1,0].set_ylabel('g(r)', size='large')
axarr[0,1].set_ylabel('g(r)', size='large')
axarr[1,0].set_xlabel(r'r($\sigma$)', size='large')
axarr[1,1].set_xlabel(r'r($\sigma$)', size='large')
axarr[1,1].set_ylabel(r'V(r)', size='large')
axarr[1,1].legend()
axarr[1,1].set_xlim(0.75,3)
axarr[1,1].set_ylim(-1.5, 5)
axarr[0,0].set_ylim(0, 3.5)
axarr[0,1].set_ylim(0, 3.0)
axarr[1,0].set_ylim(0, 3.0)

# add keys for label
axarr[0,0].text(0,0,'(a)',position=calc_spacing(axarr[0,0]), ha='left', va='top',size='large')
axarr[0,1].text(0,0,'(b)',position=calc_spacing(axarr[0,1]), ha='left', va='top',size='large')
axarr[1,0].text(0,0,'(c)',position=calc_spacing(axarr[1,0]), ha='left', va='top',size='large')
axarr[1,1].text(0,0,'(d)',position=calc_spacing(axarr[1,1]), ha='left', va='top',size='large')
axarr[1,1].text(1,-4,'After %d steps' % iter, ha='left', va='bottom', size='large')

#fig.suptitle('Single state optimization', size='xx-large')
plt.tight_layout()

plt.savefig('t%dt%d-%d.pdf' % (t1, t2, iter))
#plt.show()

######################################################################




######################################################################
fig2, axarr2 = plt.subplots(2,2)
fig2.subplots_adjust(wspace = 0.3)
ax00 = axarr2[0,0]
ax01 = axarr2[0,1]
ax10 = axarr2[1,0]
ax11 = axarr2[1,1]

ax00.set_ylabel('g(r)', size='large')
ax10.set_ylabel('g(r)', size='large')
ax01.set_ylabel('g(r)', size='large')
ax11.set_ylabel(r'V(r)', size='large')

ax10.set_xlabel(r'r($\sigma$)', size='large')
ax11.set_xlabel(r'r($\sigma$)', size='large')

ax00.set_xlim(0,5)
ax01.set_xlim(0,5)
ax10.set_xlim(0,5)
ax11.set_xlim(0.9,3)

ax00.set_ylim(0,3.5)
ax01.set_ylim(0,2.5)
ax10.set_ylim(0,2.5)
ax11.set_ylim(-1.5,3)

ax00.text(0,0,'(a)',position=calc_spacing(ax00), ha='left', va='top',size='large')
ax01.text(0,0,'(b)',position=calc_spacing(ax01), ha='left', va='top',size='large')
ax10.text(0,0,'(c)',position=calc_spacing(ax10), ha='left', va='top',size='large')
ax11.text(0,0,'(d)',position=calc_spacing(ax11), ha='left', va='top',size='large')

line00, = axarr2[0,0].plot([], [], 'o', markerfacecolor='none', markeredgecolor='b', markeredgewidth=1.5)
line01, = axarr2[0,1].plot([], [], 'o', markerfacecolor='none', markeredgecolor='orange', markeredgewidth=1.5)
line10, = axarr2[1,0].plot([], [], 'o', markerfacecolor='none', markeredgecolor='r', markeredgewidth=1.5)
line11, = axarr2[1,1].plot([], [], '-o', color='#FF00FF', label='Derived Potential', linewidth=1.5, markerfacecolor='none', markeredgecolor='#FF00FF', markeredgewidth=1.5)

r, tgr0 = np.loadtxt("rdfs/rdf.target%d.t%dt%d.txt" % (states[0], t1, t2), unpack=True)
r, tgr1 = np.loadtxt("rdfs/rdf.target%d.t%dt%d.txt" % (states[1], t1, t2), unpack=True)
r, tgr2 = np.loadtxt("rdfs/rdf.target%d.t%dt%d.txt" % (states[2], t1, t2), unpack=True)
ax00.plot(r, tgr0, color='k', linewidth=2.0)
ax01.plot(r, tgr1, color='k', linewidth=2.0)
ax10.plot(r, tgr2, color='k', linewidth=2.0)
ax11.plot(r, calc_LJ(r, epsilon, sigma), color='k', linewidth=2.0)

iter_label = ax00.text(2.5, 0.5, '', position=calc_spacing_bot_r(ax00), ha='right', va='bottom', size='large')
#ax11.legend()




def animate(i):
	r, gr0 = np.loadtxt("rdfs/rdf.%d.query%d.t%dt%d.txt" % (i, states[0], t1, t2), unpack=True)
	r, gr1 = np.loadtxt("rdfs/rdf.%d.query%d.t%dt%d.txt" % (i, states[1], t1, t2), unpack=True)
	r, gr2 = np.loadtxt("rdfs/rdf.%d.query%d.t%dt%d.txt" % (i, states[2], t1, t2), unpack=True)
	rv, V0, F0 = np.loadtxt("potentials/pot_full.time%d.t%dt%d.txt" % (i, t1, t2), unpack=True)
	line00.set_data(r, gr0)
	line01.set_data(r, gr1)
	line10.set_data(r, gr2)
	line11.set_data(rv, V0)
	iter_label.set_text('%d Iterations' % i)
	return line00, line01, line10, line11, iter_label
	
if mov==True:
	anim = animation.FuncAnimation(fig2, animate, frames=iter+1, blit=True)
	anim.save('optimization-%d-%d.mp4' % (t1, t2), fps=10.0)
	
