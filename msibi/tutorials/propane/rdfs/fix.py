import glob
import os

for letter in ['A', 'B', 'C']:
    for file in glob.glob('*state{0}*'.format(letter)):
        newname = file.replace('state{0}'.format(letter), 'state_{0}'.format(letter))
        os.system('cp {0} {1}'.format(file, newname))	
