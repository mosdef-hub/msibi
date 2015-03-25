from setuptools import setup
import sys

try:
    import mdtraj
except ImportError:
    print('Building and running msibi requires mdtraj. See '
          'http://mdtraj.org/latest/installation.html for help!', file=sys.stderr)
    sys.exit(1)

requirements = [line.strip() for line in open('requirements.txt').readlines()]


setup(name='msibi',
      version='0.1',
      description='',
      url='http://github.com/ctk3b/misibi',
      author='Christoph Klein, Timothy C. Moore',
      author_email='christoph.klein@vanderbilt.edu, timothy.c.moore@vanderbilt.edu',
      license='MIT',
      packages=['msibi'],
      install_requires=requirements,
      zip_safe=False)