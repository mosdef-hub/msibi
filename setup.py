from setuptools import setup
from setuptools.command.test import test as TestCommand
import sys

try:
    import mdtraj
except ImportError:
    print('Building and running msibi requires mdtraj. See '
          'http://mdtraj.org/latest/installation.html for help!')
    sys.exit(1)

requirements = [line.strip() for line in open('requirements.txt').readlines()]


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(['msibi'])
        sys.exit(errcode)


setup(name='msibi',
      version='0.1',
      description='',
      url='http://github.com/ctk3b/misibi',
      author='Christoph Klein, Timothy C. Moore', 'Davy Yue'
      author_email='christoph.klein@vanderbilt.edu, timothy.c.moore@vanderbilt.edu', 'davy.yue@vanderbilt.edu'
      license='MIT',
      packages=['msibi'],
      install_requires=requirements,
      zip_safe=False,
      test_suite='tests',
      cmdclass={'test': PyTest},
      extras_require={'utils': ['pytest']},
)
