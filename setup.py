#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
pygrid3d - a Python package for grid search of energy and origin time of source

:copyright:
    Wenjie Lei (lei@Princeton.EDU), 2015
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lgpl.html)
'''
from __future__ import print_function
import os
import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand


def read(fname):
    """
    Utility function to read the README file.
    Used for the long_description.  It's nice, because now 1) we have a top
    level README file and 2) it's easier to type in the README file than to
    put a raw string in below ...
    """
    try:
        return open(os.path.join(os.path.dirname(__file__), fname)).read()
    except Exception as err:
        return "Can't open %s as error:%s" % (fname, err)


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(
    name='pygrid3d',
    version='0.1.0',
    license='GNU Lesser General Public License, Version 3',
    description="software for 3 dimensional centroid moment tensor inversion",
    long_description=read("README.md"),
    author='Wenjie Lei',
    author_email='lei@princeton.edu',
    url='https://github.com/wjlei1990/pygrid3d',
    packages=find_packages("src"),
    package_dir={"": "src"},
    test_require=['pytest'],
    zip_safe=False,
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # The project is intended for
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        # Supproted python version
        'Programming Language :: Python :: 2.7',
    ],
    keywords=['seismology', 'grid3d', 'moment tensor',
              'centroid moment inversion'],

    # What does your project relate to?
    install_requires=[
        "obspy", "numpy", "future>=0.14.1", "matplotlib"
    ],
    extras_require={
        "docs": ["sphinx"]
    }
)
