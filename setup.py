#!/usr/bin/env python

from distutils.core import setup

setup(name='aPop',
      version='1.0',
      description='Effect of Active conductances on the LFP',
      author='Torbjorn V. Ness',
      author_email='torbness@gmail.com',
      url='https://github.com/torbjone/aPop.git',
      packages=['aPop', 'aPop.neuron_models'],
     )