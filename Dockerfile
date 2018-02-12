#
# NEURON Dockerfile
#

# Pull base image.
#FROM continuumio/anaconda3
FROM andrewosh/binder-python-3.5-mini
#FROM continuumio/miniconda3
MAINTAINER Tester <tester@test.no>

USER root

RUN \
  apt-get update && \
  apt-get install -y libncurses-dev

# Make ~/neuron directory to hold stuff.
WORKDIR neuron

# Fetch NEURON source files, extract them, delete .tar.gz file.
RUN \
  wget http://www.neuron.yale.edu/ftp/neuron/versions/v7.5/nrn-7.5.tar.gz && \
  tar -xzf nrn-7.5.tar.gz && \
  rm nrn-7.5.tar.gz

# Fetch Interviews.
# RUN \
#  wget http://www.neuron.yale.edu/ftp/neuron/versions/v7.4/iv-19.tar.gz  && \  
#  tar -xzf iv-19.tar.gz && \
#  rm iv-19.tar.gz

WORKDIR nrn-7.5

# Compile NEURON.
RUN \
  ./configure --prefix=`pwd` --without-iv --with-nrnpython=$HOME/anaconda3/bin/python && \
  make && \
  make install

# Install python interface
WORKDIR src/nrnpython
RUN python setup.py install

RUN git clone https://github.com/LFPy/LFPy.git
WORKDIR LFPy
RUN python setup.py install

# Install PyNeuron-Toolbox
# WORKDIR $HOME
# RUN git clone https://github.com/ahwillia/PyNeuron-Toolbox
# WORKDIR PyNeuron-Toolbox
# RUN python setup.py install

# Install JSAnimation
# WORKDIR $HOME
# RUN git clone https://github.com/jakevdp/JSAnimation.git
# RUN python JSAnimation/setup.py install

# Install other requirements
# RUN pip install palettable

# ENV PYTHONPATH $PYTHONPATH:$HOME/JSAnimation/:$HOME/PyNeuron-Toolbox/

# Add NEURON to path
# TODO: detect "x86_64" somehow?
ENV PATH $HOME/neuron/nrn-7.4/x86_64/bin:$PATH

# Switch back to non-root user privledges
WORKDIR $HOME
USER main
