#
# NEURON Dockerfile
#

# Pull base image.
FROM andrewosh/binder-python-3.5-mini
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

RUN conda install mpich2
RUN pip install numpy scipy matplotlib mpi4py h5py cython

WORKDIR nrn-7.5

# Compile NEURON.
RUN ./configure --prefix=`pwd` --without-iv --with-nrnpython && \
  make && \
  make install

# Install python interface
WORKDIR src/nrnpython
RUN python setup.py install

# Add NEURON to path
ENV PATH $HOME/neuron/nrn-7.5/x86_64/bin:$PATH


RUN git clone https://github.com/LFPy/LFPy.git
WORKDIR LFPy
RUN python setup.py install
ENV PYTHONPATH $PYTHONPATH:$HOME/LFPy/

# Switch back to non-root user privledges
WORKDIR $HOME

RUN git clone https://github.com/torbjone/aPop.git
RUN pip install -e aPop


# Switch back to non-root user privledges
WORKDIR $HOME
USER main