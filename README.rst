==================================================
vlfeat-ctypes: minimal VLFeat interface for python
==================================================

This is a minimal port of a few components of the MATLAB interface of `the
vlfeat library <http://www.vlfeat.org>`_ for computer vision to Python. Vlfeat's
core library is written in C - in practice, though, a significant amount of the
library lies in the MATLAB interface. This project is a port of a few functions
from that interface to python/numpy, using ctypes.

This fork is based on `Dougal J. Sutherland's version
<https://github.com/dougalsutherland/vlfeat-ctypes>`_. At the moment, the port
uses vlfeat 0.9.20 and supports the following functions:

* ``vl_dsift``
* ``vl_fisher``
* ``vl_gmm``
* ``vl_imsmooth``
* ``vl_kmeans``
* ``vl_phow``

Installation
------------

The package does not require any installation. Just check out the source tree,
run ``make`` to download required dependencies, and add the directory where
this README is located to your ``PYTHONPATH``. Optionally, you can also set the
``PYTHONPATH`` in your ``~/.bash_profile``.

::

    # Check out and switch to source directory
    git clone https://github.com/slackner/vlfeat-ctypes.git
    cd vlfeat-ctypes

    # Download required dependencies (vlfeat libraries)
    make

    # Add directory to PYTHONPATH
    export PYTHONPATH="/path/to/vlfeat-ctypes:$PYTHONPATH"
