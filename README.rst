.. image:: https://matthieumeo.github.io/pycsou/html/_images/pycsou.png
  :width: 50 %
  :align: center
  :target: https://github.com/matthieumeo/pycsou-gsp

.. image:: https://zenodo.org/badge/277582581.svg
   :target: https://zenodo.org/badge/latestdoi/277582581


*Pycsou-gsp* is the graph signal processing extension of the Python 3 package `Pycsou <https://github.com/matthieumeo/pycsou>`_ for solving linear inverse problems. The extension offers implementations of graph *convolution* and *differential* operators, compatible with Pycsou's interface for linear operators. Such tools can be useful when solving linear inverse problems involving signals defined on non Euclidean discrete manifolds.

Graphs in *Pycsou-gsp* are instances from the class ``pygsp.graphs.Graph`` from the `pygsp <https://github.com/epfl-lts2/pygsp>`_ library for graph signal processing with Python. 

Content
=======

The package, named `pycgsp <https://pypi.org/project/pycgsp>`_, is organised as follows:

1. The subpackage ``pycgsp.linop`` implements the following common graph linear operators:
  
   * Graph convolution operators: ``GraphConvolution``
   * Graph differential operators: ``GraphLaplacian``, ``GraphGradient``, ``GeneralisedGraphLaplacian``.

2. The subpackage ``pycgsp.tesselation`` provides routines for generating graphs from discrete tessellations of continuous manifolds such as the sphere. 
   
Installation
============

Pycsou-gsp requires Python 3.6 or greater. It is developed and tested on x86_64 systems running MacOS and Linux.


Dependencies
------------

Before installing Pycsou-gsp, make sure that the base package `Pycsou <https://github.com/matthieumeo/pycsou>`_ is correctly installed on your machine.
Installation instructions for Pycsou are available at `that link <https://matthieumeo.github.io/pycsou/html/general/install.html>`_.

The package extra dependencies are listed in the files ``requirements.txt`` and ``requirements-conda.txt``.
It is recommended to install those extra dependencies using `Miniconda <https://conda.io/miniconda.html>`_ or
`Anaconda <https://www.anaconda.com/download/#linux>`_. This
is not just a pure stylistic choice but comes with some *hidden* advantages, such as the linking to
``Intel MKL`` library (a highly optimized BLAS library created by Intel).

.. code-block:: bash

   >> conda install --channel=conda-forge --file=requirements-conda.txt


Quick Install
-------------

Pycsou-gsp is also available on `Pypi <https://pypi.org/project/pycsou-gsp/>`_. You can hence install it very simply via the command:

.. code-block:: bash

   >> pip install pycsou-gsp

If you have previously activated your conda environment ``pip`` will install Pycsou in said environment.
Otherwise it will install it in your ``base`` environment together with the various dependencies obtained from the file ``requirements.txt``.


Developer Install
------------------

It is also possible to install Pycsou-gsp from the source for developers:


.. code-block:: bash

   >> git clone https://github.com/matthieumeo/pycsou-gsp
   >> cd <repository_dir>/
   >> pip install -e .

The package documentation can be generated with:

.. code-block:: bash

   >> conda install sphinx=='2.1.*'            \
                    sphinx_rtd_theme=='0.4.*'
   >> python3 setup.py build_sphinx

You can verify that the installation was successful by running the package doctests:

.. code-block:: bash

   >> python3 test.py


Cite
====

For citing this package, please see: http://doi.org/10.5281/zenodo.4486431




