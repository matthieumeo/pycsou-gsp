.. image:: /images/pycsou.png
  :width: 50 %
  :align: center
  :target: https://github.com/matthieumeo/pycsou-gsp

.. image:: https://zenodo.org/badge/277582581.svg
   :target: https://zenodo.org/badge/latestdoi/277582581


*Pycsou-gsp* is the graph signal processing extension of the Python 3 package `Pycsou <https://github.com/matthieumeo/pycsou>`_ for solving linear inverse problems. The extension offers implementations of graph *convolution* and *differential* operators, compatible with Pycsou's interface for linear operators. Such tools can be useful when solving linear inverse problems involving signals defined on non Euclidean discrete manifolds.

Graphs in *Pycsou-gsp* are instances from the class ``pygsp.graphs.Graph`` from the `pygsp <https://github.com/epfl-lts2/pygsp>`_ library for graph signal processing with Python. 

Content
-------

The package is organised as follows:

1. The subpackage ``pycgsp.linop`` implements the following common graph linear operators:
  
* Graph convolution operators (``pycgsp.linop.conv``):

.. autosummary::

    pycgsp.linop.conv.GraphConvolution

* Graph differential operators (``pycgsp.linop.diff``):

.. autosummary::

    pycgsp.linop.diff.GraphLaplacian
    pycgsp.linop.diff.GraphGradient
    pycgsp.linop.diff.GeneralisedGraphLaplacian

2. The module ``pycgsp.graph`` provides routines for generating graphs from discrete tessellations of continuous manifolds such as the sphere:

.. autosummary::

    pycgsp.graph.cvxhull_graph
    pycgsp.graph.healpix_nngraph

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   general/install

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference documentation

   api/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: More

   notes/index

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
