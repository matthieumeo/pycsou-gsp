.. _installation:

Installation
============

Pycsou-gsp requires Python 3.6 or greater.It is developed and tested on x86_64 systems running MacOS and Linux.


Dependencies
------------
Before installing Pycsou-gsp, make sure that the base package Pycsou is correctly installed on your machine. Installation instructions for Pycsou are available at `that link <https://matthieumeo.github.io/pycsou/html/general/install.html>`_. 

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

If you have previously activated your conda environment ``pip`` will install Pycsou in said environment. Otherwise it will install it in your base environment together with the various dependencies obtained from the file ``requirements.txt``.


Developper Install
------------------

It is also possible to install Pycsou-gsp from the source for developpers: 


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
