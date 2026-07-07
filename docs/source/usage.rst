Getting started
===============

.. _installation:
Installation
------------

To use `microscopy_metrics`, Python 3.12 is required.

It is strongly recommended to create a Python virtual environment to avoid conflicts with other packages.
You can choose between `venv` (built-in Python module) or `conda` (Anaconda/Miniconda) to create your virtual environment.

.. tabs::

    .. tab:: Using venv
    
        1. Create a virtual environment:

            .. code-block:: bash

                python3 -m venv microscopy_metrics_env
    
        2. Activate the virtual environment:

            - On Windows:

              .. code-block:: bash

                  microscopy_metrics_env\Scripts\activate

            - On macOS and Linux:

              .. code-block:: bash

                  source microscopy_metrics_env/bin/activate
    
        3. Install the package:

            .. code-block:: bash

                pip install microscopy_metrics
    
    .. tab:: Using conda
    
        1. Create a conda environment:

            .. code-block:: bash

                conda create --name microscopy_metrics_env python=3.12
    
        2. Activate the conda environment:

            .. code-block:: bash

                conda activate microscopy_metrics_env
    
        3. Install the package:

            .. code-block:: bash

                pip install microscopy_metrics

You can install `microscopy_metrics` directly from the GitHub repository using pip:

.. code-block:: bash

    pip install git+https://github.com/MontpellierRessourcesImagerie/microscopy-metrics.git

.. _run:

Running the project
-------------------

Microscopy metrics can be run using the napari plugin interface:

.. code-block:: bash
    
    pip install git+https://github.com/MontpellierRessourcesImagerie/napari-microscopy-metrics.git
    napari

This project can be used without a GUI interface, but it is recommended to use the napari plugin for a more user-friendly experience.

You can also use the different scripts provided in the `src/microscopy_metrics/scripts` directory to evaluate fittings and metrics computation.

