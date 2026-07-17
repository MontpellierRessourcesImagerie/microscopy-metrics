.. _getting_started:

===============
Getting Started
===============

This guide will help you install and use **Microscopy Metrics**, a Python library for **quality control of Point Spread Functions (PSF)** in microscopy.
The project includes both a **standalone Python library** and a **Napari plugin** for interactive visualization.

---

.. _installation:

Installation
------------

**Microscopy Metrics** requires **Python 3.11**.
It is **strongly recommended** to use a **virtual environment** to avoid conflicts with other packages.

.. tabs::

   .. tab:: Using venv

      **1. Create a virtual environment:**

      .. code-block:: bash

         python -m venv microscopy_metrics_env

      **2. Activate the virtual environment:**

      - **On Windows:**

        .. code-block:: bash

           microscopy_metrics_env\Scripts\activate

      - **On macOS and Linux:**

        .. code-block:: bash

           source microscopy_metrics_env/bin/activate

      **3. Install the package and its dependencies:**

      .. code-block:: bash

         pip install --upgrade pip
         pip install microscopy-metrics

   .. tab:: Using conda (Recommended)

      **1. Create a conda environment:**

      .. code-block:: bash

         conda create --name microscopy_metrics_env python==3.11

      **2. Activate the conda environment:**

      .. code-block:: bash

         conda activate microscopy_metrics_env

      **3. Install the package:**

      .. code-block:: bash

         pip install microscopy-metrics

---

.. _install_from_source:

Install from Source
--------------------

If you want to **contribute to the project** or use the latest development version, you can install **Microscopy Metrics** directly from the GitHub repository:

.. code-block:: bash

   git clone https://github.com/MontpellierRessourcesImagerie/microscopy-metrics.git
   cd microscopy-metrics
   pip install -e .

.. note::
   The ``-e`` flag installs the package in **editable mode**, allowing you to modify the code and see changes immediately.

---

.. _dependencies:

Dependencies
------------

**Microscopy Metrics** relies on the following **core dependencies** (automatically installed via ``pip``):

.. list-table:: Core Dependencies
   :header-rows: 1
   :widths: 30 70

   * - **Package**
     - **Purpose**
   * - ``numpy``
     - Numerical computations and array operations.
   * - ``scipy``
     - Scientific computing (e.g., optimization, signal processing).
   * - ``scikit-image``
     - Image processing (e.g., filtering, segmentation).
   * - ``scikit-learn``
     - Machine learning tools for data analysis.
   * - ``matplotlib``
     - Plotting and visualization.
   * - ``psfmodels``
     - PSF (Point Spread Function) generation.
   * - ``tqdm``
     - Progress bars for long-running tasks.
   * - ``jinja2``
     - Template engine for report generation.
   * - ``Pillow``
     - Image handling (e.g., saving plots as images).
   * - ``simpleitk``
     - Image registration and processing.
   * - ``open3d``
     - 3D data visualization.
   * - ``skan``
     - Skeleton analysis for 3D structures.
   * - ``pyvista``
     - 3D plotting and mesh visualization.
   * - ``reportlab``
     - PDF report generation.

---

.. _optional_dependencies:

Optional Dependencies
---------------------

For **development** or **additional features**, you can install the following optional dependencies:

.. code-block:: bash

   pip install microscopy-metrics[dev]

This will install:

.. list-table:: Optional Dependencies
   :header-rows: 1
   :widths: 30 70

   * - **Package**
     - **Purpose**
   * - ``pytest``
     - Testing framework.
   * - ``pytest-cov``
     - Coverage reporting for tests.
   * - ``coverage``
     - Code coverage analysis.
   * - ``sphinx``
     - Documentation generation.
   * - ``sphinx-rtd-theme``
     - Theme for Sphinx documentation.
   * - ``sphinx-tabs``
     - Tab support in Sphinx documentation.

---

.. _run:

Running the Project
-------------------

There are **two ways** to use **Microscopy Metrics**:

.. tabs::

   .. tab:: Using the Napari Plugin (Recommended)

      The **Napari plugin** provides an **interactive interface** for visualizing and analyzing microscopy images.

      **1. Install the Napari plugin:**

      .. code-block:: bash

         pip install git+https://github.com/MontpellierRessourcesImagerie/napari-microscopy-metrics.git

      **2. Launch Napari:**

      .. code-block:: bash

         napari

      **3. Use the plugin:**
         - Open your microscopy image in Napari.
         - Use the **Microscopy Metrics** plugin from the **Plugins** menu to analyze PSFs, detect beads, and generate reports.

      .. note::
         Napari provides a **user-friendly interface** for exploring and analyzing your data interactively.

   .. tab:: Using the Python Library

      You can also use **Microscopy Metrics** as a **standalone Python library** for programmatic analysis.
      See the :doc:`how_to` guide for detailed usage examples.

      **Example: Generate a PSF and detect beads**

      .. code-block:: python

         from microscopy_metrics.scripts.PSFGenerator.PSF import PSFRandomParameter
         from microscopy_metrics.detection import Detection

         # Generate a PSF with comatic aberration
         psf = PSFRandomParameter(aberrationType="comatic")

         # Detect beads in the PSF
         detector = Detection()
         detector.image = psf.psf
         # Configure and run detection (see :doc:`how_to` for details)
         # ...

---

.. _scripts:

Using Scripts
-------------

**Microscopy Metrics** includes **pre-built scripts** in the ``src/microscopy_metrics/scripts/`` directory for common tasks such as:
    - Evaluating fittings.
    - Computing metrics.
    - Generating reports.

**Example: Run a fitting evaluation script**

.. code-block:: bash

   python src/microscopy_metrics/scripts/evaluate_fitting.py

---

.. _contributing:

Contributing
------------

We welcome contributions! Here’s how you can help:

1. **Fork the repository** on GitHub.
2. **Create a new branch** for your feature or bug fix:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

3. **Commit your changes**:

   .. code-block:: bash

      git commit -m "Add your feature or fix"

4. **Push to your branch**:

   .. code-block:: bash

      git push origin feature/your-feature-name

5. **Open a Pull Request** on GitHub.

For bug reports or feature requests, please open an **issue** on the `GitHub repository <https://github.com/MontpellierRessourcesImagerie/microscopy-metrics/issues>`_.

---

.. _support:

Support
-------

If you encounter any issues or have questions, feel free to:

- Open an **issue** on `GitHub <https://github.com/MontpellierRessourcesImagerie/microscopy-metrics/issues>`_.
- Contact the development team at **mri-cia@mri.cnrs.fr**.

---

.. _license_page:

License
-------

This project is licensed under the **CECILL-B** license. See the `LICENSE <https://cecill.info/licences/Licence_CeCILL-B_V1-fr.html>`_ file for details.