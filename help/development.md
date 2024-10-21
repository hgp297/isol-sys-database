# Development Guide

Python packages can be installed from a local source tree in two modes:

- In a **regular** install, files are copied from the local source tree into the Python installation directory. This is achieved by running
  ```shell
  python -m pip install .
  ```
  from the project root directory (ie, wherever the `setup.py` file is located).

- In an **editable** install, no files are copied and instead, all imports of the package are referred to the source tree from which they where installed. This is achieved by running
  ```shell
  python -m pip install -e .
  ```
  Notice the `-e` flag.

An **editable** install is almost always preferred to a **regular** install, as it facilitates active development of the project.
