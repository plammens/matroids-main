 # Dynamic Maximal Independent Set on a Matroid

This project aims to understand the complexity of finding a maximal independent set on a matroid when the underlying ground 
set E undergoes changes, i.e., elements are either inserted or deleted from the ground set. 

This repository provides a Python package named `matroids`, which contains 
a framework for working abstractly with matroids in Python 
and implementations of some static and dynamic maximal independent set algorithms. 


## Repository structure

The basic file structure is the following:

- `src/`: Source root for the main Python packages that provide the functionality of the project.
  - `matroids`: The main Python package exported by this project.
- `scripts/`: Contains individual Python scripts for e.g. performing execution time experiments.
  - `utils`: Python package containing miscellaneous utilities used by the scripts. The reason it's included under `scripts/` and not in `src/` is to avoid having to export it as a top-level package during installation, because Python automatically adds the `scripts/` directory to the `PYTHONPATH` when invoking a script like this: `python scripts/some_script.py`.
    - `utils.downloads`: I have taken this module, containing utilities to download files from URLs, from one of my past projects, https://github.com/plammens/alaquintavalavencida (this repository is private, contact me at 2475444L@student.gla.ac.uk or lammenspaolo@gmail.com to gain access).
- `tests/`: Source root for test modules to be run with `pytest`.
- `setup.py`: Contains project metadata such as dependencies and enables installation through `pip install`.

> **Note:** the package version is automatically inferred from Git tags, so do not delete the `.git` directory from the submission. 


## Setup

### Basic requirements

- **Python 3.9 or later** –
this repository has been written and tested in CPython 3.9.0
- **Python 3.9 development headers** - used for compiling C extensions of some of the
  dependencies.
  On Windows they are usually automatically installed by the Python 3.9 installer.
  On *nix, they can be installed with `apt` as follows:
  ```bash
  sudo apt install python3.9-dev
  ```
  See [here](https://stackoverflow.com/a/21530768/6117426) for instructions for other package managers.
- **OS**: Tested on Windows 10 and Ubuntu in WSL.


### [Optional] Step 0: Activate a virtual Python environment

It is strongly recommended that you activate an isolated Python environment
so that the packages you install for this project don't interfere with
your system interpreter.

For example, to create a Python 3.9 environment with [`virtualenv`](https://virtualenv.pypa.io/en/latest/):
```bash
virtualenv venv --python=py39
```
Then activate it according to [platform-specific instructions](https://virtualenv.pypa.io/en/latest/user_guide.html#activators).

For more detailed instructions, see [here](https://virtualenv.pypa.io/en/latest/user_guide.html).


### Step 1: Installation

To install the main packages, run the following from the repository root:
```bash
pip install -e .[all]
```
This automatically installs the required dependencies for the main packages.
Some of the dependencies require compiling C extensions, for which the corresponding Python development headers are needed.
If installation fails mentioning a missing "Python.h" header, this is because they aren't installed—see [basic requirements](#basic-requirements) for installation instructions.

The `[all]` syntax indicates to install all extra dependencies, namely for tests and scripts.
Alternatively one can specify, for example, `[tests,scripts]`, only `[tests]`, or nothing at all (`pip install -e .`), but then only the corresponding components will work.
(The lack of whitespace in this syntax is important.)

The `-e` flag is optional; it makes the installation editable:
i.e., the source files are used directly without first copying them to the default installation location.


## Tests

*Follow the [setup](#Setup) instructions first.*

Tests are written using the `pytest` framework.
To run automated tests, run the following from the repository root:
```bash
pytest tests
```

## Running scripts

*Follow the [setup](#setup) instructions first.*

To run a script in the `scripts` directory, run the following from the repository root:

```bash
python -O scripts/some_script.py
```

(Replace `python` with `python3` in *nix as appropriate.) 
The `-O` flag is to disable assertions (to get the most accurate performance measurements).

