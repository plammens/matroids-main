 # Dynamic Maximal Independent Set on a Matroid

This project aims to understand the complexity of finding a maximal independent set on a matroid when the underlying ground 
set E undergoes changes, i.e., elements are either inserted or deleted from the ground set. 
There are two main approaches to attack this problem. The first one is based on the idea of maintaining a random permutation of the ground set, while the second one runs the greedy algorithms from scratch and exploits the stability of matroids. 
The goal is to empirically evaluate the performance of these two approaches, and understand whether randomness helps in improving the running time for this problem.

## Repository structure

The basic file structure is the following:

- `src/`: Source root for the main Python packages that provide the functionality of the project (e.g. matroid classes, algorithms).
- `scripts/`: Contains individual Python scripts for e.g. performing execution time experiments.
  - `utils`: Python package containing miscellaneous utilities used by the scripts. The reason it's included under `scripts/` and not in `src/` is to avoid having to export it as a top-level package during installation, because Python automatically adds the `scripts/` directory to the `PYTHONPATH` when invoking a script like this: `python scripts/some_script.py`.
- `tests/`: Source root for test modules to be run with `pytest`.
- `setup.py`: Contains project metadata such as dependencies and enables installation through `pip install`.


## Setup

### Activate a virtual Python environment (optional)

It is strongly recommended that you activate an isolated Python environment
so that the dependencies you install for this project don't interfere with
your system interpreter.

For example, to create a [`virtualenv`](https://virtualenv.pypa.io/en/latest/):
```bash
virtualenv venv
```
Then activate it according to [platform-specific instructions](https://virtualenv.pypa.io/en/latest/user_guide.html#activators);
e.g. on bash:
```bash
source venv/bin/activate
```

### Install dependencies

To install the required Python packages for development, run
```bash
pip install -r requirements.txt
```


### PYTHONPATH

If running everything from source (i.e. without installing anything), the correct value of the PYTHONPATH environment variable should be set up first.
It suffices to prepend `src` to PYTHONPATH:

- On *nix:
    ```bash
    export PYTHONPATH=src:$PYTHONPATH
    ```
- On Windows:
    ```cmd
    set PYTHONPATH=src;%PYTHONPATH%
    ```
  

## Running scripts

To run a script in the `scripts` directory, do the following from the repository root:

```bash
python -O scripts/some_script.py
```

If the above setup steps have been followed, this should run fine.
The `-O` flag is to disable assertions (to get the most accurate performance measurements).


## Tests

Tests are written using the `pytest` framework.
To run automated tests, do:
```bash
pytest tests
```


