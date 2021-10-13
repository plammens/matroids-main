 # Dynamic Maximal Independent Set on a Matroid

This project aims to understand the complexity of finding a maximal independent set on a matroid when the underlying ground 
set E undergoes changes, i.e., elements are either inserted or deleted from the ground set. 
There are two main approaches to attack this problem. The first one is based on the idea of maintaining a random permutation of the ground set, while the second one runs the greedy algorithms from scratch and exploits the stability of matroids. 
The goal is to empirically evaluate the performance of these two approaches, and understand whether randomness helps in improving the running time for this problem.

## Repository setup

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
python scripts/some_script.py
```

If the above setup steps have been followed, this should run fine.


## Tests

Tests are written using the `pytest` framework.
To run automated tests, do:
```bash
pytest tests
```


