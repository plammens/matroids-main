"""
Utilities to access the Stanford Large Network Dataset Collection [1].


References:
-----------

[1] J. Leskovec and A. Krevl.
    SNAP Datasets: Stanford Large Network Dataset Collection, June 2014.
    URL http://snap.stanford.edu/data .
    [Last accessed: 2022-03-26].
"""

import pathlib
import tarfile
import typing as tp

import networkx as nx

from utils.download import ensure_downloaded


def load_facebook_dataset() -> tp.List[nx.Graph]:
    """
    Download and load the Facebook dataset from the SLNDC.

    :returns: a list of networkx graph objects from the Facebook dataset.
    """

    path = ensure_downloaded(
        "https://snap.stanford.edu/data/facebook.tar.gz",
        path=pathlib.Path.cwd().joinpath("downloads").joinpath("facebook.tar.gz"),
    )
    tar = tarfile.open(path)
    filenames = [name for name in tar.getnames() if name.endswith("edges")]
    networks = []
    for name in filenames:
        with tar.extractfile(name) as file:
            edges = [tuple(map(int, line.split())) for line in file.readlines()]
            network = nx.from_edgelist(edges)
            networks.append(network)

    return networks
