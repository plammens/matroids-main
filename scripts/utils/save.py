import functools
import logging
import os
import pathlib
import string
import typing as tp
from typing import Callable, Sequence

import matplotlib.pyplot as plt

from utils.misc import ROOT_OUTPUT_PATH, ensure_output_dir_exists


VALID_CHARS = string.ascii_letters + string.digits + " "


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_output(
    save_func: Callable[[str], None],
    output_dir: os.PathLike,
    output_name: str,
    identifiers: Sequence[str],
    file_extension: str,
    include_output_name: bool = False,
) -> None:
    """Greatest common denominator for output saving functions"""
    output_dir = pathlib.Path(output_dir)

    components = list(identifiers)
    if include_output_name:
        components.insert(0, output_name)
    name = "-".join(components)
    file_extension = file_extension.lstrip(".")
    filename = "{}.{}".format(name, file_extension)
    path = output_dir / filename

    message = " ".join(
        [
            "Saving {}".format(output_name),
            "for {}".format(", ".join(identifiers)) if identifiers else "",
            "to {}".format(path),
        ]
    )
    logger.info(message)
    ensure_output_dir_exists(output_dir)
    save_func(str(path))


def save_figure(
    fig: plt.Figure,
    identifiers: tp.Sequence[str],
    output_dir: os.PathLike = ROOT_OUTPUT_PATH/"figures",
    extra_artists: tp.Collection[plt.Artist] = None,
) -> None:
    save_output(
        functools.partial(
            fig.savefig, bbox_extra_artists=extra_artists, bbox_inches="tight"
        ),
        output_dir,
        "figure",
        identifiers,
        "pdf",
    )
    plt.close(fig)
