import functools
import logging
import os
import pathlib
import string
import typing as tp
from typing import Callable, Sequence

import matplotlib.pyplot as plt


ROOT_OUTPUT_PATH = pathlib.Path(__file__).parent.parent.parent.resolve() / "artifacts"
VALID_CHARS = string.ascii_letters + string.digits + " "


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_output_dir(path: os.PathLike = ROOT_OUTPUT_PATH):
    path = os.path.normpath(path)  # avoid empty path at the end (foo/bar/)
    if not os.path.exists(path):
        ensure_output_dir(os.path.dirname(path))
        os.mkdir(path)
    elif not os.path.isdir(path):
        raise IOError("Output path {} already exists and is a file".format(path))
    return path


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
    name = "-".join(
        "".join(filter(VALID_CHARS.__contains__, x)) for x in components if x
    )
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
    ensure_output_dir(output_dir)
    save_func(str(path))


def save_figure(
    fig: plt.Figure,
    identifiers: tp.Sequence[str],
    output_dir: os.PathLike = ROOT_OUTPUT_PATH / "figures",
    extra_artists: tp.Collection[plt.Artist] = None,
):
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
