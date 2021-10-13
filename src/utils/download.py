import pathlib
import urllib.parse
import urllib.request
from typing import Union

import tqdm


class DownloadTqdm(tqdm.tqdm):
    def __init__(self, url: str):
        url = urllib.parse.urlparse(url)
        filename = extract_filename(url)
        description = f"downloading {filename} from {url.hostname}"
        super().__init__(unit="B", unit_scale=True, miniters=1, desc=description)

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)  # also sets self.n = b * bsize

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.total = self.n
        return super().__exit__(exc_type, exc_val, exc_tb)


def extract_filename(url: Union[str, urllib.parse.ParseResult]):
    if not isinstance(url, urllib.parse.ParseResult):
        url = urllib.parse.urlparse(url)
    return pathlib.PurePosixPath(url.path).name


def download(url: str, path=None) -> pathlib.Path:
    """
    Download the file at the given URL to the given local path.
    :param url: URL of the file to be downloaded.
    :param path: Local path to which to download the file. Default is cwd + filename.
    :return: The path to which the file was saved.
    """
    if path is None:
        filename = extract_filename(url)
        path = pathlib.Path.cwd().joinpath(filename)

    with DownloadTqdm(url) as progressbar:  # all optional kwargs
        urllib.request.urlretrieve(
            url, filename=path, reporthook=progressbar.update_to, data=None
        )
    return path


def ensure_downloaded(url: str, path=None) -> pathlib.Path:
    """
    Download a file if it doesn't exist, otherwise do nothing.
    :param url: URL of file to be downloaded.
    :param path: Local path to use as cache for the file.
    :return: The path of the file that was downloaded or that was previously cached.
    """
    path = (
        pathlib.Path(path)
        if path is not None
        else pathlib.Path.cwd().joinpath(extract_filename(url))
    )
    if path.is_file():
        return path
    else:
        return download(url, path)
