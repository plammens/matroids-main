import itertools
import pathlib

from setuptools import find_packages, setup


REPO_ROOT = pathlib.Path(__file__).parent.resolve()


# Get the long description from the README file
long_description = (REPO_ROOT / "README.md").read_text(encoding="utf-8")


setup(
    setup_requires=["setuptools_scm"],
    #
    name="matroids",
    use_scm_version=True,
    description="Matroids and algorithms to compute a maximal independent set.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Paolo Lammens (2475444L)",
    author_email="2475444L@student.gla.ac.uk",
    project_urls={
        "Source": "https://stgit.dcs.gla.ac.uk/2475444l/l4-project-matroids-main",
    },
    #
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    #
    python_requires=">=3.9, <4",
    install_requires=[
        "numpy~=1.21.2",
        "llist==0.7.1",
        "more_itertools~=8.10.0",
    ],
    extras_require=(
        extras_require := {
            "test": [
                "pytest~=6.2.5",
                "pytest-cases==3.6.11",
            ],
            "scripts": [
                "matplotlib~=3.4.3",
                "tqdm~=4.63.0",
                "perfplot==0.9.15",
                "matplotx[all]==0.3.6",
            ],
        }
    )
    | {"all": list(itertools.chain.from_iterable(extras_require.values()))},
)
