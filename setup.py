from ez_setup import use_setuptools
use_setuptools()

from setuptools import setup, find_packages

exec(open("deesolver/version.py").read())  # loads __version__

setup(
    name="deesolver",
    version=__version__,
    author="Reshief",
    description="Scipy-based Delayed Differential Equations (DDE) solver based on the DDEint implementation by Zulko",
    long_description=open("README.md").read(),
    license="see LICENSE.txt",
    keywords="delay delayed differential equation DDE",
    packages=find_packages(exclude="docs"),
    install_requires=["numpy", "scipy"],
)
