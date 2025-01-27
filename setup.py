from setuptools import setup, find_packages

exec(open("ddesolver/version.py").read())  # loads __version__

setup(
    name="ddesolver",
    version=__version__,
    author="Reshief",
    author_email='reshief@gmx.net',
    description="Scipy-based Delayed Differential Equations (DDE) solver based on the DDEint implementation by Zulko",
    long_description=open("README.md").read(),
    license="see LICENSE.txt",
    keywords="delay delayed differential equation DDE",
    packages=find_packages(exclude="docs"),
    install_requires=["numpy", "scipy"],
)
