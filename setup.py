# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import os


def get_version():
    directory = os.path.abspath(os.path.dirname(__file__))
    init_file = os.path.join(directory, "evo", "__init__.py")
    with open(init_file) as f:
        for line in f:
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
        else:
            raise RuntimeError("Unable to find version string.")


with open("README.md", "r") as rf:
    README = rf.read()

with open("LICENSE", "r") as lf:
    LICENSE = lf.read()

with open("requirements.txt", "r") as reqs:
    requirements = reqs.read().split()
    for i, requirement in enumerate(requirements):
        if requirement.startswith("git+"):
            package_name = requirement.rsplit("=", maxsplit=1)[1]
            requirements[i] = f"{package_name} @ {requirement}"

setup(
    name="evo",
    packages=find_packages(),
    version=get_version(),
    description="Mono-repository of protein utilities",
    author="Roshan Rao",
    author_email="roshan_rao@berkeley.edu",
    url="https://github.com/rmrao/evo",
    license=LICENSE,
    keywords=["Proteins", "Deep Learning", "Pytorch"],
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
        "Topic :: Scientific/Engineering :: Bio-Informatics"
    ],
)
