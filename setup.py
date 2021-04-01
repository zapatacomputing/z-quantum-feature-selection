import setuptools
import os

setuptools.setup(
    name="z-quantum-feature-selection",
    version="0.1.0",
    author="Zapata Computing, Inc.",
    author_email="info@zapatacomputing.com",
    description="A library for performing feature selection with Orquestra.",
    url="https://github.com/zapatacomputing/z-quantum-feature-selection",
    packages=setuptools.find_namespace_packages(
        include=["zquantum.*"], where="src/python"
    ),
    package_dir={"": "src/python"},
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
    install_requires=["z-quantum-core", "z-quantum-qubo", "dimod==0.9.11", "cvxopt"],
)
