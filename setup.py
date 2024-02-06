import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    "cvx",
    "cvxpy",
    "cvxopt",
    "more-itertools",
    "numpy",
    "picos",
    "scipy",
    "scikit-image",
    "pytest",
    "pytest-cov",
    "qiskit",
    "scs"
]

setuptools.setup(
    name="toqito",
    version="1.0.7",
    author="Vincent Russo",
    author_email="vincentrusso1@gmail.com",
    description="Python toolkit for quantum information theory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vprusso/toqito",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    test_suite="tests",
)
