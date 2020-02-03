from setuptools import setup

setup(
    name="toqito",
    description="Python module for quantum information.",
    long_description=open("README.md").read(),
    url="https://github.com/vprusso/toqito",

    author="Vincent Russo",
    author_email="vincentrusso1@gmail.com",

    packages=["toqito"],
    version="0.1.0",

    license="MIT",

    keywords=["toqito", "quantum information", "quantum computing"],

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.5",
    ],

    project_urls={
        "Homepage": "http://vprusso.github.io/",
    }
)
