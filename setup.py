import setuptools
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="DASCL", # Replace with your own username
    version="0.0.1",
    author="Patrick Y. Wu, Richard Bonneau, Joshua A. Tucker, and Jonathan Nagler",
    author_email="pyw230@nyu.edu",
    description="dictionary-assisted supervised contrastive learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SMAPPNYU/DASCL",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)