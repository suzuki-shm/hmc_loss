#!/usr/bin/env python
# vim:fileencoding=utf-8
#Author: Shinya Suzuki
#Created: 2016-06-23
try:
    import setuptools
    from setuptools import setup, find_packages
except ImportError:
    print("Please install setuptools.")
import os

if os.path.exists("README.txt"):
    long_description = open("README.txt").read()
else:
    long_description = "Validation metrics for hierarchical multi-label classification"

setup(
        name = "hmc_loss",
        packages = find_packages(),
        version = "0.3.1",
        url = "https://github.com/TaskeHAMANO/hmc_loss",
        discription = "Validation metrics for hierarchical multi-label classification",
        long_description = long_description,
        author = "Shinya SUZUKI",
        author_email = "shinya.s.825@gmail.com",
        keywords = ["validation", "metrics", "machine learning"],
        classifiers = [
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering :: Information Analysis"
            ],
        license="MIT",
        install_requires = ["networkx", "numpy"]
)
