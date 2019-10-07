import os
from os import path
from setuptools import setup

this_directory = path.abspath(path.dirname(__file__))

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# read the contents of requirements.txt
with open(path.join(this_directory, 'requirements.txt'),
          encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name = "autousp",
    version = "0.0.1",
    description = ("A python framework for unsupervised learning with state of art models and classifical machine learning algorithms."),
    keywords = "Unsupervised learning,outlier detection, anomaly detection,Clutering",
    install_requires=requirements,
    long_description=read('README'),
)


