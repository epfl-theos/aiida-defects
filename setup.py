from __future__ import absolute_import
import json
from setuptools import setup, find_packages

if __name__ == '__main__':
    with open('setup.json', 'r') as handle:
        kwargs = json.load(handle)
    setup(
        packages=find_packages(),
        **kwargs
)
