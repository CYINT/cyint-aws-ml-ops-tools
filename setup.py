# _*_ coding: utf-8 _*_
from setuptools import find_packages
from distutils.core import setup

setup(
    name='cyint-aws-ml-ops-tools',
    version='1.0.0',
    author='Daniel Fredriksen',
    author_email='dfredriksen@cyint.technology',
    packages=find_packages(),
    url='https://github.com/CYINT/ml-ops-tools',
    license='MIT',
    description='Useful helper functions for preparing a modern ML Ops Pipeline'
)