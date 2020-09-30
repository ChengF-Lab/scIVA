#!/usr/bin/env python
"""
# File Name: setup.py
# Description:

"""

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='scIVA-atac',
      version='1.0.2',
      description='Dimensionality reduction and visualization of single-cell RNA-seq data with an improved deep variational autoencoder',
      packages=find_packages(),

      author='junlin Xu',
      author_email='xjl@hnu.edu.cn.com',
      url='',
      scripts=['scIVA.py'],
      install_requires=requirements,
      python_requires='>3.6.0',

      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.7',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
     ],
     )
