import os
import sys
from distutils.core import setup
from setuptools import find_packages

setup(name='Zappy',
      version='0.1',
      description="Load Flow Electrical Analysis",
      long_description="""
            Zappy is a simple electrical load flow modeling library for both AC and DC electrical systems. 
            Zappy is built on top of the OpenMDAO framework, with the code relying on OpenMDAO for data passing, solvers and optimizers among other things. 
            The Zappy implementation includes analaytic derivatives for all components to enable efficient gradient-based optimization when included in larger MDAO problems.
            """,
      classifiers=[
        'Development Status :: 1 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache 2.0',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
      ],
      keywords='',
      author='Eric Hendricks',
      author_email='eric.hendricks@nasa.gov',
      license='Apache License, Version 2.0',
      packages=[ 
        'zappy/LF_elements',
        'zappy/LF_examples',
        'zappy/NV_elements',
        ],
      install_requires=[
        'openmdao',
        'numpy>=1.9.2',
        'scipy',
        'pep8',
        'parameterized',
      ],
)