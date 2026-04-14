"""
Build and install the package.
"""
from setuptools import find_packages, setup

NAME = 'pyswapp'
FULLNAME = NAME
AUTHOR = "Nathalie Roser"
AUTHOR_EMAIL = 'nathalie.roser@geo.tuwien.ac.at'
LICENSE = "MIT License"
URL = ""
DESCRIPTION = ""
LONG_DESCRIPTION = DESCRIPTION

VERSION = '0.1'

PACKAGES = find_packages()
SCRIPTS = []

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Topic :: Scientific/Engineering",
    "Topic :: Software Development :: Libraries",
    "Programming Language :: Python :: 3.9",
    "License :: OSI Approved :: {}".format(LICENSE),
]
PLATFORMS = "Any"
INSTALL_REQUIRES = []

if __name__ == '__main__':
    setup(name=NAME, 
          fullname=FULLNAME, 
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION, 
          # ~ version=VERSION, 
          author=AUTHOR,
          author_email=AUTHOR_EMAIL, 
          license=LICENSE, url=URL,
          platforms=PLATFORMS, 
          scripts=SCRIPTS, 
          packages=PACKAGES,
          classifiers=CLASSIFIERS, 
          install_requires=INSTALL_REQUIRES,
          setuptools_git_versioning={"enabled": True},
          setup_requires=["setuptools-git-versioning"])
