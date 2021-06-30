import os
import sys
from pathlib import Path

import setuptools

long_description = (
        Path(os.path.realpath(__file__)).parent.parent.parent / "README.md").read_text()

if sys.version_info < (3, 6):
    sys.exit("Python>=3.6 is required by Forte.")

setuptools.setup(
    name="forte.nltk",
    version="0.0.1",
    url="https://github.com/asyml/forte_wrappers/nltk",
    description="Provide Forte implementations of a fantastic collection of "
                "NLP tools.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license="Apache License Version 2.0",
    packages=["forte.nltk"],
    include_package_data=True,
    platforms="any",
    install_requires=[
        "forte @ git+https://git@github.com/asyml/forte.git",
        "more-itertools>=8.0.0",
        "nltk==3.4.5",
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
