import os
import sys
from pathlib import Path

import setuptools

print(os.path.realpath(__file__))
print(os.path.abspath(__file__))

long_description = (
    Path(os.path.abspath(os.path.realpath(__file__))).parent.parent.parent
    / "README.md"
).read_text()

if sys.version_info < (3, 6):
    sys.exit("Python>=3.6 is required by Forte.")

setuptools.setup(
    name="forte.allennlp",
    version="0.0.1",
    url="https://github.com/asyml/forte_wrappers/allennlp",
    description="Provide Forte implementations of a fantastic collection of "
    "NLP tools.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache License Version 2.0",
    packages=["forte.allennlp"],
    include_package_data=True,
    platforms="any",
    install_requires=[
        "forte @ git+https://git@github.com/asyml/forte.git",
        "more-itertools>=8.0.0",
        "allennlp==1.2.0",
        "allennlp-models==1.2.0",
        "torch>=1.5.0",
        "pillow==8.2.0",
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
