import sys
from pathlib import Path

import setuptools

readme = Path("README.md")
if readme.exists():
    long_description = (Path("README.md")).read_text()
else:
    long_description = (
        "Provide Forte implementations of a fantastic collection of NLP tools."
    )

if sys.version_info < (3, 6):
    sys.exit("Python>=3.6 is required by Forte.")

setuptools.setup(
    name="forte.spacy",
    version="0.0.1.post1",
    url="https://github.com/asyml/forte-wrappers/tree/main/src/spacy",
    description="Provide Forte implementations of a fantastic collection of "
    "NLP tools.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache License Version 2.0",
    packages=["forte.spacy"],
    include_package_data=True,
    platforms="any",
    install_requires=[
        "forte==0.1.1",
        "spacy>=2.3.0, <=2.3.5",
        "scispacy==0.3.0",
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
