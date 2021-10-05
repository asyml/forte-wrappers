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

supported_spacey_version = ">=2.3.0, <=2.3.5"

setuptools.setup(
    name="forte.spacy",
    version="0.0.2",
    url="https://github.com/asyml/forte-wrappers/tree/main/src/spacy",
    description="Provide Forte implementations of a fantastic collection of "
    "NLP tools.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache License Version 2.0",
    packages=["fortex.spacy"],
    namespace_packages=["fortex"],
    include_package_data=True,
    platforms="any",
    install_requires=[
        "forte==0.1.2",
        f"spacy{supported_spacey_version}",
        "scispacy==0.3.0",
    ],
    extra_requires={
        "cuda": [f"spacy[cuda]{supported_spacey_version}"],
        "cuda80": [f"spacy[cuda80]{supported_spacey_version}"],
        "cuda90": [f"spacy[cuda90]{supported_spacey_version}"],
        "cuda91": [f"spacy[cuda91]{supported_spacey_version}"],
        "cuda92": [f"spacy[cuda92]{supported_spacey_version}"],
        "cuda100": [f"spacy[cuda100]{supported_spacey_version}"],
        "cuda101": [f"spacy[cuda101]{supported_spacey_version}"],
        "cuda102": [f"spacy[cuda102]{supported_spacey_version}"],
        "cuda110": [f"spacy[cuda110]{supported_spacey_version}"],
        "cuda111": [f"spacy[cuda111]{supported_spacey_version}"],
        "cuda112": [f"spacy[cuda112]{supported_spacey_version}"],
    },
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
