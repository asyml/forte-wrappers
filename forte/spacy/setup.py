import sys

import setuptools

if sys.version_info < (3, 6):
    sys.exit('Python>=3.6 is required by Forte.')

setuptools.setup(
    name="forte.spacy",
    version="0.0.1",
    url="https://github.com/asyml/forte_wrappers/spacy",

    description="Provide Forte implementations of a fantastic collection of "
                "NLP tools.",
    license='Apache License Version 2.0',
    packages=["forte.spacy"],
    include_package_data=True,
    platforms='any',

    install_requires=[
        'forte @ git+https://git@github.com/asyml/forte.git',
        'spacy>=2.3.0, <=2.3.5',
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
