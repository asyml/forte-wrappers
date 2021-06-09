import sys

import setuptools

if sys.version_info < (3, 6):
    sys.exit('Python>=3.6 is required by Forte.')

setuptools.setup(
    name="forte.huggingface",
    version="0.0.1",
    url="https://github.com/asyml/forte_wrappers/huggingface",

    description="Provide Forte implementations of a fantastic collection of "
                "NLP tools.",
    license='Apache License Version 2.0',
    packages=["forte.huggingface"],
    include_package_data=True,
    platforms='any',

    install_requires=[
        'forte @ git+https://git@github.com/asyml/forte.git',
        'more-itertools>=8.0.0',
        'transformers >= 3.1',
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
