import sys
import setuptools
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text()

if sys.version_info < (3, 6):
    sys.exit('Python>=3.6 is required by Forte.')

setuptools.setup(
    name="forte-wrappers",
    version="0.0.1",
    url="https://github.com/asyml/forte_wrappers",

    description="Provide Forte implementations of a fantastic collection of "
                "NLP tools.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='Apache License Version 2.0',
    packages=setuptools.find_namespace_packages(
        exclude=["tests*", "examples*", "docs*"]
    ),
    include_package_data=True,
    platforms='any',

    install_requires=[
        'forte @ git+https://git@github.com/asyml/forte.git',
        'more-itertools>=8.0.0'
    ],
    extras_require={
        'nltk': ['nltk==3.4.5'],
        'varder': ['vaderSentiment==3.2.1'],
        'stanza': ['stanza==1.0.1'],
        'elastic': ['elasticsearch==7.5.1'],
        'faiss': ['faiss-cpu>=1.6.1'],
        'spacy': ['spacy>=2.3.0, <=2.3.5'],  # Download breaks at 2.3.6
        'scispacy': ['scispacy==0.3.0'],
        'allennlp': ['allennlp==1.2.0', 'allennlp-models==1.2.0',
                     'torch>=1.5.0'],
        'cliner': ['python-crfsuite==0.9.7'],
        'gpt2-example': ['termcolor>=1.1.0'],
        'twitter': ['tweepy==3.10.0'],
        'huggingface': ['transformers >= 3.1']
    },
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
