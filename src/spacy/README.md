<div align="center">
   <img src="https://raw.githubusercontent.com/asyml/forte/master/docs/_static/img/logo_h.png"><br><br>
</div>

-----------------

[comment]: <> ([![codecov]&#40;https://codecov.io/gh/asyml/forte/branch/master/graph/badge.svg&#41;]&#40;https://codecov.io/gh/asyml/forte&#41;)

[![Python Build](https://github.com/asyml/forte-wrappers/actions/workflows/main.yml/badge.svg)](https://github.com/asyml/forte-wrappers/actions/workflows/main.yml)
[![Documentation Status](https://readthedocs.org/projects/forte-wrappers/badge/?version=latest)](https://forte-wrappers.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/asyml/forte/blob/master/LICENSE)
[![Chat](http://img.shields.io/badge/gitter.im-asyml/forte-blue.svg)](https://gitter.im/asyml/community)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Forte** is a toolkit for building Natural Language Processing pipelines. This
repository tries to wrap the fantastic collections of NLP libraries built by the
community.

This project is part of the [CASL Open Source](http://casl-project.ai/) family.

### Get Started

- First, install the library along with SpaCy.

To install the PyPI repository version:
```shell
pip install spacy
```

You can also install GPU supports via:
```shell
pip install spacy[cuda]
```
See https://spacy.io/usage/#gpu for available cuda supports.

- NOTE: if the command does not install the GPU supported SpaCy, you can follow
the instructions here to install it manually: https://spacy.io/usage/#gpu

To install the latest version:
```shell
git clone https://github.com/asyml/forte-wrappers.git
cd forte-wrappers
pip install src/spacy
```


### License

This project is licensed by [Apache License 2.0](./LICENSE). The project provides wrappers to other open-sourced projects. To use
them in your project, please check the license of the corresponding project.

### Companies and Universities Supporting Forte

<p float="left">
   <img src="https://raw.githubusercontent.com/asyml/forte/master/docs/_static/img/Petuum.png" width="200" align="top">
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   <img src="https://asyml.io/assets/institutions/cmu.png", width="200" align="top">
</p>

