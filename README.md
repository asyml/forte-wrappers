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

- First, install the library along with the desired tools. Let's take AllenNLP
  as an example:

```shell
git clone https://github.com/asyml/forte-wrappers.git
cd forte-wrappers
pip install src/allennlp
```

### Libraries and Tools Supported

- [NLTK](https://www.nltk.org/) ([Processors](https://github.com/asyml/forte-wrappers/tree/main/src/nltk))
    - From PyPI: `pip install forte.nltk`
    - From source: `pip install src/nltk`
    - Features:
      - POS Tagger
      - Sentence Segmenter
      - Tokenizer
      - Lemmatizer
      - NER
- [spaCy](https://spacy.io/) ([Processors](https://github.com/asyml/forte-wrappers/tree/main/src/spacy))
    - From PyPI: `pip install forte.spacy`
    - From source: `pip install src/spacy`
    - Features:
      - Tokenizer, Lemmatizer and POS Tagging
      - NER
- [AllenNLP](https://allennlp.org/) ([Processors](https://github.com/asyml/forte-wrappers/tree/main/src/allennlp))
    - From PyPI: `pip install forte.allennlp`
    - From source: `pip install src/allennlp`
    - Features:
      - Tokenizer, POS Tagging
      - Semantic Role Labeling
      - Dependency Parsing
- [Stanza](https://stanfordnlp.github.io/stanza/) ([Processors](https://github.com/asyml/forte-wrappers/tree/main/src/stanza))
    - From PyPI: `pip install forte.stanza`
    - From source: `pip install src/stanza`
    - Features:
      - Tokenization, POS Tagging, Lemmatizer
      - Dependency Parsing
- [HuggingFace Models](https://huggingface.co/)
    - From PyPI: `pip install forte.huggingface`
    - From source: `pip install src/huggingface`
    - Features:
      - [BioBERT NER](https://github.com/dmis-lab/biobert-pytorch) ([Processors](https://github.com/asyml/forte-wrappers/tree/main/src/huggingface/forte/huggingface/biobert_ner))
      - [Zero Shot Classifier](https://huggingface.co/models?pipeline_tag=zero-shot-classification)([Processors](https://github.com/asyml/forte-wrappers/blob/main/src/huggingface/forte/huggingface/zero_shot_classifier.py))
      - [Question Answering](https://huggingface.co/models?pipeline_tag=question-answering)([Processors](https://github.com/asyml/forte-wrappers/blob/main/src/huggingface/forte/huggingface/question_and_answering_single.py))
- [Vader Sentiment](https://github.com/cjhutto/vaderSentiment) ([Processors](https://github.com/asyml/forte-wrappers/tree/main/src/vader))
    - From PyPI: `pip install forte.vader`
    - From source: `pip install src/vader`
    - Features:
      - Sentiment Analysis
- [Elastic Search](https://www.elastic.co/) ([Processors](https://github.com/asyml/forte-wrappers/tree/main/src/elastic))
    - From PyPI: `pip install forte.elastic`
    - From source: `pip install src/elastic`
    - Features:
      - Elastic Indexer
      - Elastic Search
- [Faiss](https://github.com/facebookresearch/faiss) ([Processors](https://github.com/asyml/forte-wrappers/tree/main/src/faiss))
    - From PyPI: `pip install forte.faiss`
    - From source: `pip install src/faiss`
    - Features:
      - Faiss Indexer
- [GPT2](https://openai.com/blog/gpt-2-1-5b-release/) ([Processors](https://github.com/asyml/forte-wrappers/tree/main/src/gpt2))
    - From PyPI: `pip install forte.gpt2`
    - From source: `pip install src/gpt2`
    - Features:
      - GPT2 Text Generation
- [Tweepy](https://docs.tweepy.org/en/latest/index.html) ([Processors](https://github.com/asyml/forte-wrappers/tree/main/src/tweepy))
    - From PyPI: `pip install forte.tweepy`
    - From source: `pip install src/tweepy`
    - Features:
      - TwitterAPI Search

### Contributing

If you are interested in making enhancement to this repository, Forte or other
ASYML/CASL projects, please first go over
our [Code of Conduct](https://github.com/asyml/forte/blob/master/CODE_OF_CONDUCT.md)
and [Contribution Guideline](https://github.com/asyml/forte/blob/master/CONTRIBUTING.md)

### License

This project is licensed by [Apache License 2.0](./LICENSE). The project provides wrappers to other open-sourced projects. To use
them in your project, please check the license of the corresponding project.

### Companies and Universities Supporting Forte

<p align="center">
   <img src="https://asyml.io/assets/institutions/cmu.png", width="200" align="top">
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   <img src="https://www.ucsd.edu/_resources/img/logo_UCSD.png" width="200" align="top">
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   <img src="https://raw.githubusercontent.com/asyml/forte/master/docs/_static/img/Petuum.png" width="200" align="top">
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</p>


