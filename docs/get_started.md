<div align="center">
   <img src="https://raw.githubusercontent.com/asyml/forte/master/docs/_static/img/logo_h.png"><br><br>
</div>

-----------------

[comment]: <> ([![codecov]&#40;https://codecov.io/gh/asyml/forte/branch/master/graph/badge.svg&#41;]&#40;https://codecov.io/gh/asyml/forte&#41;)

[![Python Build](https://github.com/asyml/forte-wrappers/actions/workflows/main.yml/badge.svg)](https://github.com/asyml/forte-wrappers/actions/workflows/main.yml)
[![Documentation Status](https://readthedocs.org/projects/forte-wrappers/badge/?version=latest)](https://forte-wrappers.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/asyml/forte/blob/master/LICENSE)
[![Chat](http://img.shields.io/badge/gitter.im-asyml/forte-blue.svg)](https://gitter.im/asyml/community)

**Forte** is a toolkit for building Natural Language Processing pipelines. This
repository tries to wrap the fantastic collections of NLP libraries built by the
community.

This project is part of the [CASL Open Source](http://casl-project.ai/) family.

# Get Started

- First, install the library along with the desired tools. Let's take AllenNLP
  as an example:

```shell
git clone https://github.com/asyml/forte-wrappers.git
cd forte-wrappers
pip install ."[allennlp]"
```

# Libraries and Wrapper Source

- [NLTK](https://www.nltk.org/) ([Processors](https://github.com/asyml/forte-wrappers/tree/main/forte/nltk))
    - POS Tagger
    - Sentence Segmenter
    - Tokenizer
    - Lemmatizer
    - NER
- [spaCy](https://spacy.io/) ([Processors](https://github.com/asyml/forte-wrappers/tree/main/forte/spacy))
    - Tokenizer, Lemmatizer and POS Tagging
    - NER
- [AllenNLP](https://allennlp.org/) ([Processors](https://github.com/asyml/forte-wrappers/tree/main/forte/allennlp))
    - Tokenizer, POS Tagging
    - Semantic Role Labeling
    - Dependency Parsing
- [Stanza](https://stanfordnlp.github.io/stanza/) ([Processors](https://github.com/asyml/forte-wrappers/tree/main/forte/stanza))
    - Tokenization, POS Tagging, Lemmatizer
    - Dependency Parsing
- [HuggingFace Models](https://huggingface.co/)
    - [BioBERT NER](https://github.com/dmis-lab/biobert-pytorch) ([Processors](https://github.com/asyml/forte-wrappers/tree/main/forte/huggingface/biobert_ner))
- [Vader Sentiment](https://github.com/cjhutto/vaderSentiment) ([Processors](https://github.com/asyml/forte-wrappers/tree/main/forte/vader))
    - Sentiment Analysis
- [Elastic Search](https://www.elastic.co/) ([Processors](https://github.com/asyml/forte-wrappers/tree/main/forte/elastic))
    - Elastic Indexer
    - Elastic Search
- [Faiss](https://github.com/facebookresearch/faiss) ([Processors](https://github.com/asyml/forte-wrappers/tree/main/forte/faiss))
    - Faiss Indexer
- [GPT2](https://openai.com/blog/gpt-2-1-5b-release/) ([Processors](https://github.com/asyml/forte-wrappers/tree/main/forte/gpt2))
    - GPT2 Text Generation
- [Tweepy](https://docs.tweepy.org/en/latest/index.html) ([Processors](https://github.com/asyml/forte-wrappers/tree/main/forte/twitter))
    - TwitterAPI Search
    
# License

[Apache License 2.0](https://github.com/asyml/forte-wrappers/blob/master/LICENSE)


