# Copyright 2021 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Dict, Set

import nltk
from nltk import pos_tag, ne_chunk, PunktSentenceTokenizer
from nltk.chunk import RegexpParser
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordTokenizer
from ft.onto.base_ontology import EntityMention, Token, Sentence, Phrase

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor

__all__ = [
    "NLTKPOSTagger",
    "NLTKSentenceSegmenter",
    "NLTKWordTokenizer",
    "NLTKLemmatizer",
    "NLTKChunker",
    "NLTKNER",
]


class NLTKWordTokenizer(PackProcessor):
    r"""A wrapper of NLTK word tokenizer."""

    def __init__(self):
        super().__init__()
        self.tokenizer = TreebankWordTokenizer()

    def _process(self, input_pack: DataPack):
        for begin, end in self.tokenizer.span_tokenize(input_pack.text):
            Token(input_pack, begin, end)

    def record(self, record_meta: Dict[str, Set[str]]):
        r"""Method to add output type record of `NLTKWordTokenizer`, which is
        `ft.onto.base_ontology.Token`,
        to :attr:`forte.data.data_pack.Meta.record`.

        Args:
            record_meta: the field in the datapack for type record that need to
                fill in for consistency checking.
        """
        record_meta["ft.onto.base_ontology.Token"] = set()


class NLTKPOSTagger(PackProcessor):
    r"""A wrapper of NLTK pos tagger."""

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        nltk.download("averaged_perceptron_tagger")

    def __init__(self):
        super().__init__()
        self.token_component = None

    def _process(self, input_pack: DataPack):
        token_entries = list(
            input_pack.get(entry_type=Token, components=self.token_component)
        )
        token_texts = [token.text for token in token_entries]
        taggings = pos_tag(token_texts)
        for token, tag in zip(token_entries, taggings):
            token.pos = tag[1]

    def record(self, record_meta: Dict[str, Set[str]]):
        r"""Method to add output type record of `NLTKPOSTagger`, which adds
        attribute `pos` to `ft.onto.base_ontology.Token`
        to :attr:`forte.data.data_pack.Meta.record`.

        Args:
            record_meta: the field in the datapack for type record that need to
                fill in for consistency checking.
        """
        record_meta["ft.onto.base_ontology.Token"].add("pos")

    def expected_types_and_attributes(self):
        r"""Method to add expected type `ft.onto.base_ontology.Token` for input
        which would be checked before running the processor if
        the pipeline is initialized with
        `enforce_consistency=True` or
        :meth:`~forte.pipeline.Pipeline.enforce_consistency` was enabled for
        the pipeline.
        """
        return {"ft.onto.base_ontology.Token": set()}


class NLTKLemmatizer(PackProcessor):
    r"""A wrapper of NLTK lemmatizer."""

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        nltk.download("wordnet")

    def __init__(self):
        super().__init__()
        self.token_component = None
        self.lemmatizer = WordNetLemmatizer()

    def _process(self, input_pack: DataPack):
        token_entries: List[Token] = list(
            input_pack.get(entry_type=Token, components=self.token_component)
        )

        token_texts: List[str] = []
        token_poses: List[str] = []
        for token in token_entries:
            token_texts.append(token.text)
            assert token.pos is not None
            token_poses.append(penn2morphy(token.pos))

        lemmas = [
            self.lemmatizer.lemmatize(token_texts[i], token_poses[i])
            for i in range(len(token_texts))
        ]
        for token, lemma in zip(token_entries, lemmas):
            token.lemma = lemma

    def record(self, record_meta: Dict[str, Set[str]]):
        r"""Method to add output type record of `NLTKLemmatizer` which adds
        attribute `lemma` to `ft.onto.base_ontology.Token`
        to :attr:`forte.data.data_pack.Meta.record`.

        Args:
            record_meta: the field in the datapack for type record that need to
                fill in for consistency checking.
        """
        record_meta["ft.onto.base_ontology.Token"].add("lemma")

    def expected_types_and_attributes(self):
        r"""Method to add expected type `ft.onto.base_ontology.Token` with
        attribute `pos` which
        would be checked before running the processor if
        the pipeline is initialized with
        `enforce_consistency=True` or
        :meth:`~forte.pipeline.Pipeline.enforce_consistency` was enabled for
        the pipeline.
        """
        return {"ft.onto.base_ontology.Token": {"pos"}}


def penn2morphy(penntag: str) -> str:
    r"""Converts tags from Penn format to Morphy."""
    morphy_tag = {"NN": "n", "JJ": "a", "VB": "v", "RB": "r"}
    if penntag[:2] in morphy_tag:
        return morphy_tag[penntag[:2]]
    else:
        return "n"


class NLTKChunker(PackProcessor):
    r"""A wrapper of NLTK chunker."""

    def __init__(self):
        super().__init__()
        self.chunker = None

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        nltk.download("maxent_ne_chunker")
        self.chunker = RegexpParser(configs.pattern)

    @classmethod
    def default_configs(cls):
        r"""This defines a basic config structure for NLTKChunker."""
        config = super().default_configs()
        config.update(
            {
                "pattern": "NP: {<DT>?<JJ>*<NN>}",
                "token_component": None,
                "sentence_component": None,
            }
        )
        return config

    def _process(self, input_pack: DataPack):
        for sentence in input_pack.get(
            Sentence, components=self.configs.sentence_component
        ):
            token_entries = list(
                input_pack.get(
                    entry_type=Token,
                    range_annotation=sentence,
                    components=self.configs.token_component,
                )
            )

            tokens = [(token.text, token.pos) for token in token_entries]
            cs = self.chunker.parse(tokens)

            index = 0
            for chunk in cs:
                if hasattr(chunk, "label"):
                    # For example:
                    # chunk: Tree('NP', [('This', 'DT'), ('tool', 'NN')])
                    begin_pos = token_entries[index].span.begin
                    end_pos = token_entries[index + len(chunk) - 1].span.end
                    phrase = Phrase(input_pack, begin_pos, end_pos)
                    phrase.phrase_type = chunk.label()

                    index += len(chunk)
                else:
                    # For example:
                    # chunk: ('is', 'VBZ')
                    index += 1

    def record(self, record_meta: Dict[str, Set[str]]):
        r"""Method to add output type record of `NLTKChunker` which adds
        `ft.onto.base_ontology.Phrase` with attribute `phrase_type`
        to :attr:`forte.data.data_pack.Meta.record`.

        Args:
            record_meta: the field in the datapack for type record that need to
                fill in for consistency checking.
        """
        record_meta["ft.onto.base_ontology.Phrase"] = {"phrase_type"}

    def expected_types_and_attributes(self):
        r"""Method to add expected type ft.onto.base_ontology.Token` with
        attribute `pos` and `ft.onto.base_ontology.Sentence` which
        would be checked before running the processor if
        the pipeline is initialized with
        `enforce_consistency=True` or
        :meth:`~forte.pipeline.Pipeline.enforce_consistency` was enabled for
        the pipeline.
        """
        return {
            "ft.onto.base_ontology.Sentence": set(),
            "ft.onto.base_ontology.Token": {"pos"},
        }


class NLTKSentenceSegmenter(PackProcessor):
    r"""A wrapper of NLTK sentence tokenizer."""

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        nltk.download("punkt")

    def __init__(self):
        super().__init__()
        self.sent_splitter = PunktSentenceTokenizer()

    def _process(self, input_pack: DataPack):
        for begin, end in self.sent_splitter.span_tokenize(input_pack.text):
            Sentence(input_pack, begin, end)

    def record(self, record_meta: Dict[str, Set[str]]):
        r"""Method to add output type record of `NLTKSentenceSegmenter`, which
        is `ft.onto.base_ontology.Sentence`
        to :attr:`forte.data.data_pack.Meta.record`.

        Args:
            record_meta: the field in the datapack for type record that need to
                fill in for consistency checking.
        """
        record_meta["ft.onto.base_ontology.Sentence"] = set()


class NLTKNER(PackProcessor):
    r"""A wrapper of NLTK NER."""

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        nltk.download("maxent_ne_chunker")
        nltk.download("words")

    def __init__(self):
        super().__init__()
        self.token_component = None

    def _process(self, input_pack: DataPack):
        for sentence in input_pack.get(Sentence):
            token_entries = list(
                input_pack.get(
                    entry_type=Token,
                    range_annotation=sentence,
                    components=self.token_component,
                )
            )
            tokens = [(token.text, token.pos) for token in token_entries]
            ne_tree = ne_chunk(tokens)

            index = 0
            for chunk in ne_tree:
                if hasattr(chunk, "label"):
                    # For example:
                    # chunk: Tree('GPE', [('New', 'NNP'), ('York', 'NNP')])
                    begin_pos = token_entries[index].span.begin
                    end_pos = token_entries[index + len(chunk) - 1].span.end
                    entity = EntityMention(input_pack, begin_pos, end_pos)
                    entity.ner_type = chunk.label()
                    index += len(chunk)
                else:
                    # For example:
                    # chunk: ('This', 'DT')
                    index += 1

    def record(self, record_meta: Dict[str, Set[str]]):
        r"""Method to add output type record of `NLTKNER` which is
        `ft.onto.base_ontology.EntityMention` with attribute `phrase_type`
        to :attr:`forte.data.data_pack.Meta.record`.

        Args:
            record_meta: the field in the datapack for type record that need to
                fill in for consistency checking.
        """
        record_meta["ft.onto.base_ontology.EntityMention"] = {"ner_type"}

    def expected_types_and_attributes(self):
        r"""Method to add expected type ft.onto.base_ontology.Token` with
        attribute `pos` and `ft.onto.base_ontology.Sentence` which
        would be checked before running the processor if
        the pipeline is initialized with
        `enforce_consistency=True` or
        :meth:`~forte.pipeline.Pipeline.enforce_consistency` was enabled for
        the pipeline.
        """
        return {
            "ft.onto.base_ontology.Sentence": set(),
            "ft.onto.base_ontology.Token": {"pos"},
        }
