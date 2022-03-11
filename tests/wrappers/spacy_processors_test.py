# Copyright 2019 The Forte Authors. All Rights Reserved.
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
"""
Unit tests for spaCy processors.
"""
import unittest
from typing import List

import spacy
from spacy.language import Language

from ddt import ddt, data
from forte.common import ProcessorConfigError
from forte.data.data_pack import DataPack
from forte.data.readers import StringReader
from forte.pipeline import Pipeline
from forte.utils import get_class
from ft.onto.base_ontology import Token, EntityMention, Dependency

from fortex.spacy import SpacyProcessor, SpacyBatchedProcessor
from fortex.spacy.spacy_processors import set_up_pipe


@ddt
class TestSpacyProcessor(unittest.TestCase):
    def test_spacy_processor(self):
        pipeline = Pipeline[DataPack]()
        pipeline.set_reader(StringReader())

        config = {
            "processors": ["sentence", "tokenize"],
            "lang": "en_core_web_sm",
            # Language code for the language to build the Pipeline
        }
        pipeline.add(SpacyProcessor(), config=config)
        pipeline.initialize()

        sentences = [
            "This tool is called Forte.",
            "The goal of this project to help you build NLP " "pipelines.",
            "NLP has never been made this easy before.",
        ]
        document = " ".join(sentences)
        pack = pipeline.process(document)

        # Check document
        self.assertEqual(pack.text, document)

        # Check tokens
        tokens = [x.text for x in pack.get(Token)]
        document = document.replace(".", " .")
        self.assertEqual(tokens, document.split())

    def check_results(self, processors, raw_results, data_pack, config):
        forte_tokens: List[Token] = list(data_pack.get(Token))  # type: ignore

        if "tokenize" in processors:
            exp_pos = []
            exp_lemma = []
            exp_deps = []
            for s in raw_results.sents:
                for w in s:
                    exp_lemma.append(w.lemma_)
                    exp_pos.append(w.tag_)

                    if not w.dep_ == "ROOT":
                        exp_deps.append(
                            (raw_results[w.head.i].text, w.text, w.dep_)
                        )

            tokens_text = [x.text for x in forte_tokens]
            self.assertEqual(
                tokens_text, data_pack.text.replace(".", " .").split()
            )

            pos = [x.pos for x in forte_tokens]
            lemma = [x.lemma for x in forte_tokens]

            dependencies = [
                (dep.get_parent().text, dep.get_child().text, dep.dep_label)
                for dep in data_pack.get(Dependency)
            ]

            # Check token texts
            for token, text in zip(forte_tokens, tokens_text):
                start, end = token.span.begin, token.span.end
                self.assertEqual(data_pack.text[start:end], text)

            if "pos" in processors:
                self.assertListEqual(pos, exp_pos)
            else:
                none_pos = [None] * len(pos)
                self.assertListEqual(pos, none_pos)

            if "lemma" in processors:
                self.assertListEqual(lemma, exp_lemma)
            else:
                none_lemma = [None] * len(lemma)
                self.assertListEqual(lemma, none_lemma)

            if "dep" in processors:
                self.assertListEqual(dependencies, exp_deps)
        else:
            self.assertListEqual(forte_tokens, [])

        if "ner" in processors:
            pack_ents: List[EntityMention] = list(data_pack.get(EntityMention))
            entities_text = [x.text for x in pack_ents]
            entities_type = [x.ner_type for x in pack_ents]

            raw_ents = raw_results.ents
            exp_ent_text = [
                data_pack.text[ent.start_char : ent.end_char]
                for ent in raw_ents
            ]
            exp_ent_types = [ent.label_ for ent in raw_ents]

            self.assertEqual(entities_text, exp_ent_text)
            self.assertEqual(entities_type, exp_ent_types)

    @data(
        ["sentence", "tokenize", "dep"],
        ["sentence", "tokenize", "pos"],
        ["sentence", "tokenize", "lemma"],
        ["sentence", "tokenize", "pos", "lemma"],
        ["sentence", "ner", "tokenize", "lemma", "pos"],
        ["ner"],
        ["sentence", "tokenize", "dep"],
    )
    def test_spacy_batch_pipeline(self, value):
        pipeline = Pipeline[DataPack]()
        pipeline.set_reader(StringReader())
        config = {
            "processors": value,
            "medical_onto_type": "ftx.onto.clinical.MedicalEntityMention",
            "umls_onto_type": "ftx.onto.clinical.UMLSConceptLink",
            "lang": "en_core_web_sm",
            # Language code for the language to build the Pipeline
            "batcher": {"batch_size": 2},
        }
        pipeline.add(SpacyBatchedProcessor(), config)
        pipeline.initialize()

        sentences = [
            "This tool is called Forte.",
            "The goal of this project to help you build NLP pipelines.",
            "NLP has never been made this easy before.",
        ]

        spacy_pipe: Language = spacy.load(config["lang"])
        set_up_pipe(spacy_pipe, pipeline.component_configs[0])

        for raw_results, pack in zip(
            spacy_pipe.pipe(sentences), pipeline.process_dataset(sentences)
        ):
            self.check_results(value, raw_results, pack, config)

    @data(
        ["sentence", "tokenize"],
        ["sentence", "tokenize", "pos"],
        ["sentence", "tokenize", "lemma"],
        ["sentence", "tokenize", "pos", "lemma"],
        ["sentence", "ner", "tokenize", "lemma", "pos"],
        ["ner"],
        ["sentence", "tokenize", "dep"],
    )
    def test_spacy_variation_pipeline(self, value):
        pipeline = Pipeline[DataPack]()
        pipeline.set_reader(StringReader())

        config = {
            "processors": value,
            "medical_onto_type": "ftx.onto.clinical.MedicalEntityMention",
            "umls_onto_type": "ftx.onto.clinical.UMLSConceptLink",
            "lang": "en_core_web_sm",
            # Language code for the language to build the Pipeline
        }
        pipeline.add(SpacyProcessor(), config=config)
        pipeline.initialize()

        sentences = [
            "This tool is called Forte.",
            "The goal of this project to help you build NLP pipelines.",
            "NLP has never been made this easy before.",
            "Head CT revealed no lesions.",
        ]
        document = " ".join(sentences)
        pack: DataPack = pipeline.process(document)

        spacy_pipe: Language = spacy.load(config["lang"])
        set_up_pipe(spacy_pipe, pipeline.component_configs[0])
        raw_results = spacy_pipe(document)

        self.check_results(value, raw_results, pack, config)

    @data(
        ["lemma"],  # tokenize is required for lemma
        ["tokenize"],  # sentence is required for tokenize
        ["pos"],  # sentence, tokenize is required for pos
    )
    def test_spacy_processor_with_invalid_config(self, processor):
        pipeline = Pipeline[DataPack]()
        pipeline.set_reader(StringReader())

        config = {
            "processors": processor,
            "medical_onto_type": "ftx.onto.clinical.MedicalEntityMention",
            "umls_onto_type": "ftx.onto.clinical.UMLSConceptLink",
            "lang": "en_core_web_sm",
            # Language code for the language to build the Pipeline
        }
        pipeline.add(SpacyProcessor(), config=config)

        with self.assertRaises(ProcessorConfigError):
            pipeline.initialize()

    @data(
        ["umls_link"],
    )
    def test_spacy_processor_for_umls_link(self, processor):
        pipeline = Pipeline[DataPack]()
        pipeline.set_reader(StringReader())

        config = {
            "processors": processor,
            "medical_onto_type": "ftx.onto.clinical.MedicalEntityMention",
            "umls_onto_type": "ftx.onto.clinical.UMLSConceptLink",
            "lang": "en_core_web_sm",
            # Language code for the language to build the Pipeline
        }
        pipeline.add(SpacyProcessor(), config=config)

        with not self.assertRaises(ProcessorConfigError):
            pipeline.initialize()


if __name__ == "__main__":
    unittest.main()
