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
"""
Unit tests for TestTokenClassification processor.
"""
import unittest

from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.data.readers import StringReader
from forte.nltk import NLTKSentenceSegmenter
from forte.huggingface import TokenClassification


class TestTokenClassification(unittest.TestCase):
    def test_huggingface_ner_bio_classification(self):
        nlp = Pipeline[DataPack](enforce_consistency=True)
        nlp.set_reader(StringReader())
        nlp.add(NLTKSentenceSegmenter())
        token_config = {
            "entry_type": "ft.onto.base_ontology.Sentence",
            "output_entry_type": "ft.onto.base_ontology.EntityMention",
            "task": "ner",
            "strategy": "bio-merge",
            "model_name": "dslim/bert-base-NER",
            "tokenizer": "dslim/bert-base-NER",
            "framework": "pt",
        }
        nlp.add(TokenClassification(), config=token_config)
        nlp.initialize()
        sentences = [
            "My name is Wolfgang and I live in Berlin. "
            "His name is Chris and he lives in Hawaii Island."
        ]

        pack = nlp.process(sentences)

        expected_type = [["PER", "LOC"], ["PER", "LOC"]]
        expected_index = [[(11, 19), (34, 40)], [(54, 59), (76, 89)]]

        for entry_idx, entry in enumerate(pack.get(token_config["entry_type"])):
            for idx, token in enumerate(
                pack.get(
                    entry_type=token_config["output_entry_type"],
                    range_annotation=entry,
                )
            ):

                self.assertEqual(token.ner_type, expected_type[entry_idx][idx])
                self.assertEqual(token.begin, expected_index[entry_idx][idx][0])
                self.assertEqual(token.end, expected_index[entry_idx][idx][1])

    def test_huggingface_ner_token_classification(self):
        nlp = Pipeline[DataPack](enforce_consistency=True)
        nlp.set_reader(StringReader())
        nlp.add(NLTKSentenceSegmenter())
        token_config = {
            "entry_type": "ft.onto.base_ontology.Sentence",
            "output_entry_type": "ft.onto.base_ontology.EntityMention",
            "task": "ner",  # "pos"
            "strategy": "no_merge",  # 'bio-merge'
            "model_name": "jplu/tf-xlm-r-ner-40-lang",
            "tokenizer": "jplu/tf-xlm-r-ner-40-lang",
            "framework": "tf",
        }
        nlp.add(TokenClassification(), config=token_config)
        nlp.initialize()
        sentences = ["Barack Obama was born in Hawaii."]

        pack = nlp.process(sentences)

        expected_type = [["PER", "PER", "LOC"]]
        expected_index = [[(0, 6), (7, 12), (25, 31)]]

        for entry_idx, entry in enumerate(pack.get(token_config["entry_type"])):
            for idx, token in enumerate(
                pack.get(
                    entry_type=token_config["output_entry_type"],
                    range_annotation=entry,
                )
            ):

                self.assertEqual(token.ner_type, expected_type[entry_idx][idx])
                self.assertEqual(token.begin, expected_index[entry_idx][idx][0])
                self.assertEqual(token.end, expected_index[entry_idx][idx][1])

    def test_huggingface_pos_token_classification(self):
        nlp = Pipeline[DataPack]()
        nlp.set_reader(StringReader())
        nlp.add(NLTKSentenceSegmenter())
        token_config = {
            "entry_type": "ft.onto.base_ontology.Sentence",
            "output_entry_type": "ft.onto.base_ontology.Token",
            "task": "pos",
            "strategy": "no_merge",
            "model_name": "vblagoje/bert-english-uncased-finetuned-pos",
            "tokenizer": "vblagoje/bert-english-uncased-finetuned-pos",
            "framework": "pt",
        }
        nlp.add(TokenClassification(), config=token_config)
        nlp.initialize()
        sentences = ["My name is Clara and I live in Berkeley, California."]

        pack = nlp.process(sentences)

        expected_type = [
            [
                "PRON",
                "NOUN",
                "AUX",
                "PROPN",
                "CCONJ",
                "PRON",
                "VERB",
                "ADP",
                "PROPN",
                "PUNCT",
                "PROPN",
                "PUNCT",
            ]
        ]
        expected_index = [
            [
                (0, 2),
                (3, 7),
                (8, 10),
                (11, 16),
                (17, 20),
                (21, 22),
                (23, 27),
                (28, 30),
                (31, 39),
                (39, 40),
                (41, 51),
                (51, 52),
            ]
        ]

        for entry_idx, entry in enumerate(pack.get(token_config["entry_type"])):
            for idx, token in enumerate(
                pack.get(
                    entry_type=token_config["output_entry_type"],
                    range_annotation=entry,
                )
            ):
                self.assertEqual(token.pos, expected_type[entry_idx][idx])
                self.assertEqual(token.begin, expected_index[entry_idx][idx][0])
                self.assertEqual(token.end, expected_index[entry_idx][idx][1])

    def test_huggingface_ner_doc_token_classification(self):
        nlp = Pipeline[DataPack]()
        nlp.set_reader(StringReader())

        token_config = {
            "entry_type": "ft.onto.base_ontology.Document",
            "output_entry_type": "ft.onto.base_ontology.EntityMention",
            "task": "ner",  # "pos"
            "strategy": "no_merge",  # 'bio-merge'
            "model_name": "jplu/tf-xlm-r-ner-40-lang",
            "tokenizer": "jplu/tf-xlm-r-ner-40-lang",
            "framework": "tf",
        }
        nlp.add(TokenClassification(), config=token_config)
        nlp.initialize()
        sentences = "Barack Obama was born in Hawaii."

        pack = nlp.process(sentences)

        expected_type = [["PER", "PER", "LOC"]]
        expected_index = [[(0, 6), (7, 12), (25, 31)]]

        for entry_idx, entry in enumerate(pack.get(token_config["entry_type"])):
            for idx, token in enumerate(
                pack.get(
                    entry_type=token_config["output_entry_type"],
                    range_annotation=entry,
                )
            ):
                self.assertEqual(token.ner_type, expected_type[entry_idx][idx])
                self.assertEqual(token.begin, expected_index[entry_idx][idx][0])
                self.assertEqual(token.end, expected_index[entry_idx][idx][1])

    def test_huggingface_ws_token_classification(self):
        nlp = Pipeline[DataPack]()
        nlp.set_reader(StringReader())
        nlp.add(NLTKSentenceSegmenter())
        token_config = {
            "entry_type": "ft.onto.base_ontology.Sentence",
            "output_entry_type": "ft.onto.base_ontology.Token",
            "task": "ws",
            "strategy": "bio-merge",
            "model_name": "ckiplab/bert-base-chinese-ws",
            "tokenizer": "ckiplab/bert-base-chinese-ws",
            "framework": "pt",
        }
        nlp.add(TokenClassification(), config=token_config)
        nlp.initialize()
        sentences = ["我叫克拉拉，我住在加州伯克利。"]

        pack = nlp.process(sentences)

        expected_index = [
            [
                (0, 1),
                (1, 2),
                (2, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 11),
                (11, 14),
                (14, 15),
            ]
        ]

        for entry_idx, entry in enumerate(pack.get(token_config["entry_type"])):
            for idx, token in enumerate(
                pack.get(
                    entry_type=token_config["output_entry_type"],
                    range_annotation=entry,
                )
            ):
                self.assertEqual(token.begin, expected_index[entry_idx][idx][0])
                self.assertEqual(token.end, expected_index[entry_idx][idx][1])


if __name__ == "__main__":
    unittest.main()
