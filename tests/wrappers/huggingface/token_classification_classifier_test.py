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
Unit tests for ZeroShotClassifier processor.
"""
import unittest

from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.data.readers import StringReader
from forte.nltk import NLTKSentenceSegmenter
from forte.huggingface import TokenClassification
from ft.onto.base_ontology import Sentence



class TestTokenClassification(unittest.TestCase):


    def test_huggingface_ner_token_classification(self):
        nlp = Pipeline[DataPack](enforce_consistency=True)
        nlp.set_reader(StringReader())
        nlp.add(NLTKSentenceSegmenter())
        token_config = {
            "entry_type": "ft.onto.base_ontology.Sentence",
            "output_entry_type": "ft.onto.base_ontology.EntityMention",
            "attribute_name": "ner", # "pos"
            'strategy': 'bio-merge', #'no_merge', # 'bio-merge'
            "model_name": 'dslim/bert-base-NER',
            "tokenizer": 'dslim/bert-base-NER',
            "framework": "pt",
        }
        nlp.add(TokenClassification(), config=token_config)
        nlp.initialize()
        sentences = [
            "My name is Wolfgang and I live in Berlin. "
            "His name is Chris and he lives in Hawaii Island."
        ]
        # document = " ".join(sentences)
        # pack = self.nlp.process(document)
        pack = nlp.process(sentences)

        # expected_type = [['PER', 'LOC']]
        # expected_index = [[(12, 17), (34, 47)]]
        expected_type = [['PER', 'LOC'], ['PER', 'LOC']]
        expected_index = [[(11, 19), (34, 40)], [(54, 59), (76, 89)]]

        for entry_idx, entry in enumerate(pack.get(token_config["entry_type"])):
            # print(entry.text)
            for idx, token in enumerate(pack.get(entry_type=token_config["output_entry_type"], range_annotation=entry)):
                # print(entry_idx, idx, token.ner_type, token.begin, token.end)


                # print('=====', entry_idx, idx)
                print(token.ner_type, expected_type[entry_idx][idx])
                print(token.begin, expected_index[entry_idx][idx][0])
                print(token.end, expected_index[entry_idx][idx][1])


                self.assertEqual(token.ner_type, expected_type[entry_idx][idx])
                self.assertEqual(token.begin, expected_index[entry_idx][idx][0])
                self.assertEqual(token.end, expected_index[entry_idx][idx][1])


if __name__ == "__main__":
    unittest.main()
