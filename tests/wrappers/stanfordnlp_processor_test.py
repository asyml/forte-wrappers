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
Unit tests for Stanford NLP processors.
"""
import os
import unittest

from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.data.readers import StringReader
from fortex.stanza import StandfordNLPProcessor, StanfordNLPBioNERProcessor
from ft.onto.base_ontology import Token, Sentence, EntityMention


class TestStanfordNLPProcessor(unittest.TestCase):
    def setUp(self):
        self.stanford_nlp = Pipeline[DataPack]()
        self.stanford_nlp.set_reader(StringReader())
        config = {
            "processors": "tokenize,pos,lemma,depparse,ner",
            "lang": "en",
            # Language code for the language to build the Pipeline
            "use_gpu": False,
        }
        self.stanford_nlp.add(StandfordNLPProcessor(), config=config)
        self.stanford_nlp.initialize()

    def test_stanford_processing(self):
        sentences = [
            "This tool is called Forte.",
            "The goal of this project to help you build NLP " "pipelines.",
            "NLP has never been made this easy before."
            # "I have to say I am suffering headache and acutebronchitis."
        ]
        document = " ".join(sentences)
        pack = self.stanford_nlp.process(document)

        for idx, sentence in enumerate(pack.get(Sentence)):
            # sentence assertation
            self.assertEqual(sentence.text, sentences[idx])
        
        tokens = [
            ["This", "tool", "is", "called", "Forte", "."],
            [
                "The",
                "goal",
                "of",
                "this",
                "project",
                "to",
                "help",
                "you",
                "build",
                "NLP",
                "pipelines",
                "."
            ],
            [
                "NLP",
                "has",
                "never",
                "been",
                "made",
                "this",
                "easy",
                "before",
                "."
            ],
        ]

        pos = [['DET', 'NOUN', 'AUX', 'VERB', 'PROPN', 'PUNCT'], 
        ['DET', 'NOUN', 'ADP', 'DET', 'NOUN', 'PART', 'VERB', 'PRON', 'VERB', 'NOUN', 'NOUN', 'PUNCT'], 
        ['NOUN', 'AUX', 'ADV', 'AUX', 'VERB', 'PRON', 'ADJ', 'ADV', 'PUNCT']]

        for i, sentence in enumerate(pack.get(Sentence)):
            for j, token in enumerate(
                pack.get(entry_type=Token, range_annotation=sentence)
            ):
                # token/pos assertation
                self.assertEqual(token.text, tokens[i][j])
                self.assertEqual(token.pos, pos[i][j])

        entities_entries = list(pack.get(entry_type=EntityMention))

        texts = ['Forte', 'NLP']
        types = ['ORG','ORG']
        entities_text = [x.text for x in entities_entries]
        entities_type = [x.ner_type for x in entities_entries]

        # ner assertation
        self.assertEqual(entities_text, texts)
        self.assertEqual(entities_type, types)

        
class TestStanfordBioNERProcessor(unittest.TestCase):
    def setUp(self):
        self.stanford_nlp = Pipeline[DataPack]()
        self.stanford_nlp.set_reader(StringReader())
        self.stanford_nlp.add(StanfordNLPBioNERProcessor())
        self.stanford_nlp.initialize()

    def test_stanford_processing(self):
        sentences = [
            "I lived in New York.",
            "Yesterday my stomach ached violently.",
            "I think it may be appendicitis or gastroenteritis.",
            "Also it may be the acute pancreatitis.",
            "Is this Forte?"
        ]
        document = " ".join(sentences)

        print(document)
        pack = self.stanford_nlp.process(document)

        for idx, sentence in enumerate(pack.get(Sentence)):
            # sentence assertation
            self.assertEqual(sentence.text, sentences[idx])
        
        entities_entries = list(pack.get(entry_type=EntityMention))

        target_texts = ['appendicitis', 'gastroenteritis', 'the acute pancreatitis']
        target_types = ['PROBLEM', 'PROBLEM', 'PROBLEM']
        entities_text = [x.text for x in entities_entries]
        entities_type = [x.ner_type for x in entities_entries]

        self.assertEqual(entities_text, target_texts)
        self.assertEqual(entities_type, target_types)


if __name__ == "__main__":
    unittest.main()
