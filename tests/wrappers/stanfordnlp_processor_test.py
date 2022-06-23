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
import imp
import os
import unittest
from typing import List

from ddt import ddt, data
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.data.readers import StringReader
from fortex.stanza import StandfordNLPProcessor
from ft.onto.base_ontology import Token, Sentence, EntityMention, Dependency


@ddt
class TestStanfordNLPProcessor(unittest.TestCase):

    def test_stanford_processing(self):
        pipeline = Pipeline[DataPack]()
        pipeline.set_reader(StringReader())

        config = {
            "processors":{
               "tokenize":"default",
               "pos":"defualt",
               "lemma":"default",
               "depparse":"default",
               "ner":"i2b2"
            },
            "lang": "en",
            # Language code for the language to build the Pipeline
            "use_gpu": False,
        }

        pipeline.add(StandfordNLPProcessor(), config=config)
        pipeline.initialize()

        sentences = [
            "This tool is called Forte.",
            "The goal of this project to help you build NLP " "pipelines.",
            "NLP has never been made this easy before.",
            "I lived in New York.",
            "Yesterday my stomach ached violently.",
            "I think it may be appendicitis or gastroenteritis.",
            "Also it may be the acute pancreatitis.",
        ]

        document = " ".join(sentences)
        pack = pipeline.process(document)

        # Check document
        self.assertEqual(pack.text, document)

        # Check tokens
        tokens = [x.text for x in pack.get(Token)]
        document = document.replace(".", " .")
        self.assertEqual(tokens, document.split())

    @data(
        {
            "tokenize":"default",
            "pos":"defualt",
            "lemma":"default",
            "depparse":"default",
            "ner":"default"
        },  # all the configurations are set as defualt
        {
            "tokenize":"default",
            "pos":"defualt",
            "lemma":"default",
            "depparse":"default",
            "ner":"i2b2"
        },  # get bio ner
        {
            "ner":"i2b2"
        },  # test with ner only
        # {
        #     "tokenize":"mimic",
        #     "pos":"mimic",
        #     "lemma":"mimic",
        #     "depparse":"mimic",
        #     "ner":"i2b2"
        # } # test pipelie all based on mimic model
    )
    def test_stanza_pipeline(self, value):
        pipeline = Pipeline[DataPack]()
        pipeline.set_reader(StringReader())

        config = {
            "processors": value,
            "lang": "en",
            # Language code for the language to build the Pipeline
            "use_gpu": False,
        }

        pipeline.add(StandfordNLPProcessor(), config=config)
        pipeline.initialize()

        sentences = [
            "This tool is called Forte.",
            "The goal of this project to help you build NLP " "pipelines.",
            "NLP has never been made this easy before.",
            "I lived in New York.",
            "Yesterday my stomach ached violently.",
            "I think it may be appendicitis or gastroenteritis.",
            "Also it may be the acute pancreatitis.",
        ]

        document = " ".join(sentences)
        pack = pipeline.process(document)

        forte_tokens: List[Token] = list(pack.get(Token))  # type: ignore

        if "tokenize" in value:
            # Check tokenization
            tokens_text = [x.text for x in forte_tokens]
            self.assertEqual(
                tokens_text, pack.text.replace(".", " .").split()
            )

            # Check Part-of-Speech Tagger
            if "pos" in value:
                pos = [x.pos for x in forte_tokens]
                exp_pos = [
                    'DET', 'NOUN', 'AUX', 'VERB', 'PROPN', 'PUNCT',
                    'DET', 'NOUN', 'ADP', 'DET', 'NOUN', 'PART', 'VERB', 'PRON', 'VERB', 'NOUN', 'NOUN', 'PUNCT',
                    'NOUN', 'AUX', 'ADV', 'AUX', 'VERB', 'PRON', 'ADJ', 'ADV', 'PUNCT',
                    'PRON', 'VERB', 'ADP', 'PROPN', 'PROPN', 'PUNCT',
                    'NOUN', 'PRON', 'NOUN', 'VERB', 'ADV', 'PUNCT',
                    'PRON', 'VERB', 'PRON', 'AUX', 'AUX', 'NOUN', 'CCONJ', 'NOUN', 'PUNCT',
                    'ADV', 'PRON', 'AUX', 'AUX', 'DET', 'ADJ', 'NOUN', 'PUNCT'
                    ]
                self.assertListEqual(pos, exp_pos)

            # check lemmatization
            if "lemma" in value:
                lemma = [x.lemma for x in forte_tokens]
                exp_lemma = [
                    'this', 'tool', 'be', 'call', 'Forte', '.',
                    'the', 'goal', 'of', 'this', 'project', 'to', 'help', 'you', 'build', 'nlp', 'pipeline', '.',
                    'nlp', 'have', 'never', 'be', 'make', 'this', 'easy', 'before', '.',
                    'I', 'live', 'in', 'New', 'York', '.',
                    'yesterday', 'my', 'stomach', 'ach', 'violently', '.',
                    'I', 'think', 'it', 'may', 'be', 'appendicitis', 'or', 'gastroenteritis', '.',
                    'also', 'it', 'may', 'be', 'the', 'acute', 'pancreatitis', '.'
                    ]
                self.assertListEqual(lemma, exp_lemma)

            # Check dependency parsing
            if "depparse" in value:
                dependencies = [
                (dep.get_parent().text, dep.get_child().text, dep.rel_type)
                for dep in pack.get(Dependency)
                ]
                exp_dependencies = [
                    ('tool', 'This', 'det'), ('called', 'tool', 'nsubj:pass'), ('called', 'is', 'aux:pass'), ('.', 'called', 'root'), ('called', 'Forte', 'xcomp'), ('called', '.', 'punct'),
                    ('goal', 'The', 'det'), ('.', 'goal', 'root'), ('project', 'of', 'case'), ('project', 'this', 'det'), ('goal', 'project', 'nmod'), ('help', 'to', 'mark'), ('goal', 'help', 'acl'), ('help', 'you', 'obj'), ('help', 'build', 'xcomp'), ('pipelines', 'NLP', 'compound'), ('build', 'pipelines', 'obj'), ('goal', '.', 'punct'),
                    ('made', 'NLP', 'nsubj:pass'), ('made', 'has', 'aux'), ('made', 'never', 'advmod'), ('made', 'been', 'aux:pass'), ('.', 'made', 'root'), ('made', 'this', 'obj'), ('made', 'easy', 'xcomp'), ('made', 'before', 'advmod'), ('made', '.', 'punct'),
                    ('lived', 'I', 'nsubj'), ('.', 'lived', 'root'), ('York', 'in', 'case'), ('York', 'New', 'compound'), ('lived', 'York', 'obl'), ('lived', '.', 'punct'),
                    ('ached', 'Yesterday', 'obl:tmod'), ('stomach', 'my', 'nmod:poss'), ('ached', 'stomach', 'nsubj'), ('.', 'ached', 'root'), ('ached', 'violently', 'advmod'), ('ached', '.', 'punct'),
                    ('think', 'I', 'nsubj'), ('.', 'think', 'root'), ('appendicitis', 'it', 'nsubj'), ('appendicitis', 'may', 'aux'), ('appendicitis', 'be', 'cop'), ('think', 'appendicitis', 'ccomp'), ('gastroenteritis', 'or', 'cc'), ('appendicitis', 'gastroenteritis', 'conj'), ('think', '.', 'punct'),
                    ('pancreatitis', 'Also', 'advmod'), ('pancreatitis', 'it', 'nsubj'), ('pancreatitis', 'may', 'aux'), ('pancreatitis', 'be', 'cop'), ('pancreatitis', 'the', 'det'), ('pancreatitis', 'acute', 'amod'), ('.', 'pancreatitis', 'root'), ('pancreatitis', '.', 'punct')
                    ]
                self.assertListEqual(dependencies, exp_dependencies)

        # Check Bio NER
        if "ner" in value and value["ner"] == "i2b2":
            pack_ents: List[EntityMention] = list(pack.get(EntityMention))

            entities_text = [x.text for x in pack_ents]
            entities_type = [x.ner_type for x in pack_ents]

            exp_entities_text = ['appendicitis', 'gastroenteritis', 'the acute pancreatitis']
            exp_entities_type = ['PROBLEM', 'PROBLEM', 'PROBLEM']

            self.assertEqual(entities_text, exp_entities_text)
            self.assertEqual(entities_type, exp_entities_type)

        # Chech NER
        if "ner" in value and value["ner"] != "i2b2":
            pack_ents: List[EntityMention] = list(pack.get(EntityMention))

            entities_text = [x.text for x in pack_ents]
            entities_type = [x.ner_type for x in pack_ents]

            exp_entities_text = ['Forte', 'NLP', 'New York', 'Yesterday']
            exp_entities_type = ['ORG', 'ORG', 'GPE', 'DATE']

            self.assertEqual(entities_text, exp_entities_text)
            self.assertEqual(entities_type, exp_entities_type)


if __name__ == "__main__":
    unittest.main()
