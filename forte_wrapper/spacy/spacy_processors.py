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
from typing import Optional

import spacy
from spacy.language import Language
from spacy.cli.download import download
from forte.common import ProcessExecutionException
from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from ft.onto.base_ontology import EntityMention, Sentence, Token

__all__ = [
    "SpacyProcessor",
]


class SpacyProcessor(PackProcessor):
    """
    A wrapper for spaCy processors
    """

    def __init__(self):
        super().__init__()
        self.processors: str = ""
        self.nlp: Optional[Language] = None
        self.lang_model: str = ''

    def set_up(self):
        try:
            self.nlp = spacy.load(self.lang_model)
        except OSError:
            download(self.lang_model)
            self.nlp = spacy.load(self.lang_model)

        if 'ent_link' in self.processors:
            from scispacy.linking import EntityLinker
            linker = EntityLinker(resolve_abbreviations=True, name="umls")

            self.nlp.add_pipe(linker)

    # pylint: disable=unused-argument
    def initialize(self, resources: Resources, configs: Config):
        self.processors = configs.processors
        self.lang_model = configs.lang
        self.set_up()

    @classmethod
    def default_configs(cls):
        """
        This defines a basic config structure for spaCy.
        Returns:
            dictionary with the default config for this processor.
        Following are the keys for this dictionary:
            - processors: defines what operations to be done on the sentence,
                default value is "tokenize,pos,lemma" which performs all the
                three operations.
            - lang: language model, default value is 'en_core_web_sm'.
            - use_gpu: use gpu or not, default value is False.
        """
        config = super().default_configs()
        config.update({
            'processors': 'tokenize, pos, lemma, sent_segment, ent_link',
            'lang': 'en_core_web_sm',
            # Language code for the language to build the Pipeline
            'use_gpu': False,
        })
        return config

    def _process_parser(self, sentences, input_pack):
        """Parse the sentence. Default behaviour is to segment sentence, POSTag
        and Lemmatize.

        Args:
            sentences: Generator object which yields sentences in document
            input_pack: input pack which needs to be modified

        Returns:

        """
        for sentence in sentences:
            Sentence(input_pack, sentence.start_char, sentence.end_char)

            if "tokenize" in self.processors:
                # Iterating through spaCy token objects
                for word in sentence:
                    begin_pos_word = word.idx
                    end_pos_word = begin_pos_word + len(word.text)
                    token = Token(input_pack, begin_pos_word, end_pos_word)

                    if "pos" in self.processors:
                        token.pos = word.tag_

                    if "lemma" in self.processors:
                        token.lemma = word.lemma_

    def _process_ner(self, result, input_pack):
        """Perform spaCy's NER Pipeline on the document.

        Args:
            result: SpaCy results
            input_pack: Input pack to fill

        Returns:

        """
        for item in result.ents:
            entity = EntityMention(input_pack, item.start_char,
                                   item.end_char)
            entity.ner_type = item.label_

    def _process_entity_linking(self, result, input_pack):
        """
        Do entity linking task, and store medical entity mentions
        :param result:
        :param input_pack:
        :return:
        """
        from onto.medical import MedicalEntityMention, UMLSConceptLink

        # if "sent_segment" in self.processors:
        #     for sentence in result.sents:
        #         Sentence(input_pack, sentence.start_char, sentence.end_char)

        medical_entities = result.ents
        # linker = self.nlp.get_pipe("scispacy_linker")
        linker = self.nlp.get_pipe('EntityLinker')

        # get medical entity mentions and UMLS concepts
        for item in medical_entities:
            entity = MedicalEntityMention(input_pack, item.start_char,
                                   item.end_char)
            entity.ner_type = item.label_

            for umls_ent in item._.kb_ents:
                cui = umls_ent[0]
                score = str(umls_ent[1])

                cui_entity = linker.kb.cui_to_entity[cui]

                umls = UMLSConceptLink(input_pack)
                umls.cui = cui
                umls.score = score
                umls.name = cui_entity.canonical_name
                umls.definition = cui_entity.definition
                umls.tuis = cui_entity.types
                umls.aliases = cui_entity.aliases

                entity.umls_entities.append(umls)

    def _process(self, input_pack: DataPack):
        doc = input_pack.text

        # Do all process.
        if self.nlp is None:
            raise ProcessExecutionException(
                "The SpaCy pipeline is not initialized, maybe you "
                "haven't called the initialization function.")
        result = self.nlp(doc)

        # Record NER results.
        self._process_ner(result, input_pack)

        # Process sentence parses.
        if 'sent_segment' in self.processors:
            self._process_parser(result.sents, input_pack)

        # Record entity linking results.
        if 'ent_link' in self.processors:
            self._process_entity_linking(result, input_pack)
