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
from copy import deepcopy
from typing import Optional, Dict, Set, List, Any, Iterator

import spacy
from forte.common import ProcessExecutionException, ProcessorConfigError
from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.base_pack import PackType
from forte.data.batchers import ProcessingBatcher, FixedSizeDataPackBatcher
from forte.data.data_pack import DataPack
from forte.data.ontology import Annotation
from forte.processors.base import PackProcessor, FixedSizeBatchProcessor
from ft.onto.base_ontology import EntityMention, Sentence, Token, Dependency
from ftx.medical import MedicalEntityMention, UMLSConceptLink
from packaging import version
from spacy.cli.download import download
from spacy.language import Language

__all__ = [
    "SpacyProcessor",
    "SpacyBatchedProcessor",
]

IS_SPACY_3 = False
if version.parse(spacy.__version__) >= version.parse("3.0.0"):  # type: ignore
    IS_SPACY_3 = True

CUSTOM_SPACYMODEL_URL = {
    "en_core_sci_sm": "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy"
    "/releases/v0.3.0/en_core_sci_sm-0.3.0.tar.gz",
    "en_core_sci_md": "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy"
    "/releases/v0.3.0/en_core_sci_md-0.3.0.tar.gz",
    "en_core_sci_lg": "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy"
    "/releases/v0.3.0/en_core_sci_lg-0.3.0.tar.gz",
    "en_ner_craft_md": "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy"
    "/releases/v0.3.0/en_ner_craft_md-0.3.0.tar.gz",
    "en_ner_jnlpba_md": "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy"
    "/releases/v0.3.0/en_ner_jnlpba_md-0.3.0.tar.gz",
    "en_ner_bc5cdr_md": "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy"
    "/releases/v0.3.0/en_ner_bc5cdr_md-0.3.0.tar.gz",
    "en_ner_bionlp13cg_md": "https://s3-us-west-2.amazonaws.com/ai2-s2"
    "-scispacy/releases/v0.3.0/en_ner_bionlp13cg_md-0"
    ".3.0.tar.gz",
}

SPACY2_DEFAULT_CONFIG2COMPONENT = {
    "sentence": "sentencizer",
    "pos": "tagger",
    "ner": "ner",
    "dep": "parser",
}

SPACY3_DEFAULT_CONFIG2COMPONENT = deepcopy(SPACY2_DEFAULT_CONFIG2COMPONENT)

SPACY3_DEFAULT_CONFIG2COMPONENT.update(
    {
        "sentence": "senter",
        "lemma": "lemmatizer",
    }
)


def validate_spacy_configs(configs: Config):
    """
    Validate the configuration of spacy.
    """
    if (
        "pos" in configs.processors
        or "lemma" in configs.processors
        or "dep" in configs.processors
    ):
        if "tokenize" not in configs.processors:
            raise ProcessorConfigError(
                "'tokenize' is necessary in configs.processors for 'pos', "
                "'lemma' or 'dep' (dependency parse)."
            )

    if "tokenize" in configs.processors:
        if "sentence" not in configs.processors:
            raise ProcessorConfigError(
                "'sentence' is necessary in configs.processors for 'tokenize'."
            )


def set_up_pipe(nlp: Language, configs: Config):
    config2component = (
        SPACY3_DEFAULT_CONFIG2COMPONENT
        if IS_SPACY_3
        else SPACY2_DEFAULT_CONFIG2COMPONENT
    )

    if IS_SPACY_3:
        for component in configs.processors:
            if component in config2component:
                component_ = config2component[component]
                if not nlp.has_pipe(component_):
                    nlp.add_pipe(component_)
    else:
        for component in configs.processors:
            if component in config2component:
                component_ = config2component[component]
                if not nlp.has_pipe(component_):
                    nlp.add_pipe(nlp.create_pipe(component_))

        # Haven't studied how to use scispacy in SpaCy 3.0+.
        if "umls_link" in configs.processors:
            # pylint: disable=import-outside-toplevel
            from scispacy.linking import EntityLinker

            linker = EntityLinker(resolve_abbreviations=True, name="umls")
            nlp.add_pipe(linker)

    # Remove some components to save some time.
    if configs.lang.startswith("en_core_web_sm"):
        for p in "pos", "ner", "dep", "sentence":
            if p not in configs.processors:
                if nlp.has_pipe(config2component[p]):
                    nlp.remove_pipe(config2component[p])


def load_lang_model(lang_model) -> Language:
    # pylint: disable=import-outside-toplevel
    if lang_model in CUSTOM_SPACYMODEL_URL:
        # download ScispaCy model using URL
        import subprocess
        import sys
        import os
        import importlib

        download_url = CUSTOM_SPACYMODEL_URL[lang_model]
        command = [sys.executable, "-m", "pip", "install"] + [download_url]
        subprocess.run(
            command, env=os.environ.copy(), encoding="utf8", check=False
        )
        cls = importlib.import_module(lang_model)
        return cls.load()  # type: ignore
    else:
        # Use spaCy download
        try:
            nlp = spacy.load(lang_model)  # type: ignore
        except OSError:
            download(lang_model)
            nlp = spacy.load(lang_model)  # type: ignore
    return nlp


class TextOnlyDataPackBatcher(FixedSizeDataPackBatcher):
    def _get_instance(self, data_pack: DataPack) -> Iterator[Dict[str, Any]]:
        yield {"text": data_pack.text}


class SpacyBatchedProcessor(FixedSizeBatchProcessor):
    """
    This processor wraps spaCy(v2.3.x) and ScispaCy(v0.3.0) models,
    providing most models included in the SpaCy pipeline, such as including
    sentence parsing, tokenize, POS tagging, lemmatization, NER, and medical
    entity linking. This is the batch processing version for
    :class:`~fortex.spacy.SpacyProcessor`, where it supports to
    batching across different data packs.

    This processor will do user defined tasks according to configs.
    The supported tasks includes:

    - `sentence`: sentence segmentation

    - `tokenize`: word tokenize

    - `pos`: Part-of-speech tagging

    - `lemma`: word lemmatization

    - `ner`: named entity recognition

    - `dep`: dependency parsing

    - `umls_link`: medical entity linking to UMLS concepts

    Citation:

    - spaCy: Industrial-strength Natural Language Processing in Python

    - ScispaCy: Fast and Robust Models for Biomedical Natural Language
      Processing.
    """

    def __init__(self):
        super().__init__()
        self.processors: str = ""
        self.nlp: Optional[Language] = None

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        validate_spacy_configs(configs)
        if self.configs.require_gpu:
            spacy.require_gpu(self.configs.gpu_id)
        if self.configs.prefer_gpu:
            spacy.prefer_gpu(self.configs.gpu_id)
        self.nlp = load_lang_model(self.configs.lang)
        set_up_pipe(self.nlp, configs)

    @classmethod
    def define_batcher(cls) -> ProcessingBatcher:
        """
        The batcher take raw text from a fixed number of data packs.
        """
        return TextOnlyDataPackBatcher()

    def predict(self, data_batch: Dict) -> Dict[str, List[Any]]:
        return {
            "results": list(self.nlp.pipe(data_batch["text"]))  # type: ignore
        }

    def pack(
        self,
        pack: PackType,
        predict_results: Dict[str, Any],
        _: Optional[Annotation] = None,
    ):
        for result in predict_results["results"]:
            # Record NER results.
            if "ner" in self.configs.processors:
                process_ner(result, pack)

            # Process sentence and tokenize.
            if "sentence" in self.configs.processors:
                indexed_tokens = process_tokens(
                    self.configs.processors, result.sents, pack
                )

                # Process dependency parse.
                if "dep" in self.configs.processors:
                    process_parse(result, pack, indexed_tokens)

            # Record medical entity linking results.
            if "umls_link" in self.configs.processors:
                linker = self.nlp.get_pipe("EntityLinker")  # type: ignore
                process_umls_entity_linking(linker, result, pack)

    def record(self, record_meta: Dict[str, Set[str]]):
        r"""Method to add output type record of current processor
        to :attr:`forte.data.data_pack.Meta.record`. The processor produce
        different types with different settings of `processors` in config.

        Args:
            record_meta: the field in the data pack for type record that need to
                fill in for consistency checking.
        """
        set_records(record_meta, self.configs)

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        """
        Specify additional parameters for SpaCy processor.

        The available parameters are:

        - `batcher.batch_size`: max size of the batch (in terms of number of
           data packs).

        - `processors`: List of strings that defines which components
          will be included and will be performed on the input pack,
          default value is `["sentence", "tokenize", "pos", "lemma"]`
          which performs the basic operations included in spaCy models like
          `en_core_web_sm`, `sentence` performs segmentation, `tokenize`
          will perform tokenization and pos tagging, `ner` will perform
          named entity recognition, `lemma` will perform lemmatization.
          Additional values for this list further includes:
          `ner` for named entity and `dep` for dependency parsing.

        - `lang`: language model, default is spaCy `en_core_web_sm` model.
          The pipeline support spaCy and ScispaCy models.
          A list of available spaCy models could be found at
          https://spacy.io/models.
          For UMLS entity linking task, ScispaCy model trained on
          biomedical dataset is preferred. A list of available models
          could be found at
          https://github.com/allenai/scispacy/tree/v0.3.0

        - `require_gpu`: whether GPU is required, default value is False.
          This value is directly used by
          https://spacy.io/api/top-level#spacy.require_gpu

        - `prefer_gpu`: whether gpu is preferred, default value is False.
          This value is directly used by
          https://spacy.io/api/top-level#spacy.prefer_gpu

        - `gpu_id`: the GPU device index to use when GPU is enabled. Default
          is 0.

        """
        return {
            "batcher": {
                "batch_size": 1000,
            },
            "processors": ["sentence", "tokenize", "pos", "lemma"],
            "lang": "en_core_web_sm",
            "require_gpu": False,
            "prefer_gpu": False,
        }


class SpacyProcessor(PackProcessor):
    """
    This processor wraps spaCy(v2.3.x) and ScispaCy(v0.3.0) models,
    providing functions including sentence parsing, tokenize, POS tagging,
    lemmatization, NER, and medical entity linking.

    This processor will do user defined tasks according to configs.
    The supported tasks includes:

    - `sentence`: sentence segmentation

    - `tokenize`: word tokenize

    - `pos`: Part-of-speech tagging

    - `lemma`: word lemmatization

    - `ner`: named entity recognition

    - `dep`: dependency parsing

    - `umls_link`: medical entity linking to UMLS concepts

    spaCy is a library for advanced Natural Language Processing in Python
    and Cython.
    spaCy github page: https://github.com/explosion/spaCy/tree/v2.3.1

    ScispaCy is a Python package containing spaCy models for processing
    biomedical, scientific or clinical text.
    ScispaCy github page: https://github.com/allenai/scispacy/tree/v0.3.0

    Citation:

    - spaCy: Industrial-strength Natural Language Processing in Python

    - ScispaCy: Fast and Robust Models for Biomedical Natural Language
      Processing.

    """

    def __init__(self):
        super().__init__()
        self.nlp: Optional[Language] = None

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        validate_spacy_configs(configs)
        if self.configs.require_gpu:
            spacy.require_gpu(self.configs.gpu_id)
        if self.configs.prefer_gpu:
            spacy.prefer_gpu(self.configs.gpu_id)
        self.nlp = load_lang_model(self.configs.lang)
        set_up_pipe(self.nlp, configs)

    @classmethod
    def default_configs(cls):
        """
        This defines a basic config structure for spaCy.

        Following are the keys for this dictionary:

        - `processors`: List of strings that defines which components
          will be included and will be performed on the input pack,
          default value is `["sentence", "tokenize", "pos", "lemma"]`
          which performs the basic operations included in spaCy models like
          `en_core_web_sm`, `sentence` performs segmentation, `tokenize`
          will perform tokenization and pos tagging, `ner` will perform
          named entity recognition, `lemma` will perform lemmatization.

          Additional values for this list further includes:
          `ner` for named entity and `dep` for dependency parsing.

        - `lang`: language model, default is spaCy `en_core_web_sm` model.
          The pipeline support spaCy and ScispaCy models.
          A list of available spaCy models could be found at
          https://spacy.io/models.
          For UMLS entity linking task, ScispaCy model trained on
          biomedical dataset is preferred. A list of available models
          could be found at
          https://github.com/allenai/scispacy/tree/v0.3.0.

        - `require_gpu`: whether GPU is required, default value is False.
          This value is directly used by
          https://spacy.io/api/top-level#spacy.require_gpu

        - `prefer_gpu`: whether gpu is preferred, default value is False.
          This value is directly used by
          https://spacy.io/api/top-level#spacy.prefer_gpu

        - `gpu_id`: the GPU device index to use when GPU is enabled. Default
          is 0.

        Returns: A dictionary with the default config for this processor.
        """
        return {
            "processors": ["sentence", "tokenize", "pos", "lemma"],
            "lang": "en_core_web_sm",
            "require_gpu": False,
            "prefer_gpu": False,
            "gpu_id": 0,
        }

    def _process(self, input_pack: DataPack):
        doc = input_pack.text

        # Do all process.
        if self.nlp is None:
            raise ProcessExecutionException(
                "The SpaCy pipeline is not initialized, maybe you "
                "haven't called the initialization function."
            )
        result = self.nlp(doc)

        # Record NER results.
        if "ner" in self.configs.processors:
            process_ner(result, input_pack)

        # Process sentence and tokenize.
        if "sentence" in self.configs.processors:
            indexed_tokens = process_tokens(
                self.configs.processors, result.sents, input_pack
            )

            # Process dependency parse.
            if "dep" in self.configs.processors:
                process_parse(result, input_pack, indexed_tokens)

        # Record medical entity linking results.
        if "umls_link" in self.configs.processors:
            linker = self.nlp.get_pipe("EntityLinker")
            process_umls_entity_linking(linker, result, input_pack)

    def record(self, record_meta: Dict[str, Set[str]]):
        r"""Method to add output type record of current processor
        to :attr:`forte.data.data_pack.Meta.record`. The processor produce
        different types with different settings of `processors` in config.

        Args:
            record_meta: the field in the data pack for type record that need to
                fill in for consistency checking.
        """
        set_records(record_meta, self.configs)


def set_records(record_meta: Dict[str, Set[str]], configs: Config):
    if "sentence" in configs.processors:
        record_meta["ft.onto.base_ontology.Sentence"] = set()
        if "tokenize" in configs.processors:
            record_meta["ft.onto.base_ontology.Token"] = set()
            if "pos" in configs.processors:
                record_meta["ft.onto.base_ontology.Token"].add("pos")
            if "lemma" in configs.processors:
                record_meta["ft.onto.base_ontology.Token"].add("lemma")
    if "ner" in configs.processors:
        record_meta["ft.onto.base_ontology.EntityMention"] = {"ner_type"}
    if "dep" in configs.processors:
        record_meta["ft.onto.base_ontology.Dependency"] = {"dep_label"}
    if "umls_link" in configs.processors:
        record_meta["onto.medical.MedicalEntityMention"] = {"ner_type"}
        record_meta["onto.medical.UMLSConceptLink"] = {
            "cui",
            "score",
            "name",
            "definition",
            "tuis",
            "aliases",
        }


def process_tokens(
    processors, sentences, input_pack: DataPack
) -> Dict[int, Token]:
    """Basic tokenization and post tagging of the sentence.

    Args:
        processors: List of processor names.
        sentences: Generator object which yields sentences in document.
        input_pack: input pack which needs to be modified.

    Returns: A mapping from SpaCy token index to Forte Token.
    """
    indexed_tokens: Dict[int, Token] = {}

    for sentence in sentences:
        Sentence(input_pack, sentence.start_char, sentence.end_char)

        if "tokenize" in processors:
            # Iterating through spaCy token objects
            for word in sentence:
                begin_pos_word = word.idx
                end_pos_word = begin_pos_word + len(word.text)
                token = Token(input_pack, begin_pos_word, end_pos_word)

                if "pos" in processors:
                    token.pos = word.tag_

                if "lemma" in processors:
                    token.lemma = word.lemma_

                # Store the spacy token index to forte token mapping.
                indexed_tokens[word.i] = token
    return indexed_tokens


def process_parse(
    result, input_pack: DataPack, indexed_tokens: Dict[int, Token]
):
    """
    Add dependency parses to the document.

    Args:
        result: SpaCy results.
        input_pack: Input pack to fill.
        indexed_tokens: A mapping from Spacy's token id to Forte tokens.
    """
    for token in result:
        head_token = indexed_tokens[token.head.i]
        child_token = indexed_tokens[token.i]
        if not token.head.i == token.i:
            # We don't store the self dep, which is ROOT in SpaCy.
            dep = Dependency(input_pack, head_token, child_token)
            dep.dep_label = token.dep_


def process_ner(result, input_pack: DataPack):
    """Perform spaCy's NER Pipeline on the document.

    Args:
        result: SpaCy results.
        input_pack: Input pack to fill.
    """
    for item in result.ents:
        entity = EntityMention(input_pack, item.start_char, item.end_char)
        entity.ner_type = item.label_


def process_umls_entity_linking(linker, result, input_pack: DataPack):
    """
    Perform UMLS medical entity linking with EntityLinker, and store medical
    entity mentions and UMLS concepts.

    Args:
        linker: A Scispacy EntityLinker instance.
        result: SpaCy results.
        input_pack: Input data pack.

    Returns:

    """
    medical_entities = result.ents

    # get medical entity mentions and UMLS concepts
    for item in medical_entities:
        entity = MedicalEntityMention(
            input_pack, item.start_char, item.end_char
        )
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
