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
Wrapper of the Token Classification models on HuggingFace platform (context
understanding)
"""
import re
from typing import Dict, Set, List, Optional, Tuple, Any
from transformers import pipeline

from forte.common import Resources
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from forte.common import ProcessorConfigError
from forte.utils.tagging_scheme import bio_merge
from forte.utils import get_class

__all__ = [
    "TokenClassification",
]


class TokenClassification(PackProcessor):
    r"""Wrapper of the models on HuggingFace platform with pipeline tag of
    `ner`.
    https://huggingface.co/models?pipeline_tag=token-classification
    This wrapper could take the models on HuggingFace platform with pipeline
    tag of `ner` to make prediction on token classification tasks, for example,
    NER, POS tagging, word segmentation, etc.
    This processor takes user specified entry type in the input pack and the
    prediction result goes to the user specified attribute name of that entry
    type in the data pack.

    For example, if users want to do NER task on each sentence, and get BIO
    tagging merged, and store results as the attribute `ner_type` in each
    EntityMention, then the following config could be used:
        "entry_type": "ft.onto.base_ontology.Sentence",
        "output_entry_type": "ft.onto.base_ontology.EntityMention",
        "attribute_name": "ner",
        "tagging_scheme": "bio-merge"

    If user want to do POS tagging on each sentence and store results
    as the attribute `pos` in each Token, then the following config could
    be used:
        "entry_type": "ft.onto.base_ontology.Sentence",
        "output_entry_type": "ft.onto.base_ontology.Token",
        "attribute_name": "pos",
        "tagging_scheme": "no-merge"


    if user want to do word segmentation task on each sentence, and do BIO
    tagging merged, and store the results as the attribute `word_segment` in
    each Token, then the following config could be used:
        "entry_type": "ft.onto.base_ontology.Sentence",
        "output_entry_type": "ft.onto.base_ontology.Token",
        "attribute_name": "word_segment",
        "tagging_scheme": "bio-merge"

    For other settings, please refer to the `default_config` function for more
    information.

    """

    def __init__(self):
        super().__init__()
        self.classifier = None

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self.classifier = pipeline(
            "ner",
            model=self.configs.model_name,
            tokenizer=self.configs.tokenizer,
            framework=self.configs.framework,
            device=self.configs.cuda_device,
        )

    def _process(self, input_pack: DataPack):
        """Perform HuggingFace NER Pipeline on the input data pack.

        Args:
            input_pack: Input pack to fill
        Returns:
        """
        if not self.configs.entry_type:
            raise ProcessorConfigError("Please specify an input entry type!")

        output_entry = get_class(self.configs.output_entry_type)

        for entry_specified in input_pack.get(self.configs.entry_type):
            result = self.classifier(entry_specified.text)

            if self.configs.tagging_scheme == "bio-merge":  # Merge BIO tagging
                result_types, result_indices = self._merge_bio_tokens(result)

            elif self.configs.tagging_scheme == "no-merge":
                result_indices = []
                result_types = []
                for token in result:
                    start, end = token["start"], token["end"]
                    result_types.append(token["entity"])
                    result_indices.append((start, end))
            else:
                raise ProcessorConfigError(
                    f"The tagging_scheme strategy {self.configs.tagging_scheme}"
                    f"was not defined. Please check your input config."
                )

            for type, (start, end) in zip(result_types, result_indices):
                entity = output_entry(
                    pack=input_pack,
                    begin=entry_specified.span.begin + int(start),
                    end=entry_specified.span.begin + int(end),
                )
                setattr(entity, self.configs.attribute_name, type)

    def _merge_bio_tokens(
        self, tokens: List[Dict[str, Any]]
    ) -> Tuple[List[Optional[str]], List[Tuple[int, int]]]:
        """Perform token merge on BIO tagging format.

        Args:
            tokens: the output result from model
        Returns:
            result_types: list of merged entity type.
            result_indices: list of (start, end) index for the merged entities.

        """
        indices = [(token["start"], token["end"]) for token in tokens]

        tags: List[str] = []
        types: List[str] = []

        for token in tokens:
            if re.match("^[B|I]-.*", token["entity"]):
                tag, type = token["entity"].split("-")
            elif token["entity"] in ["B", "I"]:
                tag, type = token["entity"], ""
            else:
                raise ValueError("cannot process the BIO pattern.")
            tags.append(tag)
            types.append(type)

        result_types, result_indices = bio_merge(tags, types, indices)
        return result_types, result_indices

    @classmethod
    def default_configs(cls):
        r"""This defines a basic config structure for TokenClassification.

        Following are the keys for this dictionary:
            - `entry_type`: defines which entry type in the input pack to make
              prediction on. The default makes prediction on each `Sentence`
              in the input pack.
            - `output_entry_type`: defines which entry type in the input pack
              that the prediction should be saved as output. The default
              saves prediction on `Token` in the input pack.
            - `attribute_name`: defines which attribute of the
              `output_entry_type` in the input pack to save prediction to.
              The default saves prediction to the `ner` attribute for each
              `output_entry_type` in the input pack.
            - `tagging_scheme`: defines the result merged tagging_scheme,
              for example, "no-merge" to output the original result,
              "bio-merge" to merge the BIO tagging format result.
              The default is "no-merge".
            - `model_name`: language model, default is `"dslim/bert-base-NER"`.
              The wrapper supports HuggingFace models with pipeline tag of
              `ner`.
            - `tokenizer`: language model for tokenization, default is
              `"dslim/bert-base-NER"`.
            - `framework`: The framework of the model, should be `"pt"` or
              `"tf"`. This information should be obtained from Huggingface
              model hub.
            - `cuda_device`: Device ordinal for CPU/GPU supports. Setting
              this to -1 will leverage CPU, a positive will run the model
              on the associated CUDA device id. Default is -1.

        Returns: A dictionary with the default config for this processor.
        """
        config = super().default_configs()
        config.update(
            {
                "entry_type": "ft.onto.base_ontology.Sentence",
                "output_entry_type": "ft.onto.base_ontology.EntityMention",
                "attribute_name": "ner",
                "tagging_scheme": "bio-merge",  # "no-merge"
                "model_name": "dslim/bert-base-NER",
                "tokenizer": "dslim/bert-base-NER",
                "framework": "pt",
                "cuda_device": -1,
            }
        )
        return config

    def expected_types_and_attributes(self):
        r"""Method to add expected type `ft.onto.base_ontology.Sentence` which
        would be checked before running the processor if
        the pipeline is initialized with
        `enforce_consistency=True` or
        :meth:`~forte.pipeline.Pipeline.enforce_consistency` was enabled for
        the pipeline.
        """
        return {self.configs["entry_type"]: set()}

    def record(self, record_meta: Dict[str, Set[str]]):
        r"""Method to add output type record of `output_entry_type`, which is
        user specified entry type with user specified attribute name
        to :attr:`forte.data.data_pack.Meta.record`.

        Args:
            record_meta: the field in the datapack for type record that need to
                fill in for consistency checking.
        """

        if self.configs.output_entry_type not in record_meta.keys():
            record_meta[self.configs.output_entry_type] = set()
