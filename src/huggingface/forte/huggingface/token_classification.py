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
import importlib
import re
from typing import Dict, Set
from transformers import pipeline

from forte.common import Resources
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from forte.common import ProcessorConfigError
from forte.utils.tagging_scheme import bio_merge

__all__ = [
    "TokenClassification",
]


class TokenClassification(PackProcessor):
    r"""Wrapper of the models on HuggingFace platform with pipeline tag of
    `ner`.
    https://huggingface.co/models?pipeline_tag=token-classification
    This wrapper could take any model name on HuggingFace platform with pipeline
    tag of `ner` in configs to make prediction on the user
    specified entry type in the input pack and the prediction result goes to the
    user specified attribute name of that entry type in the output pack. User
    could provide task and models in the config to perform NER, POS,
    word segmentation task, etc.

    """

    def __init__(self):
        super().__init__()
        self.classifier = None

    def set_up(self):
        self.classifier = pipeline(
            "ner",
            model=self.configs.model_name,
            tokenizer=self.configs.tokenizer,
            framework=self.configs.framework,
            device=self.configs.cuda_device,
        )

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self.set_up()

    def _process(self, input_pack: DataPack):
        """Perform HuggingFace NER Pipeline on the input data pack.

        Args:
            input_pack: Input pack to fill
        Returns:
        """
        input_path_str, input_module_str = self.configs.entry_type.rsplit(".", 1)
        mod = importlib.import_module(input_path_str)
        input_entry = getattr(mod, input_module_str)

        output_path_str, output_module_str = self.configs.output_entry_type.rsplit(
            ".", 1
        )
        mod = importlib.import_module(output_path_str)
        output_entry = getattr(mod, output_module_str)

        for entry_specified in input_pack.get(entry_type=input_entry):
            result = self.classifier(entry_specified.text)

            if self.configs.strategy == "bio-merge":  # Merge BIO tagging
                result_types, result_indices = self._merge_bio_tokens(result)

            else:
                result_indices = []
                result_types = []
                for idx, token in enumerate(result):
                    start, end = token["start"], token["end"]
                    result_types.append(token["entity"])
                    result_indices.append((start, end))

            for type, (start, end) in zip(result_types, result_indices):
                entity = output_entry(
                    pack=input_pack,
                    begin=entry_specified.span.begin + int(start),
                    end=entry_specified.span.begin + int(end),
                )

                if self.configs.task == "ner":
                    if output_module_str == "EntityMention":
                        entity.ner_type = type
                    else:
                        try:
                            entity.ner = type
                        except KeyError as exc:
                            raise ProcessorConfigError(
                                f"NER type was not stored in the given "
                                f"'output_entry_type': "
                                f"{self.configs.output_entry_type}."
                                f"EntityMention or Token was recommended."
                            ) from exc

                if self.configs.task == "pos":
                    try:
                        entity.pos = type
                    except KeyError as exc:
                        raise ProcessorConfigError(
                            f"POS type was not stored in the given "
                            f"'output_entry_type': "
                            f"{self.configs.output_entry_type}."
                            f"Token was recommended."
                        ) from exc

    def _merge_bio_tokens(self, tokens):
        """Perform token merge on BIO tagging format.

        Args:
            tokens: the output result from model
        Returns:
            result_types: list of merged entity type.
            result_indices: list of (start, end) index for the merged entities.

        """
        indices = [(token["start"], token["end"]) for token in tokens]

        tags, types = [], []

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
        r"""This defines a basic config structure for ZeroShotClassifier.

        Following are the keys for this dictionary:
            - `entry_type`: defines which entry type in the input pack to make
              prediction on. The default makes prediction on each `Sentence`
              in the input pack.
            - `output_entry_type`: defines which entry type in the input pack
              that the prediction should be saved as output. The default
              saves prediction on `Token` in the input pack.
            - `task`: defines the nlp task, for example, "ner", "pos", "ws".
            - `strategy`: defines the result merged strategy, for example,
              "no_merge" to output the original result, "bio-merge" to merge
              the BIO tagging format result. The default is "no_merge".
            - `model_name`: language model, default is `"dslim/bert-base-NER"`.
              The wrapper supports HuggingFace models with pipeline tag of
              `ner`.
            - `tokenizer`: language model for tokenization, default is
              `"dslim/bert-base-NER"`.
            - `framework`: The framework of the model, should be `"pt"` or
              `"tf"`.
            - `cuda_device`: Device ordinal for CPU/GPU supports. Setting
              this to -1 will leverage CPU, a positive will run the model
              on the associated CUDA device id.

        Returns: A dictionary with the default config for this processor.
        """
        config = super().default_configs()
        config.update(
            {
                "entry_type": "ft.onto.base_ontology.Sentence",
                "output_entry_type": "ft.onto.base_ontology.Token",
                "task": "ner",  # "pos", "ws"
                "strategy": "no_merge",  # 'bio-merge'
                "model_name": "dslim/bert-base-NER",
                "tokenizer": "dslim/bert-base-NER",
                "framework": "pt",
                "cuda_device": -1,
            }
        )
        return config

    @classmethod
    def expected_types_and_attributes(cls):
        r"""Method to add expected type `ft.onto.base_ontology.Sentence` which
        would be checked before running the processor if
        the pipeline is initialized with
        `enforce_consistency=True` or
        :meth:`~forte.pipeline.Pipeline.enforce_consistency` was enabled for
        the pipeline.
        """
        return {cls().default_configs()["entry_type"]: set()}

    def record(self, record_meta: Dict[str, Set[str]]):
        r"""Method to add output type record of `ner` which is
        user specified entry type with user specified attribute name
        to :attr:`forte.data.data_pack.Meta.record`.

        Args:
            record_meta: the field in the datapack for type record that need to
                fill in for consistency checking.
        """

        if self.configs.output_entry_type not in record_meta.keys():
            record_meta[self.configs.output_entry_type] = set()
