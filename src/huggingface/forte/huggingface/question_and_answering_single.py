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
Wrapper of the Question Answering models on HuggingFace platform (context
understanding)
"""
import importlib
from typing import Dict, Set

from transformers import pipeline
from ft.onto.base_ontology import Phrase

from forte.common import Resources
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor

__all__ = [
    "QuestionAnsweringSingle",
]


class QuestionAnsweringSingle(PackProcessor):
    r"""Wrapper of the models on HuggingFace platform with pipeline tag of
    `question-answering` (reading comprehension).
    https://huggingface.co/models?pipeline_tag=question-answering
    This wrapper could take any model name on HuggingFace platform with pipeline
    tag of `question-answering` in configs to make prediction on the context of
    user specified entry type in the input pack and the prediction result would
    be annotated as `Phrase` in the output pack. User could input the question
    in the config.
    """

    def __init__(self):
        super().__init__()
        self.extractor = None

    def set_up(self):
        device_num = self.configs["cuda_devices"]
        self.extractor = pipeline(
            "question-answering",
            model=self.configs.model_name,
            tokenizer=self.configs.model_name,
            framework="pt",
            device=device_num,
        )

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self.set_up()

    def _process(self, input_pack: DataPack):
        path_str, module_str = self.configs.entry_type.rsplit(".", 1)

        mod = importlib.import_module(path_str)
        entry = getattr(mod, module_str)
        for entry_specified in input_pack.get(entry_type=entry):
            result = self.extractor(
                context=entry_specified.text,
                question=self.configs.question,
                max_answer_len=self.configs.max_answer_len,
                handle_impossible_answer=self.configs.handle_impossible_answer,
            )
            start = result["start"]
            end = result["end"]
            Phrase(pack=input_pack, begin=start, end=end)

    @classmethod
    def default_configs(cls):
        r"""This defines a basic config structure for `QuestionAnsweringSingle`.

        Following are the keys for this dictionary:
            - `entry_type`: defines which entry type in the input pack to make
              prediction on. The default makes prediction on each `Document`
              in the input pack.
            - `model_name`: language model, default is
              `"ktrapeznikov/biobert_v1.1_pubmed_squad_v2"`.
              The wrapper supports Hugging Face models with pipeline tag of
              `question-answering`.
            - `question`: One question to retrieve answer from the input pack
              context.
            - `max_answer_len`: The maximum length of predicted answers (e.g.,
              only answers with a shorter length are considered).
            - `cuda_device`: Device ordinal for CPU/GPU supports. Setting
              this to -1 will leverage CPU, a positive will run the model
              on the associated CUDA device id.
            - `handle_impossible_answer`: Whether or not we accept
              impossible as an answer.

        Returns: A dictionary with the default config for this processor.
        """
        config = super().default_configs()
        config.update(
            {
                "entry_type": "ft.onto.base_ontology.Document",
                "model_name": "ktrapeznikov/biobert_v1.1_pubmed_squad_v2",
                "question": "Where do I live",
                "max_answer_len": 15,
                "cuda_devices": -1,
                "handle_impossible_answer": False,
            }
        )
        return config

    def expected_types_and_attributes(self):
        r"""Method to add user specified expected type which
        would be checked before running the processor if
        the pipeline is initialized with
        `enforce_consistency=True` or
        :meth:`~forte.pipeline.Pipeline.enforce_consistency` was enabled for
        the pipeline.
        """
        return {self.configs["entry_type"]: set()}

    def record(self, record_meta: Dict[str, Set[str]]):
        r"""Method to add output type record of `QuestionAnsweringSingle` which
        is `"ft.onto.base_ontology.Phrase"`
        to :attr:`forte.data.data_pack.Meta.record`.

        Args:
            record_meta: the field in the datapack for type record that need to
                fill in for consistency checking.
        """
        if "ft.onto.base_ontology.Phrase" not in record_meta.keys():
            record_meta["ft.onto.base_ontology.Phrase"] = set()
