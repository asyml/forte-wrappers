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
from transformers import pipeline
from ft.onto.base_ontology import Phrase, Sentence

from forte.common import Resources
from forte.common.configuration import Config
from forte.data.multi_pack import MultiPack, MultiPackLink
from forte.processors.base import MultiPackProcessor

__all__ = [
    "QuestionAnsweringMulti",
]


class QuestionAnsweringMulti(MultiPackProcessor):
    r"""Wrapper of the models on HuggingFace platform with pipeline tag of
    `question-answering` (reading comprehension).
    https://huggingface.co/models?pipeline_tag=question-answering
    This wrapper could take any model name on HuggingFace platform with pipeline
    tag of `question-answering` in configs to make prediction on the context of
    user specified entry type in the multiple input pack and the prediction
    result would be annotated as `Phrase` in the output pack, which would be
    linked to the question pack by `MultiPackLink`.
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

    def _process(self, input_pack: MultiPack):
        context_list = list()
        doc_id_list = list()
        for doc_id in input_pack.pack_names:
            if doc_id == self.configs.question_pack_name:
                continue

            pack = input_pack.get_pack(doc_id)
            context_list.append(pack.get_single(self.configs.entry_type).text)
            doc_id_list.append(doc_id)

        question_pack = input_pack.get_pack(self.configs.question_pack_name)
        first_question = question_pack.get_single(Sentence)
        question_list = [question_pack.text for i in range(len(context_list))]
        result_collection = self.extractor(
            context=context_list,
            question=question_list,
            max_answer_len=self.configs.max_answer_len,
            handle_impossible_answer=self.configs.handle_impossible_answer,
        )
        for i, result in enumerate(result_collection):
            start = result["start"]
            end = result["end"]
            doc_pack = input_pack.get_pack(doc_id_list[i])
            ans_phrase = Phrase(pack=doc_pack, begin=start, end=end)
            input_pack.add_entry(
                MultiPackLink(input_pack, first_question, ans_phrase)
            )

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
                "question_pack_name": "question",
                "entry_type": "ft.onto.base_ontology.Document",
                "model_name": "ktrapeznikov/biobert_v1.1_pubmed_squad_v2",
                "max_answer_len": 15,
                "cuda_devices": -1,
                "handle_impossible_answer": False,
            }
        )
        return config
