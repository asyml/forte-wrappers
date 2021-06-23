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
Wrapper of the Zero Shot Classifier models on HuggingFace platform
"""

import importlib
import re
import logging
from typing import Dict, Set
from transformers import pipeline
import logging
from typing import Optional, List, Union, Tuple

from forte.common import Resources
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor

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
    could input the prediction labels in the config with any word or phrase.

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
            device=self.configs.cuda_device
        )

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self.set_up()

    def _process(self, input_pack: DataPack):
        input_path_str, input_module_str = self.configs.entry_type.rsplit(".", 1)

        mod = importlib.import_module(input_path_str)
        input_entry = getattr(mod, input_module_str)

        output_path_str, output_module_str = self.configs.output_entry_type.rsplit(".", 1)
        mod = importlib.import_module(output_path_str)
        output_entry = getattr(mod, output_module_str)


        for entry_specified in input_pack.get(entry_type=input_entry):
            result = self.classifier(entry_specified.text)

            if self.configs.strategy == 'bio-merge':
                result_type, result_index = self._merge_bio_tokens(result)

            else:
                result_index = []
                result_type = []
                for idx, token in enumerate(result):
                    start, end = token['start'], token['end']
                    result_type.append(token['entity'])
                    result_index.append((start, end))

            print(result_type, result_index)

            # for type, (start, end) in zip(result_type, result_index):
            for idx in range(len(result_type)):
                type = result_type[idx]
                start = result_index[idx][0]
                end = result_index[idx][1]
                print(idx)
                print(type)
                print(start, end)
                entity = output_entry(pack=input_pack,
                                      begin=entry_specified.span.begin + int(start),
                                      end=entry_specified.span.begin+ int(end))

                if self.configs.attribute_name == "ner":
                    if output_module_str == "EntityMention":
                        entity.ner_type = type
                    # elif output_module_str == "Token":
                    #     entity.ner = type
                    else:
                        try:
                            entity.ner = type
                        except:
                            raise ValueError(f'NER type was not stored in the given '
                                      f'\'output_entry_type\': {self.configs.output_entry_type}.'
                                      f'EntityMention or Token was recommended.')


                if self.configs.attribute_name == "pos":
                    # if output_module_str == "Token":
                    try:
                        entity.pos = type
                    # else:
                    except:
                        raise ValueError(f'POS type was not stored in the given '
                                      f'\'output_entry_type\': {self.configs.output_entry_type}.'
                                      f'Token was recommended.')


    def _merge_bio_tokens(self, tokens):
        entities = [token['entity'] for token in tokens]
        # start = [(token['start'], token['end']) for token in tokens]
        # end = [token['end'] for token in tokens]
        index = [(token['start'], token['end']) for token in tokens]

        tags, types = [], []

        for token in tokens:
            if re.match('^[B|I]-.*', token['entity']):
                tag, type = token['entity'].split('-')
            elif token['entity'] in ['B', 'I']:
                tag, type = token['entity'], None
            else:
                raise ValueError('cannot process the BIO pattern.')
            tags.append(tag)
            types.append(type)

        print(tags, types, index)
        result_type, result_index = self.bio_merge(tags, types, index)
        print(result_type, result_index)
        return result_type, result_index


    def bio_merge(
            self, tags: List[str],
            types: List[Union[str, None]],
            index: Optional[
                List[Tuple[Union[int, None], Union[int, None]]]] = None,
    ) -> Tuple[
        List[Union[str, None]], List[Tuple[Union[int, None], Union[int, None]]]
    ]:
        r"""This function merged BIO-schemed augmented tagging scheme results and
        return chunks information.

        For example, BIO NER tags could be merged by passing
        tags = ['B', 'O', 'B', 'I']
        types = ['PER', '', 'LOC', 'LOC']
        index = [(0, 1), (11, 19), (20, 22), (24, 26)]

        After merging BIO tags, the results will be returned as
        result_types = ['PER', 'LOC']
        result_index = [(0, 1), (20, 26)]

        The function handles 'I' with no leading 'B' tag. If we encounter
        "I" while its type is different from the previous type, we will consider
        this "I" as a "B" and start a new record here.

        The function can also handle tags with no types, for example, in some word
        segmentation tasks. In this case the input `types` should be set as a list
        of None, and the returned result_type will be a list of None.

        Args:
            tags: list of bio tags, contains "B", "I", "O" labels.
            types: list of entity type, could be PER, LOC in NER task.
            index: list of (start, end) index for each input tag. default is None.

        Returns:
            result_types: list of merged entity type.
            result_index: list of (start, end) index for the merged entities.
        """
        prev_type: Optional[str] = None
        prev_tag: Optional[str] = None
        prev_start: Optional[int] = None
        prev_end: Optional[int] = None
        new_entity: bool = False
        is_indexed: bool = True

        # No start or end information is provided, do not process index information
        if index is None:
            is_indexed = False
            start: List[Union[int, None]] = []
            end: List[Union[int, None]] = []
            logging.warning(
                "start and end indexes for the tags was not provided "
                "and will be returned as `None`"
            )
        else:  # get start and end index
            start, end = zip(*index)  # type: ignore

        # input check
        if len(tags) != len(types) or (
                is_indexed and (
                len(start) != len(tags) or len(end) != len(tags))
        ):
            raise ValueError(
                "The input tags, types, start and end index have "
                "different length, please check."
            )

        for tag in tags:
            if tag not in ["B", "I", "O"]:
                raise ValueError(
                    "The BIO tags contain characters beyond `BIO`, "
                    "please check the input tags."
                )

        result_types: List[Union[str, None]] = []
        result_start: List[Union[int, None]] = []
        result_end: List[Union[int, None]] = []

        for idx, (tag, type) in enumerate(zip(tags, types)):
            if (
                    tag == "B"
                    or (tag == "I" and type != prev_type)
                    or (tag == "O" and prev_tag and prev_tag != "O")
            ):  # the last entity has ended
                if prev_tag and prev_tag != "O":
                    result_types.append(prev_type)
                    result_start.append(prev_start)
                    result_end.append(prev_end)

                if is_indexed:
                    prev_start = start[idx]
                    prev_end = end[idx]

                if tag != "O":  # a new entity started
                    new_entity = True

            elif tag == "I" and type == prev_type:  # continue with the last entity
                if is_indexed:
                    prev_end = end[idx]

            else:  # "O" tag
                new_entity = False

            prev_type = type
            prev_tag = tag

        if new_entity:  # check if the last entity is added in result
            result_types.append(prev_type)
            result_start.append(prev_start)
            result_end.append(prev_end)

        result_index: List[Tuple[Union[int, None], Union[int, None]]] = list(
            zip(result_start, result_end)
        )

        return result_types, result_index


    # def bio_merge(self, tags, types, start=None, end=None):
    #
    #     prev_type = None
    #     prev_tag = None
    #     prev_start = -1
    #     prev_end = -1
    #
    #     is_indexed = False if start is None or end is None else True
    #     new_entity = False
    #
    #     result = []
    #
    #     for index, (tag, type) in enumerate(zip(tags, types)):
    #         if tag not in ["B", "I", "O"]:
    #             raise ValueError(
    #                 'The BIO tags contains characters beyond \'BIO\', please check the input tags.')
    #
    #         if tag == "B" or (tag == "I" and type != prev_type) or (
    #                 tag == "O" and prev_tag and prev_tag != "O"):
    #             new_entity = True
    #             if prev_tag:
    #                 result.append((prev_type, prev_start, prev_end))
    #
    #             if is_indexed:
    #                 prev_start = start[index]
    #                 prev_end = end[index]
    #
    #
    #         elif tag == "I" and type == prev_type:
    #             if is_indexed:
    #                 prev_end = end[index]
    #
    #         else:
    #             new_entity = False
    #
    #         prev_type = type
    #         prev_tag = tag
    #
    #     if new_entity:
    #         result.append((prev_type, prev_start, prev_end))
    #
    #     return result




    # def _merge_bio_tokens(self, tokens):
    #     last = {}
    #     result = []
    #
    #     for idx, token in enumerate(tokens):
    #         # word = token['word']
    #         entity = token['entity']
    #
    #         if re.match('^[B|I]-.*', entity):
    #             tag, NE = entity.split('-')
    #         elif entity in ['B', 'I']:
    #             tag, NE = entity, None
    #
    #         else:
    #             raise ValueError('cannot process the BIO pattern.')
    #
    #         if tag == 'B':
    #             if last:
    #                 result.append(last)
    #             last = token
    #             last['NE'] = NE
    #
    #         else:  # tag == 'I'
    #             last['end'] = token['end']
    #
    #     if last:
    #         result.append(last)
    #
    #     return result
    #



    @classmethod
    def default_configs(cls):
        r"""This defines a basic config structure for ZeroShotClassifier.

        Following are the keys for this dictionary:
            - `entry_type`: defines which entry type in the input pack to make
              prediction on. The default makes prediction on each `Sentence`
              in the input pack.
            - `attribute_name`: defines which attribute of the `entry_type`
              in the input pack to save prediction to. The default
              saves prediction to the `classification` attribute for each
              `Sentence` in the input pack.
            - `multi_class`: whether to allow multiple class true
            - `model_name`: language model, default is
              `"valhalla/distilbart-mnli-12-1"`.
              The wrapper supports Hugging Face models with pipeline tag of
              `zero-shot-classification`.
            - `candidate_labels`: The set of possible class labels to
              classify each sequence into. Can be a single label, a string of
              comma-separated labels, or a list of labels. Note that for the
              model with a specific language, the candidate_labels need to
              be of that language.
            - `hypothesis_template`: The template used to turn each label
              into an NLI-style hypothesis. This template must include a {}
              or similar syntax for the candidate label to be inserted into
              the template. For example, the default
              template is :obj:`"This example is {}."` Note that for the
              model with a specific language, the hypothesis_template need to
              be of that language.
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
                "attribute_name": "ner", # "pos"
                'strategy': 'no_merge', # 'bio-merge'
                "model_name": 'dslim/bert-base-NER',
                "tokenizer": 'dslim/bert-base-NER',
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
        r"""Method to add output type record of `ZeroShotClassifier` which is
        user specified entry type with user specified attribute name
        to :attr:`forte.data.data_pack.Meta.record`.

        Args:
            record_meta: the field in the datapack for type record that need to
                fill in for consistency checking.
        """

        if self.configs.output_entry_type not in record_meta.keys():
            record_meta[self.configs.output_entry_type] = set()
