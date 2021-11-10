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

import itertools
import logging
from typing import Any, Dict, Iterator, List, Set
import more_itertools

from transformers import BertTokenizer
from allennlp.predictors import Predictor
from ft.onto.base_ontology import (
    Token,
    Dependency,
    PredicateLink,
    PredicateArgument,
    PredicateMention,
)
from forte.common import ProcessorConfigError
from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_pack import DataPack, as_entry_type
from forte.processors.base import PackProcessor

from fortex.allennlp.utils_processor import (
    parse_allennlp_srl_tags,
    parse_allennlp_srl_results,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AllenNLPProcessor",
]

# pylint: disable=line-too-long
MODEL2URL = {
    "stanford": "https://storage.googleapis.com/allennlp-public-models"
    "/biaffine-dependency-parser-ptb-2020.04.06.tar.gz",
    "srl": "https://storage.googleapis.com/allennlp-public-models/bert-base"
    "-srl-2020.11.19.tar.gz",
    # TODO: The UD model seems to be broken at this moment.
    "universal": "https://storage.googleapis.com/allennlp-public-models"
    "/biaffine-dependency-parser-ud-2020.02.10.tar.gz",
}


class AllenNLPProcessor(PackProcessor):
    """
    Processor for AllenNLP predictors. It currently supports POS tagging
    (`pos`), Dependency Parsing (`depparse`), Semantic Role Labeling (`srl`).
    Check the `default_configs` method on how to specify the behavior of this
    processor.

    There are a few assumptions for this processor:
      - Sentences should be pre-annotated. The sentences can be stored in the
          provided type via the "sentence_type" configuration.
      - Long sentences will be skipped if they exceed the Bert model limit.
      - If unhandled runtime exception happens during predicting a batch, the
          results of the whole batch may be discarded.
    """

    # pylint: disable=attribute-defined-outside-init,unused-argument
    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        if (
            "pos" in configs.processors
            or "depparse" in configs.processors
            or "depparse" in configs.processors
            or "srl" in configs.processors
        ):
            if "tokenize" not in self.configs.processors:
                raise ProcessorConfigError(
                    "tokenize is necessary in "
                    "configs.processors for "
                    "pos, depparse or srl"
                )
        cuda_devices = itertools.cycle(configs["cuda_devices"])
        if configs.tag_formalism not in MODEL2URL:
            raise ProcessorConfigError("Incorrect value for tag_formalism")

        self.predictor = {}
        if configs.tag_formalism == "stanford":
            self.predictor = {
                "stanford": Predictor.from_path(
                    configs["stanford_url"], cuda_device=next(cuda_devices)
                )
            }
        if "srl" in configs.processors:
            # A parser seems to be needed for SRL.
            if "stanford" not in self.predictor:
                self.predictor["stanford"] = Predictor.from_path(
                    configs["stanford_url"], cuda_device=next(cuda_devices)
                )
            self.predictor["srl"] = Predictor.from_path(
                configs["srl_url"], cuda_device=next(cuda_devices)
            )

        if configs.overwrite_entries:
            logger.warning(
                "`overwrite_entries` is set to True, this means "
                "that the entries of the same type as produced by "
                "this processor will be overwritten if found."
            )
            if configs.allow_parallel_entries:
                logger.warning(
                    "Both `overwrite_entries` (whether to overwrite"
                    " the entries of the same type as produced by "
                    "this processor) and "
                    "`allow_parallel_entries` (whether to allow "
                    "similar new entries when they already exist) "
                    "are True, all existing conflicting entries "
                    "will be deleted."
                )
        else:
            if not configs.allow_parallel_entries:
                logger.warning(
                    "Both `overwrite_entries` (whether to overwrite"
                    " the entries of the same type as produced by "
                    "this processor) and "
                    "`allow_parallel_entries` (whether to allow "
                    "similar new entries when they already exist) "
                    "are False, processor will only run if there "
                    "are no existing conflicting entries."
                )

        self._sentence_type = as_entry_type(self.configs.sentence_type)

        # Use to estimate the length of the input sentence to avoid sub-word
        # go out of bound. Will be lazy initialized.
        self._subword_tokenizer = None

    @classmethod
    def default_configs(cls):
        """
        This defines a basic config structure for AllenNLP.

        Following are the keys for this dictionary:

            - processors: a comma separated string defines what operations to
              be done on the sentences, supported predictors include `tokenize`,
              `pos`, `depparse` and `srl`.
              The default value is `tokenize,pos,depparse`.

            - tag_formalism: format of the POS tags and dependency parsing,
              can be `universal` or `stanford`, default value is `stanford`.

            - overwrite_entries: whether to overwrite the entries of the same
              type as produced by this processor, default value is False.

            - allow_parallel_entries: whether to allow similar new entries when
              they already exist, e.g. allowing new tokens with same spans,
              used only when `overwrite_entries` is False.

            - model_url: URL of the corresponding model, default URL(s) for
              `stanford`, `srl` and `universal` can be found in `MODEL2URL`.

            - cuda_devices: a list of integers indicating the available
              cuda/gpu devices that can be used by this processor. When
              multiple models are loaded, cuda devices are assigned in a
              round robin fashion. E.g. [0, -1] -> first model uses gpu 0
              but second model uses cpu.

            - infer_batch_size: maximum number of sentences passed in as a
              batch to model's predict function. A value <= 0 means no limit.

            - sentence_type: the entry type that represent the sentences. The
              processor will look for this entry type to conduct processing.
              Default is `ft.onto.base_ontology.Sentence`.

        Returns: A dictionary with the default config for this processor.
        """
        return {
            "processors": "tokenize,pos,depparse",
            "tag_formalism": "stanford",
            "overwrite_entries": False,
            "allow_parallel_entries": True,
            "stanford_url": MODEL2URL["stanford"],
            "srl_url": MODEL2URL["srl"],
            "universal_url": MODEL2URL["universal"],
            "cuda_devices": [-1],
            "infer_batch_size": 0,
            "sentence_type": "ft.onto.base_ontology.Sentence",
        }

    def _process(self, input_pack: DataPack):
        # handle existing entries
        self._process_existing_entries(input_pack)

        batch_size: int = self.configs["infer_batch_size"]
        batches: Iterator[Iterator]
        # Need a copy of the one-pass iterators to support a second loop on
        # them. All other ways around it like using `itertools.tee` and `list`
        # would require extra storage conflicting with the idea of using
        # iterators in the first place. `more_itertools.ichunked` uses
        # `itertools.tee` under the hood but our usage (reading iterators
        # in order) does not cause memory issues.
        batches_copy: Iterator[Iterator]
        if batch_size <= 0:
            batches = iter([input_pack.get(self._sentence_type)])
            batches_copy = iter([input_pack.get(self._sentence_type)])
        else:
            batches = more_itertools.ichunked(
                input_pack.get(self._sentence_type), batch_size
            )
            batches_copy = more_itertools.ichunked(
                input_pack.get(self._sentence_type), batch_size
            )

        for sentences, sentences_copy in zip(batches, batches_copy):
            inputs: List[Dict[str, str]] = []
            skips = []

            for s in sentences:
                t = s.text.strip()
                if not t == "":
                    inputs.append({"sentence": t})
                    skips.append(False)
                else:
                    # Use placeholder, replace white space with dot.
                    inputs.append({"sentence": "." * len(s.text)})
                    # This result is useless, skip it.
                    skips.append(True)

            # Handle when the data is empty.
            if len(inputs) == 0:
                continue

            try:
                raw_results: Dict[str, List[Dict[str, Any]]] = {
                    k: self._predict_with_exception_retry(k, p, inputs)
                    for k, p in self.predictor.items()
                }
            except RuntimeError:
                for i, s in enumerate(skips):
                    skips[i] = True
                logging.exception(
                    "Encounter RuntimeError while performing prediction. "
                    "This batch will be skipped."
                )
                continue

            for i, sent in enumerate(sentences_copy):
                if skips[i]:
                    continue

                result: Dict[str, List[str]] = {}
                for key in self.predictor:
                    raw_result = raw_results[key]

                    if raw_result is not None:
                        if key == "srl":
                            result.update(
                                parse_allennlp_srl_results(
                                    raw_result[i]["verbs"]
                                )
                            )
                        else:
                            result.update(raw_result[i])

                if "tokenize" in self.configs.processors:
                    # creating new tokens and dependencies
                    tokens = self._create_tokens(input_pack, sent, result)
                    if "depparse" in self.configs.processors:
                        self._create_dependencies(input_pack, tokens, result)
                    if "srl" in self.configs.processors:
                        self._create_srl(input_pack, tokens, result)

    def _predict_with_exception_retry(
        self, model_name: str, predictor: Predictor, inputs
    ):
        """
        Run the predictor on the inputs. If a RuntimeError is thrown, try to
        check if it is caused by long sentences in SRL models. Try to remove
        those sentence and run again. But this step may still fail if there
        are other causes.
        """
        try:
            return predictor.predict_batch_json(inputs)
        except RuntimeError:
            if model_name == "srl":
                if self._subword_tokenizer is None:
                    self._subword_tokenizer = BertTokenizer.from_pretrained(
                        "bert-base-uncased"
                    )

                # One possible cause of RuntimeError is the input is too long,
                # See if we can fix this by retrying.
                for item in inputs:
                    subword_length = len(
                        self._subword_tokenizer(item["sentence"]).data[
                            "input_ids"
                        ]
                    )

                    if subword_length > 450:
                        # Remove long sentences that could create problems.
                        # Since the AllenNLP SRL uses Bert with a limit of 512,
                        # but we choose a smaller number to ensure we can get
                        # through it.
                        # We make this sentence to empty and skip it.
                        item["sentence"] = "."

                # Try again with the new input, hopefully it can pass.
                return predictor.predict_batch_json(inputs)

    def _process_existing_entries(self, input_pack):
        tokens_exist = any(True for _ in input_pack.get(Token))
        dependencies_exist = any(True for _ in input_pack.get(Dependency))

        if tokens_exist or dependencies_exist:
            if not self.configs.overwrite_entries:
                if not self.configs.allow_parallel_entries:
                    raise ProcessorConfigError(
                        "Found existing entries, either `overwrite_entries` or"
                        " `allow_parallel_entries` should be True"
                    )
            else:
                # delete existing tokens and dependencies
                for entry_type in (Token, Dependency):
                    for entry in list(input_pack.get(entry_type)):
                        input_pack.delete_entry(entry)

    def _create_tokens(self, input_pack, sentence, result):
        words, pos = result["words"], result["pos"]
        tokens = []
        offset = sentence.span.begin
        word_end = 0
        for i, word in enumerate(words):
            word_begin = sentence.text.find(word, word_end)
            word_end = word_begin + len(word)
            token = Token(input_pack, offset + word_begin, offset + word_end)
            if "pos" in self.configs.processors:
                token.pos = pos[i]
            tokens.append(token)

        return tokens

    @staticmethod
    def _create_dependencies(input_pack, tokens, result):
        deps = result["predicted_dependencies"]
        heads = result["predicted_heads"]
        for i, token in enumerate(tokens):
            relation = Dependency(
                input_pack, parent=tokens[heads[i] - 1], child=token
            )
            relation.rel_type = deps[i]

    @staticmethod
    def _create_srl(
        input_pack: DataPack, tokens: List[Token], result: Dict[str, List[str]]
    ) -> None:
        for _, tag in enumerate(result["srl_tags"]):
            pred_span, arguments = parse_allennlp_srl_tags(tag)
            if not pred_span:
                continue
            pred = PredicateMention(
                input_pack,
                tokens[pred_span.begin].begin,
                tokens[pred_span.end].end,
            )
            for arg_span, label in arguments:
                arg = PredicateArgument(
                    input_pack,
                    tokens[arg_span.begin].begin,
                    tokens[arg_span.end].end,
                )
                link = PredicateLink(input_pack, pred, arg)
                link.arg_type = label

    def expected_types_and_attributes(self) -> Dict[str, Set[str]]:
        r"""Method to add expected type for current processor input which
        would be checked before running the processor if
        the pipeline is initialized with
        `enforce_consistency=True` or
        :meth:`~forte.pipeline.Pipeline.enforce_consistency` was enabled for
        the pipeline.
        """
        expectation_dict: Dict[str, Set[str]] = {
            self.configs.sentence_type: set()
        }
        return expectation_dict

    def record(self, record_meta: Dict[str, Set[str]]):
        r"""Method to add output type record of current processor
        to :attr:`forte.data.data_pack.Meta.record`.

        Args:
            record_meta: the field in the datapack for type record that need to
                fill in for consistency checking.
        """
        if "tokenize" in self.configs.processors:
            record_meta["ft.onto.base_ontology.Token"] = set()
            if "pos" in self.configs.processors:
                record_meta["ft.onto.base_ontology.Token"].add("pos")
            if "depparse" in self.configs.processors:
                record_meta["ft.onto.base_ontology.Dependency"] = {"rel_type"}
            if "srl" in self.configs.processors:
                record_meta["ft.onto.base_ontology.PredicateArgument"] = set()
                record_meta["ft.onto.base_ontology.PredicateMention"] = set()
                record_meta["ft.onto.base_ontology.PredicateLink"] = {
                    "arg_type"
                }
