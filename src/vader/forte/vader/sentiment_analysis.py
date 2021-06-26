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

__all__ = [
    "VaderSentimentProcessor",
]

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from forte.common import Resources
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor


class VaderSentimentProcessor(PackProcessor):
    r"""A wrapper of a sentiment analyzer: Vader (Valence Aware Dictionary
    and Sentiment Reasoner). Vader needs to be installed to use this package

     > `pip install vaderSentiment`

     or

     > `pip install --upgrade vaderSentiment`

    This processor will add assign sentiment label to each sentence in the
    document. If the input pack contains no sentence then no processing will
    happen. If the data pack has multiple set of sentences, one can specify
    the set of sentences to tag by setting the `sentence_component` attribute.

    Vader URL: (https://github.com/cjhutto/vaderSentiment)

    Citation: VADER: A Parsimonious Rule-based Model for Sentiment Analysis of
    Social Media Text (by C.J. Hutto and Eric Gilbert)

    """

    def __init__(self):
        super().__init__()
        self.sentence_component = None
        self.analyzer = SentimentIntensityAnalyzer()

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self.sentence_component = configs.get("sentence_component")

    def _process(self, input_pack: DataPack):
        for entry_specified in input_pack.get(
            entry_type=self.configs.entry_type,
            components=self.sentence_component,
        ):
            scores = self.analyzer.polarity_scores(entry_specified.text)
            setattr(entry_specified, self.configs.attribute_name, scores)

    @classmethod
    def default_configs(cls):
        r"""This defines a basic config structure for VaderSentimentProcessor.

        Returns:
            A dictionary with the default config for this processor.

        Following are the keys for this dictionary:

        - `"entry_type"`:
            Defines which entry type in the input pack to make
            prediction on. The default makes prediction on each `Sentence`
            in the input pack.

        - `"attribute_name"`:
            Defines which attribute of the `entry_type`
            in the input pack to save score to. The default saves prediction
            to the `sentiment` attribute for each `Sentence` in the input pack.

        - `"sentence_component"`:
            str. If not None, the processor will process sentence with the
            provided component name. If None, then all sentences will be
            processed.
        """
        config = super().default_configs()
        config.update(
            {
                "entry_type": "ft.onto.base_ontology.Sentence",
                "attribute_name": "sentiment",
                "sentence_component": None,
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
