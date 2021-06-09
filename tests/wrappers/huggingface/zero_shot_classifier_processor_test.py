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
Unit tests for ZeroShotClassifier processor.
"""
import unittest

from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.data.readers import StringReader
from forte.nltk import NLTKSentenceSegmenter
from forte.huggingface.zero_shot_classifier import ZeroShotClassifier
from ft.onto.base_ontology import Sentence
from helpers.test_utils import get_top_scores_label


class TestZeroShotClassifier(unittest.TestCase):
    def setUp(self):
        self.nlp = Pipeline[DataPack](enforce_consistency=True)
        self.nlp.set_reader(StringReader())
        self.nlp.add(NLTKSentenceSegmenter())
        self.nlp.add(ZeroShotClassifier())
        self.nlp.initialize()

    def test_huggingface_zero_shot_processor(self):
        sentences = [
            "One day I will see the world.",
            "I will try out all types of the delicious cuisine!",
        ]
        document = " ".join(sentences)
        pack = self.nlp.process(document)

        # sentence: Sentence
        expected_scores = [
            {
                "travel": 0.6254,
                "exploration": 0.1733,
                "cooking": 0.0009,
                "dancing": 0.0008,
            },
            {
                "exploration": 0.89,
                "cooking": 0.5343,
                "travel": 0.0069,
                "dancing": 0.0016,
            },
        ]
        expected_tops = [get_top_scores_label(x) for x in expected_scores]
        for idx, sentence in enumerate(pack.get(Sentence)):
            self.assertEqual(
                get_top_scores_label(sentence.classification),
                expected_tops[idx],
            )


if __name__ == "__main__":
    unittest.main()
