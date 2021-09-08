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
Unit tests for QuestionAnswering processors.
"""
import unittest
from forte.pipeline import Pipeline
from forte.data.caster import MultiPackBoxer
from forte.data.readers import StringReader
from forte.data.multi_pack import MultiPack, MultiPackLink
from forte.processors.base import MultiPackProcessor
from forte.nltk import NLTKSentenceSegmenter
from forte.huggingface.question_and_answering_multi import (
    QuestionAnsweringMulti,
)
from ft.onto.base_ontology import Phrase, Document

docs = {
    "doc_1": "Acrokeratosis paraneoplastica (Bazex syndrome): report of a case "
    "associated with small cell lung carcinoma and review of the "
    "literature. Acrokeratosis paraneoplastic (Bazex syndrome) is "
    "a rare, but distinctive paraneoplastic dermatosis characterized "
    "by erythematosquamous lesions located at the acral sites and "
    "is most commonly associated with carcinomas of the upper "
    "aerodigestive tract. We report a 58-year-old female with a "
    "history of a pigmented rash on her extremities, thick keratotic "
    "plaques on her hands, and brittle nails. Chest imaging revealed "
    "a right upper lobe mass that was proven to be small cell lung "
    "carcinoma. While Bazex syndrome has been described in the "
    "dermatology literature, it is also important for the radiologist "
    "to be aware of this entity and its common presentations.",
    "doc_2": "Bazex syndrome (acrokeratosis paraneoplastica): persistence of "
    "cutaneous lesions after successful treatment of an associated "
    "oropharyngeal neoplasm. Acrokeratosis paraneoplastica is a rare "
    "paraneoplastic syndrome commonly affecting males over 40 years "
    "of age. There exists a strong association with squamous cell "
    "carcinoma (SCC) of the upper aerodigestive tract or cervical "
    "metastatic disease originating from an unknown primary. We "
    "report a case associated with SCC of the right tonsil with "
    "persistent paraneoplastic cutaneous lesions 2 years after "
    "successful treatment of the underlying neoplasm.",
    "doc_3": "Acrokeratosis paraneoplastica (Bazex syndrome) with oropharyngeal"
    " squamous cell carcinoma. A 65-year-old white man presented with "
    "all the clinical features of acrokeratosis paraneoplastica of"
    " Bazex, characterized by violaceous erythema and scaling of "
    "the nose, aural helices, fingers, and toes, with keratoderma "
    "and severe nail dystrophy. Examination of the patient for "
    "possible associated malignancy disclosed an asymptomatic "
    "squamous cell carcinoma at the oropharyngeal region. The skin "
    "lesions resolved almost completely following radiation therapy "
    "of the neoplasm, but the onychodystrophy persisted. This case "
    "report illustrates the importance of early recognition of Bazex "
    "syndrome.",
}


class MutliDocPackAdder(MultiPackProcessor):
    def _process(self, input_pack: MultiPack):
        for doc_i in docs:
            pack = input_pack.add_pack(ref_name=doc_i)
            pack.set_text(docs[doc_i])
            Document(pack, 0, len(pack.text))


class TestQuestionAnsweringMulti(unittest.TestCase):
    def setUp(self):
        self.nlp = Pipeline()
        self.nlp.set_reader(StringReader())
        self.nlp.add(NLTKSentenceSegmenter())
        boxer_config = {"pack_name": "question"}
        self.nlp.add(MultiPackBoxer(), boxer_config)
        self.nlp.add(MutliDocPackAdder())
        self.nlp.add(QuestionAnsweringMulti())
        self.nlp.initialize()

    def test_huggingface_qa_multi_processor(self):
        question = "Name synonym of Acrokeratosis paraneoplastica."
        packs: MultiPack = self.nlp.process(question)
        expected_ans = {
            "doc_1": "Bazex syndrome",
            "doc_2": "Bazex syndrome",
            "doc_3": "Bazex syndrome",
        }
        for doc_id in packs.pack_names:
            if doc_id == "question":
                continue
            pack = packs.get_pack(doc_id)
            for idx, phrase in enumerate(pack.get(entry_type=Phrase)):
                self.assertEqual(phrase.text, expected_ans[doc_id])

        linked_texts = []

        for link in packs.get(entry_type=MultiPackLink):
            parent_text = link.get_parent().text
            child_text = link.get_child().text
            linked_texts.append((parent_text, child_text))

        self.assertListEqual(
            sorted(linked_texts),
            sorted(
                [
                    (question, expected_ans["doc_1"]),
                    (question, expected_ans["doc_2"]),
                    (question, expected_ans["doc_3"]),
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
