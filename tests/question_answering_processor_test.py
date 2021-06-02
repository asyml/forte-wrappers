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

from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.data.readers import StringReader
from forte_wrapper.hugginface.question_and_answering_single \
    import QuestionAnsweringSingle
from ft.onto.base_ontology import Phrase


class TestQuestionAnswering(unittest.TestCase):
    def setUp(self):
        self.nlp = Pipeline[DataPack](enforce_consistency=True)
        self.nlp.set_reader(StringReader())
        config = {
            "use_gpu": False,
            'question': "What is the molecular function of"
                        " psoralen photobinding on DNA?"
        }
        self.nlp.add(QuestionAnsweringSingle(), config=config)
        self.nlp.initialize()

    def test_huggingface_qa_processor(self):
        sentences = ["Photoreaction of furocoumarins with DNA is similarly ",
                     "inhibited by minor and major groove-interacting ligands.",
                     " It was found that methyl green, a major groove binding ",
                     "ligand and the minor groove binding ligands, netropsin ",
                     "and 2,7-di-tert-butylproflavine inhibit, to a similar ",
                     "extend a monoadduct forming benzopsoralen and monoadduct",
                     " and diadduct forming derivatives of psoralen ",
                     "(8-methoxypsoralen and 3,4'-dimethyl-8-methoxypsoralen).",
                     " Caffeine exhibits an inhibitory effect on furocoumarin ",
                     "photobinding to DNA at 10(3) fold higher concentration. ",
                     "Together with the previously published results it is ",
                     "concluded that both occupancy of the major and minor ",
                     "groove as well as intercalation hinder photobinding of ",
                     "furocoumarins to DNA."]
        document = ''.join(sentences)
        pack = self.nlp.process(document)
        expected_ans = {'start': 682, 'end': 688, 'answer': 'hinder'}
        for idx, phrase in enumerate(pack.get(entry_type=Phrase)):
            self.assertEqual(phrase.text,
                             expected_ans['answer'])


if __name__ == "__main__":
    unittest.main()
