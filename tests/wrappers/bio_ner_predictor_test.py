import sys
import time
import os
import yaml
import logging
from requests import HTTPError
from bio_ner_predictor.mimic3_note_reader import Mimic3DischargeNoteReader

from fortex.elastic import ElasticSearchPackIndexProcessor
from fortex.huggingface.bio_ner_predictor import BioBERTNERPredictor
from fortex.huggingface.transformers_processor import BERTTokenizer

from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.processors.writers import PackIdJsonPackWriter
from fortex.nltk import NLTKSentenceSegmenter
import unittest
from ddt import ddt, data, unpack
from forte.data.data_utils import maybe_download
from ft.onto.base_ontology import EntityMention

@ddt
class TestBioNerPredictor(unittest.TestCase):
    r"""Tests Elastic Indexer."""

    def setUp(self):
        self.pl = Pipeline[DataPack]()

        script_dir_path = os.path.dirname(os.path.abspath(__file__))
        data_folder = "bio_ner_predictor"
        self.output_path = os.path.join(script_dir_path,data_folder, "test_case_output/")
        config_path = os.path.join(script_dir_path,data_folder,"bio_ner_config.yml")
        self.input_path = os.path.join(script_dir_path,data_folder, "D_ICD_DIAGNOSES.csv")
        self.num_packs = 5

        # download resources
        urls = [
            "https://drive.google.com/file/d/15RSfFkW9syQKtx-_fQ9KshN3BJ27Jf8t/"
            "view?usp=sharing",
            "https://drive.google.com/file/d/1Nh7D6Xam5JefdoSXRoL7S0DZK1d4i2UK/"
            "view?usp=sharing",
            "https://drive.google.com/file/d/1YWcI60lGKtTFH01Ai1HnwOKBsrFf2r29/"
            "view?usp=sharing",
            "https://drive.google.com/file/d/1ElHUEMPQIuWmV0GimroqFphbCvFKskYj/"
            "view?usp=sharing",
            "https://drive.google.com/file/d/1EhMXlieoEg-bGUbbQ2vN-iyNJvC4Dajl/"
            "view?usp=sharing",
        ]

        filenames = [
            "config.json",
            "pytorch_model.bin",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "vocab.txt",
        ]
        model_path = os.path.abspath("resources/NCBI-disease")
        config = yaml.safe_load(open(config_path, "r"))
        config = Config(config, default_hparams=None)
        config.BERTTokenizer.model_path = model_path
        config.BioBERTNERPredictor.model_path = model_path
        try:
            maybe_download(
                urls=urls,
                path=model_path,
                filenames=filenames,
                num_gdrive_retries=2
            )
        except HTTPError as e:
            if e.response.status_code != 403:
                raise e
            logging.warning(
                "Skipping HTTPError from %s", e.response.url, exc_info=True
            )
            return
        self.assertTrue(os.path.exists(os.path.join(model_path, "pytorch_model.bin")))
        self.pl.set_reader(
            Mimic3DischargeNoteReader(), config={"max_num_notes": self.num_packs}
        )
        self.pl.add(NLTKSentenceSegmenter())
        

        
        self.pl.add(BERTTokenizer(), config=config.BERTTokenizer)
        self.pl.add(BioBERTNERPredictor(), config=config.BioBERTNERPredictor)
        self.pl.add(ElasticSearchPackIndexProcessor())
        self.pl.add(
            PackIdJsonPackWriter(),
            {
                "output_dir": self.output_path,
                "indent": 2,
                "overwrite": True,
                "drop_record": True,
                "zip_pack": True,
            },
        )
        self.pl.initialize()

    def test_predict(self):
        for idx, data_pack in enumerate(self.pl.process_dataset(self.input_path)):
            ems = list(data_pack.get_data(EntityMention))
            self.assertTrue(len(ems) > 0)

        self.assertEqual(len(os.listdir(self.output_path)), self.num_packs)
        for f_name in os.listdir(self.output_path):
            os.remove(os.path.join(self.output_path, f_name))
        os.removedirs(self.output_path)
