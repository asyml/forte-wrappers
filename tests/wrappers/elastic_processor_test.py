import unittest
from ddt import ddt
from forte.data.multi_pack import MultiPack
from forte.data.readers import MultiPackTerminalReader
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.data.readers import StringReader
from forte.elastic import ElasticSearchProcessor, \
    ElasticSearchPackIndexProcessor, ElasticSearchTextIndexProcessor


@ddt
class TestElasticSearchProcessor(unittest.TestCase):
    def setUp(self):
        self.nlp: Pipeline[MultiPack] = Pipeline()
        self.nlp.set_reader(reader=MultiPackTerminalReader())

    def test_init(self):
        self.nlp.add(ElasticSearchProcessor())
        self.nlp.initialize()


@ddt
class TestElasticSearchPackIndexProcessor(unittest.TestCase):
    def setUp(self):
        self.nlp = Pipeline[DataPack]()
        self.nlp.set_reader(StringReader())

    def test_init(self):
        self.nlp.add(ElasticSearchPackIndexProcessor())
        self.nlp.initialize()


@ddt
class TestElasticSearchTextIndexProcessor(unittest.TestCase):
    def setUp(self):
        self.nlp = Pipeline[DataPack]()
        self.nlp.set_reader(StringReader())

    def test_init(self):
        self.nlp.add(ElasticSearchTextIndexProcessor())
        self.nlp.initialize()