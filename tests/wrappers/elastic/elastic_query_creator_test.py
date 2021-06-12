from ddt import ddt, data
import unittest
from forte.data.caster import MultiPackBoxer
from forte.elastic import ElasticSearchQueryCreator
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.data.readers import StringReader
from forte.data.ontology.top import Query


@ddt
class TestElasticSearchQueryCreator(unittest.TestCase):
    def setUp(self):
        self.nlp = Pipeline[DataPack]()
        self.nlp.set_reader(StringReader())
        self.nlp.add(MultiPackBoxer(), config={'pack_name': "query"})
        self.nlp.add(ElasticSearchQueryCreator(),
                     config={"size": 1,
                             "field": "content",
                             "query_pack_name": "query"
                             })
        self.nlp.initialize()

    @data("test")
    def test_process_query(self, query):
        m_pack = self.nlp.process_one([query])

        query_pack = m_pack.get_pack("query")
        test_query = query_pack.get_single(Query)

        self.assertEqual({'query': {'match': {'content': query}}, 'size': 1},
                         test_query.value)
