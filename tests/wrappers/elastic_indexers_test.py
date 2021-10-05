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
Unit tests for elastic indexer module.
"""

import time
import unittest

from ddt import ddt, data, unpack
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from fortex.elastic import ElasticSearchIndexer
from helpers.test_utils import performance_test


@ddt
class TestElasticSearchIndexer(unittest.TestCase):
    r"""Tests Elastic Indexer."""

    def setUp(self):
        self.indexer = ElasticSearchIndexer(
            configs={"index_name": "test_index"}
        )

    def tearDown(self):
        self.indexer.elasticsearch.indices.delete(
            index=self.indexer.hparams.index_name, ignore=[400, 404]
        )

    def test_add(self):
        document = {
            "key": "This document is created to test " "ElasticSearchIndexer"
        }
        self.indexer.add(document, refresh="wait_for")
        retrieved_document = self.indexer.search(
            query={
                "query": {"match": {"key": "ElasticSearchIndexer"}},
                "_source": ["key"],
            }
        )
        hits = retrieved_document["hits"]["hits"]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["_source"], document)

    def test_add_bulk(self):
        size = 10000
        documents = set(
            [
                f"This document {i} is created to test " f"ElasticSearchIndexer"
                for i in range(size)
            ]
        )
        self.indexer.add_bulk(
            [{"key": document} for document in documents], refresh="wait_for"
        )
        retrieved_document = self.indexer.search(
            query={"query": {"match_all": {}}},
            index_name="test_index",
            size=size,
        )
        hits = retrieved_document["hits"]["hits"]
        self.assertEqual(len(hits), size)
        results = set([hit["_source"]["key"] for hit in hits])
        self.assertEqual(results, documents)

    @performance_test
    @data([100, 0.3], [500, 0.3], [1000, 0.3])
    @unpack
    def test_speed(self, size, epsilon):
        es = Elasticsearch()
        documents = [
            {
                "_index": "test_index_",
                "_type": "document",
                "key": f"This document {i} is created to test "
                f"ElasticSearchIndexer",
            }
            for i in range(size)
        ]

        start = time.time()
        bulk(es, documents, refresh=False)
        baseline = time.time() - start
        es.indices.delete(index="test_index_", ignore=[400, 404])

        documents = set(
            [
                f"This document {i} is created to test " f"ElasticSearchIndexer"
                for i in range(size)
            ]
        )
        start = time.time()
        self.indexer.add_bulk(
            [{"key": document} for document in documents], refresh=False
        )
        forte_time = time.time() - start
        self.assertLessEqual(forte_time, baseline + epsilon)


if __name__ == "__main__":
    unittest.main()
