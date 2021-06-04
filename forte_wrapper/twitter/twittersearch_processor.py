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

import os
import tweepy as tw
import argparse
import os

import yaml

from forte.common.configuration import Config

# pylint: disable=attribute-defined-outside-init
from typing import Dict, Any

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.ontology.top import Query
from forte.processors.base import MultiPackProcessor
from forte.processors.base import PackProcessor
from ft.onto.base_ontology import Document

import os
from typing import Any, Iterator

from forte.data.data_pack import DataPack
from forte.data.data_utils_io import dataset_path_iterator
from forte.data.base_reader import PackReader
from ft.onto.base_ontology import Document

__all__ = [
    "TweetSearchProcessor",
]


class TweetSearchProcessor(MultiPackProcessor):
    """
    TweetSearchProcessor is designed to query tweets with Twitter API.
    Tweets will be returned as datapacks in input multipack.
    """

    def __init__(self):
        super().__init__()
        # tweets = self.query_tweets()

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        """This defines a basic config structure for ElasticSearchProcessor
        Returns:
            A dictionary with the default config for this processor.
            query_pack_name (str): The query pack's name, default is "query".
            index_config (dict): The ElasticSearchIndexer's config.
            field (str): Field name that will be used when creating the new
                datapack.
            response_pack_name_prefix (str): the pack name prefix to be used
                in response datapacks.
            indexed_text_only (bool): defines whether the returned
                value from the field (as specified by the field
                configuration) will be considered as plain text. If True,
                a new data pack will be created and the value will be
                used as the text for the data pack. Otherwise, the returned
                value will be considered as serialized data pack, and the
                returned data pack will be created by deserialization.
                Default is True.
        """
        config = super().default_configs()
        config.update({
            "credential_file": "",
            "tweet_items": 5,
            "lang": "en",
            "date_since": "2020-01-01",
            "result_type": 'recent',
            "query_pack_name": "query",
            "response_pack_name_prefix": "passage"
        })
        return config

    def _process(self, input_pack: MultiPack):
        r"""Searches `Elasticsearch` indexer to fetch documents for a query.
        This query should be contained in the input multipack with name
        `self.config.query_pack_name`.
        This method adds new packs to `input_pack` containing the retrieved
        results. Each result is added as a `ft.onto.base_ontology.Document`.
        Args:
             input_pack: A multipack containing query as a pack.
        """
        query_pack = input_pack.get_pack(self.configs.query_pack_name)

        query = query_pack.text
        tweets = self.query_tweets(query)

        for idx, tweet in enumerate(tweets):
            try:
                text = tweet.retweeted_status.full_text

            except AttributeError:  # Not a Retweet
                text = tweet.full_text

            pack: DataPack = input_pack.add_pack(
                f"{self.configs.response_pack_name_prefix}_{idx}"
            )
            pack.pack_name = f"{self.configs.response_pack_name_prefix}_{idx}"

            pack.set_text(text)

            Document(pack=pack, begin=0, end=len(text))


    def query_tweets(self, query):
        credentials = yaml.safe_load(open(self.configs.credential_file, "r"))
        credentials = Config(credentials, default_hparams=None)

        auth = tw.OAuthHandler(credentials.consumer_key,
                               credentials.consumer_secret)
        auth.set_access_token(credentials.access_token,
                              credentials.access_token_secret)

        api = tw.API(auth, wait_on_rate_limit=True)

        # Collect tweets
        tweets = tw.Cursor(api.search,
                           q=query,
                           lang=self.configs.lang,
                           since=self.configs.date_since,
                           result_type=self.configs.result_type,
                           tweet_mode="extended").items(self.configs.tweet_items)

        return tweets
