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

# pylint: disable=attribute-defined-outside-init
from typing import Dict, Any
import tweepy as tw
import yaml

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.multi_pack import MultiPack
from forte.processors.base import MultiPackProcessor
from forte.data.data_pack import DataPack
from ft.onto.base_ontology import Document

__all__ = [
    "TweetSearchProcessor",
]


class TweetSearchProcessor(MultiPackProcessor):
    """
    TweetSearchProcessor is designed to query tweets with Twitter API.
    Tweets will be returned as datapacks in input multipack.
    """
    # pylint: disable=useless-super-delegation
    def __init__(self):
        super().__init__()

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        # pylint: disable=line-too-long
        """This defines a basic config structure for TweetSearchProcessor.
        For more details about the parameters, refer to
        https://docs.tweepy.org/en/latest/api.html#tweepy.API.search_tweets
        and
        https://developer.twitter.com/en/docs/twitter-api/v1/tweets/search/api-reference/get-search-tweets

        Returns:
            dictionary with the default config for this processor.
        Following are the keys for this dictionary:
        - `"credential_file"`: defines the path of credential file needed for
            Twitter API usage.
        - `"tweet_items"`: defines the number of tweets returned by processor.
        - `"lang"`: language, restricts tweets to the given language,
            default is 'en'.
        - `"date_since"`: restricts tweets created after the given date.
        - `"result_type"`: defines what type of search results to receive.
            The default is “recent.” Valid values include:
            mixed : include both popular and real time results in the response
            recent : return only the most recent results in the response
            popular : return only the most popular results in the response
        - `"query_pack_name"`: The query pack's name, default is "query".
        - `"response_pack_name_prefix"`: the pack name prefix to be used
                in response datapacks.
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
        r"""Search using Twitter API to fetch tweets for a query.
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

    def query_tweets(self, query: str):
        """
        This function searches tweets using Twitter API
        Args:
            query: user's input query for twitter API search
        Returns: List of tweets
        """
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
                           tweet_mode="extended").items(
            self.configs.tweet_items)

        return tweets
