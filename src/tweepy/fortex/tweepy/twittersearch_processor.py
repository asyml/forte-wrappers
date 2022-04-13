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
from typing import Dict, Any
import yaml

from ft.onto.base_ontology import Document
import tweepy as tw

from forte.common.configuration import Config
from forte.data.multi_pack import MultiPack
from forte.processors.base import MultiPackProcessor
from forte.data.data_pack import DataPack

__all__ = [
    "TweetSearchProcessor",
]


class TweetSearchProcessor(MultiPackProcessor):
    """
    TweetSearchProcessor is designed to query tweets with Tweepy and
    Twitter API.
    Tweets will be returned as datapacks in input multipack.
    """

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        # pylint: disable=line-too-long
        """This defines a basic config structure for TweetSearchProcessor.
        For more details about the parameters, refer to
        https://docs.tweepy.org/en/latest/api.html#tweepy.API.search_tweets
        and
        https://developer.twitter.com/en/docs/twitter-api/v1/tweets/search/api-reference/get-search-tweets

        Returns:
            A dictionary with the default config for this processor.

        Following are the keys for this dictionary:

            - `"credential_file"`:
                Defines the path of credential file needed for Twitter API usage.

            - `"num_tweets_returned"`:
                Defines the number of tweets returned by processor.

            - `"lang"`:
                Language, restricts tweets to the given language, default is 'en'.

            - `"date_since"`:
                Restricts tweets created after the given date.

            - `"result_type"`:
                Defines what type of search results to receive. The default is “recent.”
                Valid values include:

                mixed : include both popular and real time results in the response

                recent : return only the most recent results in the response

                popular : return only the most popular results in the response.

            - `"query_pack_name"`:
                The query pack's name, default is "query".

            - `"response_pack_name_prefix"`:
                The pack name prefix to be used in response data packs.
        """
        # pylint: enable=line-too-long
        return {
            "credential_file": "",
            "num_tweets_returned": 5,
            "lang": "en",
            "date_since": "2020-01-01",
            "result_type": "recent",
            "query_pack_name": "query",
            "response_pack_name_prefix": "passage",
        }

    def _process(self, input_pack: MultiPack):
        r"""Search using Twitter API to fetch tweets for a query.
        This query should be contained in the input multipack with name
        `self.config.query_pack_name`.
        Each result is added as a new data pack, and a
        `ft.onto.base_ontology.Document` annotation is used to cover the whole
        document.

        Args:
             input_pack: A multipack containing query as a pack.
        """
        query_pack = input_pack.get_pack(self.configs.query_pack_name)

        query = query_pack.text
        tweets = self._query_tweets(query)
        for idx, tweet in enumerate(tweets.data):
            if tweet.lang == self.configs.lang:
                txt = tweet.text
            else:
                pass
                # skip if the tweet in not in desired language
            pack: DataPack = input_pack.add_pack(
                f"{self.configs.response_pack_name_prefix}_{idx}"
            )
            pack.pack_name = f"{self.configs.response_pack_name_prefix}_{idx}"
            pack.set_text(txt)
            Document(pack=pack, begin=0, end=len(txt))

    def _query_tweets(self, query: str):
        """
        This function searches tweets using Tweepy.

        Args:
            query: user's input query for twitter API search

        Returns:
            List of tweets
        """
        with open(self.configs.credential_file, "r", encoding="utf-8") as f:
            credentials = yaml.safe_load(f)
            credentials = Config(credentials, default_hparams=None)
            api = tw.Client(
                bearer_token=credentials.bearer_token,
            )  # type: ignore
            # Collect tweets
            tweets = api.search_recent_tweets(
                query=query,
                tweet_fields=["context_annotations", "created_at", "lang"],
                max_results=10,
            )
            return tweets
