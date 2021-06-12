from ddt import ddt
import unittest
from forte.data.multi_pack import MultiPack
from forte.data.readers import MultiPackTerminalReader
from forte.pipeline import Pipeline
from forte.tweepy import TweetSearchProcessor


@ddt
class TestTweetSearchProcessor(unittest.TestCase):
    def setUp(self):
        self.nlp: Pipeline[MultiPack] = Pipeline()
        self.nlp.set_reader(reader=MultiPackTerminalReader())

    def test_init(self):
        self.nlp.add(TweetSearchProcessor())
        self.nlp.initialize()