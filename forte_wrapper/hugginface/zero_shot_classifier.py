import importlib
from forte.common import Resources
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from transformers import pipeline


class ZeroShotClassifier(PackProcessor):
    r"""Wrapper of the models on HugginFace platform
    https://huggingface.co/models?pipeline_tag=zero-shot-classification

    """

    def __init__(self):
        super().__init__()
        self.classifier = None

    def set_up(self):
        device_num = -1
        if self.configs.use_gpu:
            device_num = 0
        self.classifier = pipeline("zero-shot-classification",
                                   model=self.configs.model_name,
                                   framework='pt',
                                   device=device_num)

    def initialize(self, resources: Resources, configs: Config):
        # pylint: disable=unused-argument
        super().initialize(resources, configs)
        self.set_up()

    def _process(self, input_pack: DataPack):
        path_str, module_str = self.configs.entry_type.rsplit('.', 1)

        mod = importlib.import_module(path_str)
        entry = getattr(mod, module_str)
        for entry_specified in input_pack.get(entry_type=entry):
            result = self.classifier(
                sequences=entry_specified.text,
                candidate_labels=self.configs.candidate_labels,
                hypothesis_template=self.configs.hypothesis_template,
                multi_class=self.configs.multi_class)
            curr_dict = getattr(entry_specified, self.configs.attribute_name)
            for idx, lab in enumerate(result['labels']):
                curr_dict[lab] = round(result['scores'][idx], 4)
            setattr(entry_specified, self.configs.attribute_name, curr_dict)

    @classmethod
    def default_configs(cls):
        r"""This defines a basic config structure for VaderSentimentProcessor.

        sentence_component (str): If not None, the processor will process
          sentence with the provided component name. If None, then all sentences
          will be processed.
        """
        config = super().default_configs()
        config.update({
            'entry_type': 'ft.onto.base_ontology.Sentence',
            'attribute_name': 'classification',
            'multi_class': True,
            'model_name': 'valhalla/distilbart-mnli-12-1',
            'candidate_labels': ['travel', 'cooking', 'dancing', 'exploration'],
            'hypothesis_template': "This example is {}.",
            'use_gpu': False
        })
        return config

    @classmethod
    def expected_types_and_attributes(cls):
        r"""Method to add expected type `ft.onto.base_ontology.Sentence` which
        would be checked before running the processor if
        the pipeline is initialized with
        `enforce_consistency=True` or
        :meth:`~forte.pipeline.Pipeline.enforce_consistency` was enabled for
        the pipeline.
        """
        return {cls().default_configs()["entry_type"]: set()}
