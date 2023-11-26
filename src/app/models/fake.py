import logging
from pathlib import Path

from .model import Model, ModelInfo

logger = logging.getLogger(__name__)

# These are series of answers based on the prompt matching a string
_prompt_answer = {
    'blah': 'blah blah blah'
}


class FakeModel(Model):
    """ 
    FakeModel generates strings based on the _prompt_answer
    """
    def __init__(self, model_info: ModelInfo):
        super().__init__(model_info=model_info)

    def decode(self, y):
        """ Fake decoder, just returns the input """
        return y

    def encode(self, text: str):
        """ Fake encoder, just returns the input """
        return text

    def _generate(self, prompt_tokenized, **kwargs):
        prompt = prompt_tokenized.lower()
        for p, a in _prompt_answer.items():
            if prompt in p.lower():
                return a
        return prompt + "...I don't know what to do with this."

    def load(self):
        """ Nothing to load """
        pass
    
