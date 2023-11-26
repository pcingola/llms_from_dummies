import logging

import torch
import lightning as L
from torch import nn

from ..config import RANDOM_SEED

logger = logging.getLogger(__name__)

class ModelInfo:
    """ Model information """
    id: str
    owned_by: str
    permissions: list[str]

    def __init__(self, id: str, owned_by: str, permissions: list[str]):
        self.id = id
        self.owned_by = owned_by
        self.permissions = permissions
        logger.info(f"Creating model info '%s'", self.id)


# TODO: Refactor / Subclass to LlmModel instead?
class Model:
    """ 
    Model: Puts toghether a ModelInfo and a nn.Module 
    Can have a generate function
    """
    model_info: ModelInfo = None
    """ Model information """
    model_nn: nn.Module = None
    """ Model neural network """
    tokenizer = None
    """ Model tokenizer """
    generate = None
    """ Model's `generate()` function """

    def __init__(self, model_info: ModelInfo):
        self.model_info = model_info
        logger.info(f"Creating model '%s'", self.model_info.id)

    def decode(self, tokens: torch.Tensor) -> str:
        """ Decode a vector """
        if self.tokenizer is not None:
            return self.tokenizer.decode(tokens)
        else:
            raise NotImplementedError(f"Function 'decode(...)' was not assigned to the model")
        
    def encode(self, text: str) -> torch.Tensor:
        """ Tokenize a string """
        if self.tokenizer is not None:
            return self.tokenizer.tokenize(text)
        else:
            raise NotImplementedError(f"Function 'encode(...)' was not assigned to the model")

    def generate(self, prompt: str
                 , num_samples: int = 1
                 , **kwargs) -> list[str]:
        """ 
        Generate a string from the model

        Arguments: The arguments are expanded from api.CompletionRequest

        FIXME: Make sure there is only one model running at a time
        Reference: https://github.com/tiangolo/fastapi/issues/4458
        Simple solution: Search for `api.lock` approach
        Advanced solution: Search for `custom router` approach
        """
        outs = []
        prompt_tokenized = self.encode(prompt)
        assert num_samples > 0, "Error: 'num_samples' must be greater than 0"
        if RANDOM_SEED:
            L.seed_everything(RANDOM_SEED)
            logging.debug(f"Using random seed {RANDOM_SEED}")
        for i in range(num_samples):
            logging.debug(f"Generating sample {i} / {num_samples}")
            y = self._generate(prompt_tokenized=prompt_tokenized, **kwargs)
            out = self.decode(y)
            outs.append(out)
        # if self.fabric.device.type == "cuda":
        #     print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)
        return out

    def _generate(self, prompt_tokenized, **kwargs):
        """ Generate a string from the model using a tokenized prompt """
        raise NotImplementedError(f"Function '_generate(...)' was not assigned to the model")

    def load(self):
        """ Load the model """
        raise NotImplementedError(f"Function 'load(...)' was not assigned to the model")

