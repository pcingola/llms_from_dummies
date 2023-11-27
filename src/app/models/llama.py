import logging
import time
import sys
from typing import Optional
from pathlib import Path
from litllama.lit_llama.utils import EmptyInitOnDevice, lazy_load, llama_model_lookup

import lightning as L
import torch
from torch import nn

from .model import Model, ModelInfo
from ..config import CHECKPOINTS_DIR
from litllama.lit_llama import LLaMA, Tokenizer
from litllama import generate as litllama_generate

logger = logging.getLogger(__name__)


class LLamaModel(Model):
    """ This class is a simple wrapper around the LLaMA model """

    def __init__(self, model_info: ModelInfo):
        super().__init__(model_info=model_info)
        self.fabric = L.Fabric(devices=1)
        self.dtype = torch.bfloat16 if self.fabric.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    def decode(self, tokens: torch.Tensor) -> str:
        txt = self.tokenizer.decode(tokens)
        print(tokens)
        logger.debug(f"Decoding with tokens: {tokens.shape}, decoded: {txt}")
        print(txt) # FIXME: Show output to STDOUT as well (ovioulsy this is not intended to be used in production)
        return txt

    def encode(self, text: str):
        """ Encode a text with a tokenizer """
        logger.debug(f"Encodeing with text: {text}")
        return self.tokenizer.encode(text, bos=True, eos=False, device=self.fabric.device)

    def _generate(self
                  , prompt_tokenized
                  , max_tokens: int = 50
                  , top_k : int = 200 
                  , temperature: float = 0.8
                  , **kwargs):
        """ Generate a text prompt with a LLaMA model """
        logger.debug(f"Generating with prompt: {prompt_tokenized.shape}, max_tokens: {max_tokens}, top_k: {top_k}, temperature: {temperature}")
        prompt_length = prompt_tokenized.size(0)
        t0 = time.perf_counter()
        assert max_tokens > 0, "Error: 'max_tokens' must be greater than 0"
        assert temperature > 0, "Error: 'temperature' must be greater than 0"
        y = litllama_generate(model=self.model_nn
                              , idx=prompt_tokenized
                              , max_new_tokens=max_tokens
                              , temperature=temperature
                              , top_k=top_k)
        t = time.perf_counter() - t0
        self.model_nn.reset_cache()
        tokens_generated = y.size(0) - prompt_length
        logger.debug(f"Time for inference: {t:.02f} sec total, tokens_generated: {tokens_generated}, speed: {tokens_generated / t:.02f} tokens/sec")
        if self.fabric.device.type == "cuda":
            logger.debug(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
        return y

    def load(self):
        """ Load a model and tokenizer """
        model_path = CHECKPOINTS_DIR / Path("lit-llama/7B/lit-llama.pth")
        self.model_nn = self.load_model(model_path)
        # Load tokenizer
        tokenizer_path = CHECKPOINTS_DIR / Path("lit-llama/tokenizer.model")
        logger.info(f"Loading toeknizer from '{tokenizer_path.absolute()}'")
        self.tokenizer = Tokenizer(tokenizer_path)

    def load_model(self, checkpoint_path: Path, quantize: Optional[str] = None) -> nn.Module:
        """
        Loads a model from a checkpoint.

        Args:
            checkpoint_path: The checkpoint path to load.
            quantize: Whether to quantize the model and using which method:
                ``"llm.int8"``: LLM.int8() mode,
                ``"gptq.int4"``: GPTQ 4-bit mode.
        """
        assert checkpoint_path.is_file(), f"Error: File '{checkpoint_path.absolute()}' not found"

        logger.info(f"Loading model from '{checkpoint_path.absolute()}'")
        t0 = time.time()
        with lazy_load(checkpoint_path) as checkpoint:
            name = llama_model_lookup(checkpoint)

            with EmptyInitOnDevice(
                    device=self.fabric.device, dtype=self.dtype, quantization_mode=quantize
            ):
                model = LLaMA.from_name(name)

            model.load_state_dict(checkpoint)
        logger.info(f"Time to load model: {time.time() - t0:.02f} seconds.")

        model.eval()
        model = self.fabric.setup_module(model)
        return model

    def load_tokenizer(tokenizer_path: Path) -> Tokenizer:
        """ Load a tokenizer from a path """
